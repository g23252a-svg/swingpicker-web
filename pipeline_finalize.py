# -*- coding: utf-8 -*-
"""pipeline_finalize.py — Stage 6: 저장 + 발송 + 검증 [v20.6.4]
═══════════════════════════════════════════════════════════════════
[v20.6.4] After-market sidecar 분리 — 추천 CSV 원본 불변 보장
 - recommend_latest.csv는 분석 시점 기준 불변
 - 시간외 가격은 aftermarket_prices_latest.csv에 별도 저장
"""
import os, logging, numpy as np, pandas as pd
from pipeline_context import PipelineContext
from shared_log import log, OUT_DIR, UTF8, ensure_dir
from collector_config import Route
from macro_filter import label_market_temp
from telegram_sender import send_telegram_auto
from validation import run_reality_check

logger = logging.getLogger(__name__)

def finalize_outputs(ctx: PipelineContext) -> None:
    from collector import make_rank_validation_report  # 아직 collector에만 있음
    df_out = ctx.df_out; trade_ymd = ctx.trade_ymd
    _am = {Route.ATTACK:1,"ATTACK":1,Route.ARMED:2,"ARMED":2,Route.WAIT:3,"WAIT":3,
           Route.NEUTRAL:4,"NEUTRAL":4,Route.OVERHEAT:5,"OVERHEAT":5,
           Route.EXIT_WARNING:6,"EXIT_WARNING":6,Route.CARRY:7,"CARRY":7}
    must_cols = [
        "LDY_RANK","종목코드","종목명","시장","업종_대분류","종가","거래대금(억원)","시가총액(억원)",
        "켈리_수량","추천금액(만원)","상태","ROUTE","ACTION_PRIORITY","IS_ACTIVE","IS_NOW_ENTRY","IS_WATCH",
        "DISPLAY_SCORE","FINAL_SCORE","STRUCT_SCORE","TIMING_SCORE","AI_SCORE","NEWS_SCORE",
        "추천매수가","손절가","추천매도가1","추천매도가2","TRIGGER","V_POWER","거래강도",
        "VWAP","POC_GAP","NEWS_REASON","TTM_SQUEEZE_CNT","Low_Trend_PCT","RSI14","이격도",
        "SCORE_REASON_TOP1","SCORE_REASON_TOP2","SCORE_RISK","ROUTE_REASON",
        "MACRO_RISK","MARKET_BREADTH"]
    for c in must_cols:
        if c not in df_out.columns: df_out[c] = np.nan
    for _cm in ["CARRY_FROM_DATE","CARRY_AGE_DAYS","IS_STALE_CARRY","STALE_PENALTY",
                "ROW_BUILD_MODE","DATA_FRESHNESS_OK"]:
        if _cm not in df_out.columns:
            if _cm == "CARRY_FROM_DATE": df_out[_cm] = np.nan
            elif _cm == "IS_STALE_CARRY": df_out[_cm] = False
            elif _cm == "ROW_BUILD_MODE": df_out[_cm] = "FRESH"
            elif _cm == "DATA_FRESHNESS_OK": df_out[_cm] = True
            else: df_out[_cm] = 0
    df_out = df_out[must_cols + [c for c in df_out.columns if c not in must_cols]]
    # Config Snapshot
    try:
        from collector_config import DEFAULT_CONFIG as _snap
        df_out["CONFIG_SNAPSHOT"]=_snap.snapshot_json(); df_out["CONFIG_VERSION"]=_snap.config_version
    except (ImportError,AttributeError) as e: logger.debug(f"CONFIG_SNAPSHOT 스킵: {e}")
    except Exception as e: logger.warning(f"⚠️ CONFIG_SNAPSHOT 오류: {e}")
    # [v20.6] macro_risk 직접 저장
    df_out["MACRO_RISK"] = ctx.macro_risk
    df_out["MARKET_BREADTH"] = ctx.breadth.get("ALL", np.nan)
    # Run Health
    _health = None
    try:
        from run_health import check_run_health, save_health
        _health = check_run_health(df_out, mcap_map=ctx.mcap_map, bench_map=ctx.bench_map,
            inv_maps=ctx.inv_maps, trade_ymd=trade_ymd)
        _health.macro_risk = ctx.macro_risk
        _health.market_breadth = ctx.breadth.get("ALL", 50.0)
        df_out = _health.inject_columns(df_out); save_health(_health, OUT_DIR, trade_ymd)
        log(_health.summary())
    except ImportError: log("ℹ️ run_health 모듈 없음")
    except Exception as e: log(f"⚠️ Run Health 실패: {e}")
    # 축 비활성 중립화 + 행동 제한
    if _health:
        _r = set(_health.reasons)
        if "NEWS_OFF" in _r: df_out["NEWS_SCORE"]=np.nan; df_out["NEWS_REASON"]="DATA_UNAVAILABLE"
        if "SECTOR_FAIL" in _r:
            for c in ["SECTOR_RANK","SECTOR_RS"]:
                if c in df_out.columns: df_out[c]=np.nan
        if "BENCH_FAIL" in _r or "BENCH_NAN" in _r:
            for _bc in ["rel_20d_%","rel_60d_%","rel_120d_%","벤치_60d_KOSPI_%","벤치_60d_KOSDAQ_%"]:
                if _bc in df_out.columns: df_out[_bc]=np.nan
        _mx = _health.max_allowed_route; _dc = 0
        if _mx != "ATTACK":
            _atk = df_out["ROUTE"]==Route.ATTACK
            if _atk.any():
                _fb = Route.ARMED if _mx=="ARMED" else Route.WAIT
                df_out.loc[_atk,"ROUTE"]=_fb; df_out.loc[_atk,"상태"]=_fb; _dc+=_atk.sum()
        if _mx == "WAIT":
            _arm = df_out["ROUTE"]==Route.ARMED
            if _arm.any(): df_out.loc[_arm,"ROUTE"]=Route.WAIT; df_out.loc[_arm,"상태"]=Route.WAIT; _dc+=_arm.sum()
        if _dc > 0:
            _cr = f"RUN_STATUS={_health.status}" if _health.status!="OK" else f"confidence={_health.confidence_score:.0f}"
            log(f"🛡️ [v20.0.2] 행동 상한 제어: {_cr} → 최대허용 {_mx}, {_dc}건 route_capped")
            df_out["IS_ACTIVE"]=df_out["ROUTE"].isin([Route.ATTACK,Route.ARMED])
            df_out["IS_NOW_ENTRY"]=df_out["ROUTE"]==Route.ATTACK
            df_out["IS_WATCH"]=df_out["ROUTE"]==Route.WAIT
            df_out["ACTION_PRIORITY"]=df_out["ROUTE"].map(_am).fillna(7).astype(int)

    # ── CSV 저장 (분석 시점 불변 원본) ──
    ensure_dir(OUT_DIR)
    op_d = os.path.join(OUT_DIR, f"recommend_{trade_ymd}{f'_{ctx.tag}' if ctx.tag else ''}.csv")
    op_l = os.path.join(OUT_DIR, "recommend_latest.csv")
    # 종목명 오염 복구
    if "종목명" in df_out.columns and "종목코드" in df_out.columns:
        df_out["종목명"] = df_out["종목명"].astype(str)
        _cm2 = df_out["종목명"].str.match(r'^\d+$'); _cc = _cm2.sum()
        if _cc > 0:
            if ctx.name_map:
                df_out.loc[_cm2,"종목명"] = df_out.loc[_cm2,"종목코드"].astype(str).str.zfill(6).map(ctx.name_map).fillna(df_out.loc[_cm2,"종목명"])
            _sc2 = df_out["종목명"].str.match(r'^\d+$')
            if _sc2.sum() > 0:
                _sp = os.path.join(OUT_DIR, "price_snapshot_latest.csv")
                if os.path.exists(_sp):
                    try:
                        _sn = pd.read_csv(_sp, dtype={"종목코드":str}, usecols=["종목코드","종목명"])
                        _sm = dict(zip(_sn["종목코드"].str.zfill(6), _sn["종목명"]))
                        df_out.loc[_sc2,"종목명"] = df_out.loc[_sc2,"종목코드"].astype(str).str.zfill(6).map(_sm).fillna(df_out.loc[_sc2,"종목명"])
                        ctx.name_map.update({c:n for c,n in _sm.items() if c!=n})
                    except Exception as _e: log(f"⚠️ snapshot 폴백 실패: {_e}")
            _fc = df_out["종목명"].str.match(r'^\d+$').sum(); _fx = _cc - _fc
            if _fx > 0: log(f"🔧 종목명 복구: {_fx}/{_cc}건")
            if _fc > 0:
                _pat = r'^\d+$'
                _remain = df_out.loc[df_out['종목명'].str.match(_pat), '종목코드'].tolist()[:5]
                log(f"⚠️ 미복구 {_fc}건: {_remain}")
    df_out.to_csv(op_d, index=False, encoding=UTF8)
    df_out.to_csv(op_l, index=False, encoding=UTF8)
    log(f"💾 저장 완료 ({len(df_out)}건) → {op_d}")

    # ── [v20.6.3] run_meta JSON sidecar ──
    try:
        import json as _json
        _meta = {
            "trade_ymd": trade_ymd,
            "macro_risk": ctx.macro_risk,
            "macro_msg": ctx.macro_msg,
            "market_breadth": ctx.breadth.get("ALL", np.nan),
            "pass_ebs": ctx.pass_ebs,
            "rec_limit": ctx.rec_limit,
            "n_stocks": len(df_out),
            "run_status": _health.status if _health else "UNKNOWN",
            "confidence_score": _health.confidence_score if _health else 0.0,
            "max_allowed_route": _health.max_allowed_route if _health else "ATTACK",
            "scoring_axes": df_out["SCORING_AXES"].iloc[0] if "SCORING_AXES" in df_out.columns else "",
            "w_struct": float(df_out["W_STRUCT"].iloc[0]) if "W_STRUCT" in df_out.columns else 0.0,
            "w_timing": float(df_out["W_TIMING"].iloc[0]) if "W_TIMING" in df_out.columns else 0.0,
            "w_ai": float(df_out["W_AI"].iloc[0]) if "W_AI" in df_out.columns else 0.0,
        }
        _meta_d = os.path.join(OUT_DIR, f"run_meta_{trade_ymd}.json")
        _meta_l = os.path.join(OUT_DIR, "run_meta_latest.json")
        for _mp in [_meta_d, _meta_l]:
            with open(_mp, 'w', encoding='utf-8') as _mf:
                _json.dump(_meta, _mf, ensure_ascii=False, indent=2, default=str)
        log(f"📋 run_meta 저장 완료 → {_meta_d}")
    except Exception as _me:
        logger.warning(f"⚠️ run_meta 저장 실패 (무해): {_me}")

    # ══════════════════════════════════════════════════════════
    #  [v20.6.4] After-market → sidecar 파일로 분리
    #  recommend_latest.csv는 절대 수정하지 않음 (원본 보존)
    # ══════════════════════════════════════════════════════════
    try:
        from naver_aftermarket import fetch_after_market_prices_sidecar
        _snl = os.path.join(OUT_DIR, 'price_snapshot_latest.csv')
        _sidecar_path = os.path.join(OUT_DIR, 'aftermarket_prices_latest.csv')
        _ac = fetch_after_market_prices_sidecar(op_l, _sidecar_path, _snl)
        if _ac > 0:
            log(f'After-market sidecar: {_ac} stocks → {_sidecar_path}')
        else:
            log('After-market: no changes')
    except ImportError:
        # 폴백: 기존 방식 (naver_aftermarket 구버전)
        try:
            from naver_aftermarket import update_csv_with_aftermarket
            _snl = os.path.join(OUT_DIR, 'price_snapshot_latest.csv')
            _ac = update_csv_with_aftermarket(op_l, _snl)
            if _ac > 0:
                import shutil; shutil.copy2(op_l, op_d)
                log(f'After-market (legacy): {_ac} stocks updated')
            else:
                log('After-market: no changes')
        except ImportError:
            log('naver_aftermarket not found')
        except Exception as e:
            log(f'After-market failed: {e}')
    except Exception as e:
        log(f'After-market sidecar failed: {e}')

    # 종목명 매핑
    try:
        _sp2 = os.path.join(OUT_DIR, "price_snapshot_latest.csv")
        if os.path.exists(_sp2):
            _nd = pd.read_csv(_sp2, dtype={"종목코드":str}, usecols=["종목코드","종목명"])
            _nd["종목코드"]=_nd["종목코드"].str.zfill(6); _nd=_nd.drop_duplicates("종목코드")
        else: _nd = df_out[["종목코드","종목명"]].drop_duplicates("종목코드")
        if ctx.name_map:
            _ex = [{"종목코드":c,"종목명":n} for c,n in ctx.name_map.items() if c not in _nd["종목코드"].values and c!=n and not n.isdigit()]
            if _ex: _nd = pd.concat([_nd, pd.DataFrame(_ex)], ignore_index=True)
        _nd = _nd[_nd["종목명"].astype(str)!=_nd["종목코드"].astype(str)]
        _np = os.path.join(OUT_DIR, "krx_names_latest.csv")
        _nd.to_csv(_np, index=False, encoding=UTF8)
        log(f"📋 종목명 매핑 저장: {len(_nd)}건 → {_np}")
    except Exception as e: log(f"⚠️ 종목명 매핑 실패: {e}")
    # DB
    try:
        from db_utils import get_db; get_db().save_recommendations(df_out, trade_ymd)
    except Exception as e: log(f"⚠️ DB 저장 실패: {e}")
    # Reality Check + Rank Validation
    run_reality_check(OUT_DIR, trade_ymd)
    make_rank_validation_report(OUT_DIR, asof_ymd=trade_ymd, methods=["DISPLAY_SCORE","FINAL_SCORE","AI_SCORE"])
    # 텔레그램
    if ctx.enable_telegram:
        mkt = label_market_temp(ctx.breadth.get("ALL", np.nan))
        st = f"🌡 {mkt} (Breadth: {ctx.breadth.get('ALL',0)}%)"
        if ctx.macro_msg: st += f"\n{ctx.macro_msg}"
        if "SECTOR_RANK" in df_out.columns:
            ts2 = df_out.sort_values("SECTOR_RS",ascending=False)["업종_대분류"].unique()[:2]
            st += f"\n🚀 주도: {' '.join(ts2)}"
        send_telegram_auto(df_out, trade_ymd, market_summary=st, limit_count=ctx.rec_limit)
    else: log("✉️ 텔레그램 발송 생략")
    # 자동 캘리브레이션
    try:
        from auto_backtest import auto_calibrate
        cs = auto_calibrate(OUT_DIR, trade_ymd)
        log(f"📊 캘리브레이션: {cs.get('n_trades',0)}건, 승률={cs.get('overall_winrate',0):.1%}")
    except Exception as e: log(f"⚠️ 자동 캘리브레이션 스킵: {e}")
    # 포지션
    try:
        from position_tracker import track_open_positions, register_from_recommendations
        register_from_recommendations(OUT_DIR, df_out, trade_ymd, top_n=ctx.rec_limit)
        tr = track_open_positions(OUT_DIR, trade_ymd)
        log(f"📍 포지션: 체크={tr.get('checked',0)}, 이벤트={tr.get('events',0)}, 청산={tr.get('closed',0)}")
    except Exception as e: log(f"⚠️ 포지션 트래킹 스킵: {e}")
    # 브리핑
    try:
        from daily_briefing import generate_daily_briefing
        br = generate_daily_briefing(OUT_DIR, trade_ymd, df_out)
        if br["count"]>0: log(f"📝 일일 브리핑: {br['count']}종목 [{', '.join(br.get('names',[]))}]")
        else: log("📝 일일 브리핑: 대상 없음")
    except Exception as e: log(f"⚠️ 일일 브리핑 스킵: {e}")
    ctx.df_out = df_out
