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


# [v3.7.27 추가 · v3.7.29 강화] CONFIG_SNAPSHOT 로드 유틸
# Single source of truth: config는 JSON 파일에서만 읽는다.
# CSV에는 CONFIG_VERSION 문자열만 남기므로, 전체 snapshot이 필요하면 이 함수 사용.
def load_config_snapshot(trade_ymd: str = None) -> dict:
    """CONFIG_SNAPSHOT을 JSON 파일에서 로드 (fallback 체인 적용).

    Args:
        trade_ymd: YYYYMMDD 문자열. None이면 latest 파일 사용.

    Returns:
        dict: 설정 스냅샷. 모든 fallback 실패 시 빈 dict (참조 코드가 안전하게 계속 진행 가능).

    Fallback 순서:
      1) data/config_snapshot_{trade_ymd}.json (지정일)
      2) data/config_snapshot_latest.json      (최신)
      3) collector_config.DEFAULT_CONFIG.snapshot_json() (런타임)
      4) {} 빈 dict

    예전 CSV 호환 코드를 바꿀 때 사용:
        # Before (v3.7.26):
        #   snapshot = json.loads(df.iloc[0]["CONFIG_SNAPSHOT"])
        # After (v3.7.27+):
        #   from pipeline_finalize import load_config_snapshot
        #   snapshot = load_config_snapshot(trade_ymd)
    """
    import json
    from pathlib import Path

    # 1) 지정일 파일
    if trade_ymd:
        try:
            path = Path(OUT_DIR) / f"config_snapshot_{trade_ymd}.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug(f"CONFIG_SNAPSHOT dated 로드 실패 ({trade_ymd}): {e}")

    # 2) latest alias
    try:
        path_latest = Path(OUT_DIR) / "config_snapshot_latest.json"
        if path_latest.exists():
            return json.loads(path_latest.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"CONFIG_SNAPSHOT latest 로드 실패: {e}")

    # 3) 런타임 재생성 (collector_config에서 직접)
    try:
        from collector_config import DEFAULT_CONFIG as _snap
        return json.loads(_snap.snapshot_json())
    except Exception as e:
        logger.debug(f"CONFIG_SNAPSHOT 런타임 생성 실패: {e}")

    # 4) 빈 dict
    logger.info("CONFIG_SNAPSHOT 사용 불가 — 빈 dict 반환 (참조 코드는 계속 진행)")
    return {}


# ═══════════════════════════════════════════════════
#  [v22] finalize_sort SSOT + adaptive IS_NOW_ENTRY
#  ─ 참조: SwingPicker_v22_Final_Consolidated.md §2.2.6
# ═══════════════════════════════════════════════════

# SORT_SPEC — 8축 정렬의 단일 소스
# TOP_PICK × IS_NOW_ENTRY × ROUTE_PRIORITY × ELITE × RR × BALANCE × ENTRY_GAP × DISPLAY_SCORE
_SORT_ROUTE_PRIORITY = {
    "ATTACK": 1, "ARMED": 2, "WAIT": 3, "NEUTRAL": 4,
    "OVERHEAT": 5, "EXIT_WARNING": 6, "CARRY": 7,
}


def _compute_is_now_entry_vectorized(df: pd.DataFrame) -> pd.Series:
    """IS_NOW_ENTRY — shared_utils.compute_is_now_entry 벡터 적용.
    
    ATR_Pct(decimal, ml_engine) 우선, 없으면 ATR_PCT(percentage, stop_logic) 허용.
    """
    try:
        from shared_utils import compute_is_now_entry as _cine
    except ImportError:
        # v22 신규 함수 미탑재 환경 — fallback: ROUTE==ATTACK
        route = df.get("ROUTE", pd.Series("", index=df.index))
        return (route.isin(["ATTACK"])).astype(int)
    
    close = pd.to_numeric(df.get("종가", 0), errors="coerce").fillna(0)
    entry = pd.to_numeric(df.get("추천매수가", 0), errors="coerce").fillna(0)
    # ATR_Pct(decimal) 우선, ATR_PCT(percentage)도 허용 — 내부 정규화
    atr = df.get("ATR_Pct", df.get("ATR_PCT", pd.Series(0.02, index=df.index)))
    mcap = pd.to_numeric(df.get("시가총액(억원)", 0), errors="coerce").fillna(0)
    
    return pd.Series(
        [_cine(c, e, a, m) for c, e, a, m in zip(close, entry, atr, mcap)],
        index=df.index,
        dtype=int,
    )


def finalize_sort(df: pd.DataFrame) -> pd.DataFrame:
    """[v22] SORT_SPEC — 8축 정렬 SSOT.
    
    정렬 우선순위 (내림차순 기준, 낮은 ROUTE_PRIORITY가 먼저):
      1. TOP_PICK (1 먼저)
      2. IS_NOW_ENTRY (1 먼저, adaptive 기반)
      3. ROUTE_PRIORITY (낮을수록 먼저: ATTACK=1 → CARRY=7)
      4. ELITE_SCORE (높을수록)
      5. RR_NOW_TP1 (높을수록)
      6. BALANCE_SCORE (높을수록)
      7. ENTRY_GAP_PCT (낮을수록)
      8. DISPLAY_SCORE (높을수록)
    
    다른 단계에서 IS_NOW_ENTRY를 ROUTE==ATTACK으로 세팅했어도 
    여기서 adaptive로 덮어쓴다. 순서가 SSOT.
    """
    df = df.copy()

    # IS_NOW_ENTRY adaptive 재계산 (항상 덮어쓰기 — 이전 단계의 단순 ROUTE==ATTACK 치환)
    df["IS_NOW_ENTRY"] = _compute_is_now_entry_vectorized(df)

    # ROUTE_PRIORITY (정렬용 임시 컬럼)
    route = df.get("ROUTE", pd.Series("", index=df.index)).astype(str)
    df["_ROUTE_PRIORITY"] = route.map(_SORT_ROUTE_PRIORITY).fillna(99).astype(int)

    # 정렬 축 모두 존재 확인 (없으면 중립 값으로 채움)
    for col, default in [
        ("TOP_PICK", 0), ("IS_NOW_ENTRY", 0),
        ("ELITE_SCORE", 0), ("RR_NOW_TP1", 0),
        ("BALANCE_SCORE", 0), ("ENTRY_GAP_PCT", 99),
        ("DISPLAY_SCORE", 0),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # SORT_SPEC 적용
    df = df.sort_values(
        by=["TOP_PICK", "IS_NOW_ENTRY", "_ROUTE_PRIORITY", "ELITE_SCORE",
            "RR_NOW_TP1", "BALANCE_SCORE", "ENTRY_GAP_PCT", "DISPLAY_SCORE"],
        ascending=[False, False, True, False, False, False, True, False],
        kind="mergesort",   # 안정 정렬 (동점 시 원래 순서 유지)
    ).reset_index(drop=True)

    # 임시 컬럼 제거
    df = df.drop(columns=["_ROUTE_PRIORITY"], errors="ignore")
    return df


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
        "ELITE_SCORE","AXIS_MEAN","AXIS_GAP","BALANCE_SCORE","RR_NOW_TP1","ENTRY_GAP_PCT","ELITE_REASON","TOP_PICK",
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
    # ══════════════════════════════════════════════════
    #  CONFIG_SNAPSHOT 저장 (v3.7.27에서 JSON 분리 · v3.7.29에서 migration 완료)
    # ══════════════════════════════════════════════════
    # 정책 — single source of truth:
    #   · CSV 행 데이터:  경량 (CONFIG_VERSION 문자열만)
    #   · JSON 파일:     config 스냅샷 전용
    #                    data/config_snapshot_{trade_ymd}.json (일자별)
    #                    data/config_snapshot_latest.json      (최신 alias)
    # 읽기:
    #   · 모든 참조 코드는 load_config_snapshot(trade_ymd) 헬퍼를 사용한다.
    #   · 예: snapshot = load_config_snapshot("20260420")
    #   · Fallback: 파일이 없으면 빈 dict 반환 (예외 없음) — 참조 코드가 그냥 계속 돌 수 있도록.
    # 주변 참조 (v3.7.29 기준 전부 이관 완료):
    #   · test_shadow_analyze.py → SKIP_KEYS에 포함 (비교에서 제외)
    try:
        from collector_config import DEFAULT_CONFIG as _snap
        # CSV에는 버전 문자열만 (작은 값, 호환성 유지)
        df_out["CONFIG_VERSION"] = _snap.config_version
        # 전체 스냅샷은 별도 JSON 파일로 — 일자별 1회 덮어쓰기
        try:
            from pathlib import Path as _P
            _snap_path = _P(OUT_DIR) / f"config_snapshot_{trade_ymd}.json"
            _snap_latest = _P(OUT_DIR) / "config_snapshot_latest.json"
            _snap_json_str = _snap.snapshot_json()
            _snap_path.write_text(_snap_json_str, encoding="utf-8")
            _snap_latest.write_text(_snap_json_str, encoding="utf-8")
            logger.info(f"✅ CONFIG_SNAPSHOT → {_snap_path.name}")
        except Exception as _ef:
            logger.warning(f"⚠️ CONFIG_SNAPSHOT JSON 저장 실패: {_ef}")
    except (ImportError, AttributeError) as e:
        logger.debug(f"CONFIG_SNAPSHOT 스킵 (구성 없음): {e}")
    except Exception as e:
        logger.warning(f"⚠️ CONFIG_SNAPSHOT 오류: {e}")
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
            # [v22] IS_NOW_ENTRY는 finalize_sort에서 adaptive로 재계산되므로 여기선 건드리지 않음
            df_out["IS_WATCH"]=df_out["ROUTE"]==Route.WAIT
            df_out["ACTION_PRIORITY"]=df_out["ROUTE"].map(_am).fillna(7).astype(int)

            # [v22] route cap 이후 TOP_PICK positive gate 재적용
            # ATTACK→WAIT/ARMED→WAIT capped 종목은 TOP_PICK에서 탈락시켜야
            # "TOP_PICK=1 but ROUTE=WAIT" 누출 재발 차단.
            _active_mask = df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED, "ATTACK", "ARMED"])
            _leaked = (df_out.get("TOP_PICK", pd.Series(0, index=df_out.index)) == 1) & (~_active_mask)
            if _leaked.any():
                _leak_n = int(_leaked.sum())
                df_out.loc[_leaked, "TOP_PICK"] = 0
                if "TOP_PICK_TYPE" in df_out.columns:
                    df_out.loc[_leaked, "TOP_PICK_TYPE"] = ""
                log(f"🎯 [v22] route cap 후 TOP_PICK 정리: {_leak_n}건 탈락")
            # [v22] route cap 발생 시 누출 여부와 무관하게 재정렬
            # (ROUTE_PRIORITY 바뀌었으므로 순서 영향)
            # finalize_sort는 아래 CSV 저장 직전에 한 번 더 호출되지만,
            # 여기서 한 번 정리해두는 게 중간 단계 일관성에 좋음.
            try:
                df_out = finalize_sort(df_out)
            except Exception as e:
                logger.warning(f"⚠️ route cap 후 재정렬 실패 (무해): {e}")

    # ── [v21.1] ACTION_PRIORITY 항상 재계산 (SSOT 보장) ──
    df_out["ACTION_PRIORITY"] = df_out["ROUTE"].map(_am).fillna(7).astype(int)

    # ── [v22] 최종 정렬 SSOT 적용 (SORT_SPEC) ──
    # TOP_PICK × IS_NOW_ENTRY(adaptive) × ROUTE × ELITE × RR × BALANCE × ENTRY_GAP × DISPLAY_SCORE
    # 정렬 직후 LDY_RANK 재부여 (stale 방지)
    try:
        df_out = finalize_sort(df_out)
        df_out["LDY_RANK"] = np.arange(1, len(df_out) + 1)
        log(f"🎯 [v22] SORT_SPEC 적용 완료 (TOP_PICK × IS_NOW_ENTRY × ELITE × RR ...)")
    except Exception as e:
        logger.warning(f"⚠️ finalize_sort 실패 (무해 — 기존 순서 유지): {e}")

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
        # sidecar 함수 없으면 시간외 업데이트 스킵 (원본 보존 원칙)
        log('After-market: sidecar 함수 없음 — 스킵 (recommend 원본 보존)')
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
    make_rank_validation_report(OUT_DIR, asof_ymd=trade_ymd, methods=["ELITE_SCORE","DISPLAY_SCORE","FINAL_SCORE","AI_SCORE"])
    # [v22.3] monotonicity_report 인라인 생성 — daily_briefing.py 별도 실행 의존성 제거
    # 평가 피드백: "ZIP 기준 최신 검증 리포트가 항상 따라오는 구조" 보장
    try:
        from daily_briefing import generate_monotonicity_report
        _mono = generate_monotonicity_report(OUT_DIR, trade_ymd)
        _mono_status = _mono.get("ci_hard", [{}])[0].get("status", "?") if _mono.get("ci_hard") else "OK"
        log(f"📊 [v22.3] monotonicity_report → {trade_ymd} (status={_mono_status})")
    except ImportError:
        log("ℹ️ daily_briefing 모듈 없음 — monotonicity_report SKIP")
    except Exception as e:
        log(f"⚠️ monotonicity_report 생성 실패: {e}")
    # [v21.2+v22] TOP_PICK 검증 리포트 — 0건에도 latest 갱신 (CI 오독 차단)
    try:
        import json as _json2
        _tp_mask = df_out.get("TOP_PICK", pd.Series(0, index=df_out.index)).astype(int) == 1
        _tp_count = int(_tp_mask.sum())
        _tp_path = os.path.join(OUT_DIR, f"top_pick_validation_{trade_ymd}.json")
        _tp_latest = os.path.join(OUT_DIR, "top_pick_validation_latest.json")

        if _tp_count > 0:
            _tp_df = df_out[_tp_mask].copy()
            # [v22] AGGRESSIVE/STABLE 분리 집계
            _by_type = (_tp_df["TOP_PICK_TYPE"].value_counts().to_dict()
                        if "TOP_PICK_TYPE" in _tp_df.columns else {})
            _tp_summary = {
                "trade_ymd": trade_ymd,
                "top_pick_count": _tp_count,
                "top_pick_by_type": _by_type,
                "avg_elite": round(float(_tp_df["ELITE_SCORE"].mean()), 1),
                "avg_rr": round(float(_tp_df["RR_NOW_TP1"].mean()), 2),
                # [v22.3.1] 최소 RR + RR<1 카운트 — 평균은 위장 가능, 최소가 진실
                "min_rr": round(float(_tp_df["RR_NOW_TP1"].min()), 2),
                "rr_lt_1_count": int((pd.to_numeric(_tp_df["RR_NOW_TP1"], errors="coerce").fillna(0) < 1.0).sum()),
                "avg_balance": round(float(_tp_df["BALANCE_SCORE"].mean()), 1),
                "avg_win_rate": round(float(_tp_df["EST_WIN_RATE"].mean()), 3),
                "est_win_rate_method": (_tp_df["EST_WIN_RATE_METHOD"].iloc[0]
                                         if "EST_WIN_RATE_METHOD" in _tp_df.columns else "UNKNOWN"),
                "est_win_rate_mode": (_tp_df["EST_WIN_RATE_MODE"].iloc[0]
                                       if "EST_WIN_RATE_MODE" in _tp_df.columns else "UNKNOWN"),
                "est_win_rate_n": (int(_tp_df["EST_WIN_RATE_N"].iloc[0])
                                    if "EST_WIN_RATE_N" in _tp_df.columns else 0),
                "routes": _tp_df["ROUTE"].value_counts().to_dict(),
                "picks": _tp_df[[
                    c for c in [
                        "종목코드", "종목명",
                        "TOP_PICK_TYPE",
                        "ELITE_SCORE", "RR_NOW_TP1", "BALANCE_SCORE",
                        "ENTRY_GAP_PCT", "TP1_PCT",
                        "ROUTE",
                        "EST_WIN_RATE", "EST_WIN_RATE_METHOD",
                        "EST_WIN_RATE_MODE", "EST_WIN_RATE_N",
                    ] if c in _tp_df.columns
                ]].to_dict("records"),
            }
            _type_msg = (f" (AGGR={_by_type.get('AGGRESSIVE',0)}, "
                         f"STBL={_by_type.get('STABLE',0)})" if _by_type else "")
            log(f"🏆 TOP_PICK 검증: {_tp_count}종목{_type_msg} → {_tp_path}")
        else:
            # [v22] 0건 날에도 latest 갱신 — stale 방지
            _meta_method = "NONE"
            _meta_mode = "NONE"
            _meta_n = 0
            if "EST_WIN_RATE_METHOD" in df_out.columns and len(df_out) > 0:
                _meta_method = str(df_out["EST_WIN_RATE_METHOD"].iloc[0])
            if "EST_WIN_RATE_MODE" in df_out.columns and len(df_out) > 0:
                _meta_mode = str(df_out["EST_WIN_RATE_MODE"].iloc[0])
            if "EST_WIN_RATE_N" in df_out.columns and len(df_out) > 0:
                try:
                    _meta_n = int(df_out["EST_WIN_RATE_N"].iloc[0])
                except Exception:
                    _meta_n = 0
            _tp_summary = {
                "trade_ymd": trade_ymd,
                "top_pick_count": 0,
                "top_pick_by_type": {},
                "avg_elite": None,
                "avg_rr": None,
                # [v22.3.1] 0건 케이스에도 동일 필드 — null 안정성
                "min_rr": None,
                "rr_lt_1_count": 0,
                "avg_balance": None,
                "avg_win_rate": None,
                "est_win_rate_method": _meta_method,
                "est_win_rate_mode": _meta_mode,
                "est_win_rate_n": _meta_n,
                "routes": {},
                "picks": [],
            }
            log(f"🏆 TOP_PICK: 0종목 (게이트 미통과) — latest.json 갱신")

        for _p in [_tp_path, _tp_latest]:
            with open(_p, 'w', encoding='utf-8') as _f:
                _json2.dump(_tp_summary, _f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.warning(f"⚠️ TOP_PICK 검증 실패: {e}")
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
    # [v21.3] 조합 최적화
    try:
        from combo_optimizer import run_combo_optimization
        opt = run_combo_optimization(OUT_DIR, horizon=3, min_samples=10)
        if opt and opt.get("best"):
            b = opt["best"]
            log(f"🎯 최적 조합: S≥{b['S_min']} T≥{b['T_min']} AI≥{b['AI_min']} | 승률 {b['win_rate']}%")
    except Exception as e:
        log(f"⚠️ 조합 최적화 스킵: {e}")
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
