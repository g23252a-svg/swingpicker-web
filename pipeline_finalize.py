# -*- coding: utf-8 -*-
"""
pipeline_finalize.py — Stage 6: 저장 + 발송 + 검증
═══════════════════════════════════════════════════════
[v20.1] collector.py 분해 — must_cols → Run Health → CSV → After-market → 텔레그램 → 포지션
"""

import os
import logging
import numpy as np
import pandas as pd

from pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


def finalize_outputs(ctx: PipelineContext) -> None:
    """Stage 6: 최종 저장, Run Health, After-market, 텔레그램, 포지션"""
    from collector import (
        log, Route, OUT_DIR, UTF8, ensure_dir,
        send_telegram_auto, run_reality_check, make_rank_validation_report,
        label_market_temp,
    )

    df_out = ctx.df_out
    trade_ymd = ctx.trade_ymd

    _action_map = {
        Route.ATTACK: 1, "ATTACK": 1,
        Route.ARMED: 2, "ARMED": 2,
        Route.WAIT: 3, "WAIT": 3,
        Route.NEUTRAL: 4, "NEUTRAL": 4,
        Route.OVERHEAT: 5, "OVERHEAT": 5,
        Route.EXIT_WARNING: 6, "EXIT_WARNING": 6,
        Route.CARRY: 7, "CARRY": 7,
    }

    # ── must_cols 방어 ──
    must_cols = [
        "LDY_RANK", "종목코드", "종목명", "시장", "업종_대분류", "종가", "거래대금(억원)", "시가총액(억원)",
        "켈리_수량", "추천금액(만원)",
        "상태", "ROUTE", "ACTION_PRIORITY", "IS_ACTIVE", "IS_NOW_ENTRY", "IS_WATCH",
        "DISPLAY_SCORE", "FINAL_SCORE", "STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE", "NEWS_SCORE",
        "추천매수가", "손절가", "추천매도가1", "추천매도가2", "TRIGGER", "V_POWER", "거래강도",
        "VWAP", "POC_GAP", "NEWS_REASON", "TTM_SQUEEZE_CNT", "Low_Trend_PCT", "RSI14", "이격도",
    ]
    for c in must_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan

    for _cm in ["CARRY_FROM_DATE", "CARRY_AGE_DAYS", "IS_STALE_CARRY", "STALE_PENALTY"]:
        if _cm not in df_out.columns:
            df_out[_cm] = np.nan if _cm == "CARRY_FROM_DATE" else (False if _cm == "IS_STALE_CARRY" else 0)

    df_out = df_out[must_cols + [c for c in df_out.columns if c not in must_cols]]

    # ── Config Snapshot ──
    try:
        from collector_config import DEFAULT_CONFIG as _snap_cfg
        df_out["CONFIG_SNAPSHOT"] = _snap_cfg.snapshot_json()
        df_out["CONFIG_VERSION"] = _snap_cfg.config_version
    except (ImportError, AttributeError) as e:
        logger.debug(f"CONFIG_SNAPSHOT 생성 스킵: {e}")
    except Exception as e:
        logger.warning(f"⚠️ CONFIG_SNAPSHOT 예기치 않은 오류: {type(e).__name__}: {e}")

    # ── Run Health 진단 ──
    _health = None
    try:
        from run_health import check_run_health, save_health
        _health = check_run_health(
            df_out, mcap_map=ctx.mcap_map, bench_map=ctx.bench_map,
            inv_maps=ctx.inv_maps, trade_ymd=trade_ymd,
        )
        df_out = _health.inject_columns(df_out)
        save_health(_health, OUT_DIR, trade_ymd)
        log(_health.summary())
    except ImportError:
        log("ℹ️ run_health 모듈 없음, 건강 진단 스킵")
    except Exception as _rh_err:
        log(f"⚠️ Run Health 진단 실패 (무시): {_rh_err}")

    # ── 축 비활성 중립화 ──
    if _health:
        _reasons = set(_health.reasons)
        if "NEWS_OFF" in _reasons:
            df_out["NEWS_SCORE"] = np.nan
            df_out["NEWS_REASON"] = "DATA_UNAVAILABLE"
        if "SECTOR_FAIL" in _reasons:
            df_out["SECTOR_RANK"] = np.nan
            df_out["SECTOR_RS"] = np.nan
        if "BENCH_FAIL" in _reasons or "BENCH_NAN" in _reasons:
            for _bc in ["rel_20d_%", "rel_60d_%", "rel_120d_%",
                        "벤치_60d_KOSPI_%", "벤치_60d_KOSDAQ_%"]:
                if _bc in df_out.columns:
                    df_out[_bc] = np.nan

        # ── 신뢰도 기반 행동 제한 ──
        _max_route = _health.max_allowed_route
        _demote_count = 0

        if _max_route != "ATTACK":
            _attack_mask = df_out["ROUTE"] == Route.ATTACK
            if _attack_mask.any():
                _fallback_route = Route.ARMED if _max_route == "ARMED" else Route.WAIT
                df_out.loc[_attack_mask, "ROUTE"] = _fallback_route
                df_out.loc[_attack_mask, "상태"] = _fallback_route
                _demote_count += _attack_mask.sum()

        if _max_route == "WAIT":
            _armed_mask = df_out["ROUTE"] == Route.ARMED
            if _armed_mask.any():
                df_out.loc[_armed_mask, "ROUTE"] = Route.WAIT
                df_out.loc[_armed_mask, "상태"] = Route.WAIT
                _demote_count += _armed_mask.sum()

        if _demote_count > 0:
            _cap_reason = f"RUN_STATUS={_health.status}" if _health.status != "OK" else f"confidence={_health.confidence_score:.0f}"
            log(f"🛡️ [v20.0.2] 행동 상한 제어: {_cap_reason} → "
                f"최대허용 {_max_route}, {_demote_count}건 route_capped_by_run_status")
            df_out["IS_ACTIVE"] = df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED])
            df_out["IS_NOW_ENTRY"] = df_out["ROUTE"] == Route.ATTACK
            df_out["IS_WATCH"] = df_out["ROUTE"] == Route.WAIT
            df_out["ACTION_PRIORITY"] = df_out["ROUTE"].map(_action_map).fillna(7).astype(int)

    # ── CSV 저장 ──
    ensure_dir(OUT_DIR)
    out_path_dated = os.path.join(OUT_DIR, f"recommend_{trade_ymd}{f'_{ctx.tag}' if ctx.tag else ''}.csv")
    out_path_latest = os.path.join(OUT_DIR, "recommend_latest.csv")

    # 종목명 오염 복구
    if "종목명" in df_out.columns and "종목코드" in df_out.columns:
        df_out["종목명"] = df_out["종목명"].astype(str)
        _corrupt_mask = df_out["종목명"].str.match(r'^\d+$')
        _corrupt_cnt = _corrupt_mask.sum()
        if _corrupt_cnt > 0:
            if ctx.name_map:
                df_out.loc[_corrupt_mask, "종목명"] = (
                    df_out.loc[_corrupt_mask, "종목코드"].astype(str).str.zfill(6)
                    .map(ctx.name_map).fillna(df_out.loc[_corrupt_mask, "종목명"])
                )
            _still_corrupt = df_out["종목명"].str.match(r'^\d+$')
            if _still_corrupt.sum() > 0:
                _snap_path = os.path.join(OUT_DIR, "price_snapshot_latest.csv")
                if os.path.exists(_snap_path):
                    try:
                        _snap = pd.read_csv(_snap_path, dtype={"종목코드": str}, usecols=["종목코드", "종목명"])
                        _snap_map = dict(zip(_snap["종목코드"].str.zfill(6), _snap["종목명"]))
                        df_out.loc[_still_corrupt, "종목명"] = (
                            df_out.loc[_still_corrupt, "종목코드"].astype(str).str.zfill(6)
                            .map(_snap_map).fillna(df_out.loc[_still_corrupt, "종목명"])
                        )
                        ctx.name_map.update({c: n for c, n in _snap_map.items() if c != n})
                    except Exception as _e:
                        log(f"⚠️ price_snapshot 폴백 실패: {_e}")
            _final_corrupt = df_out["종목명"].str.match(r'^\d+$').sum()
            _fixed = _corrupt_cnt - _final_corrupt
            if _fixed > 0:
                log(f"🔧 종목명 최종 복구: {_fixed}/{_corrupt_cnt}건 (잔여 오염: {_final_corrupt}건)")
            if _final_corrupt > 0:
                _remain_codes = df_out.loc[df_out["종목명"].str.match(r'^\d+$'), "종목코드"].tolist()[:5]
                log(f"⚠️ 종목명 미복구 {_final_corrupt}건: {_remain_codes}")

    df_out.to_csv(out_path_dated, index=False, encoding=UTF8)
    df_out.to_csv(out_path_latest, index=False, encoding=UTF8)
    log(f"💾 저장 완료 ({len(df_out)}건) → {out_path_dated}")

    # ── After-market ──
    try:
        from naver_aftermarket import update_csv_with_aftermarket
        _snap_latest = os.path.join(OUT_DIR, 'price_snapshot_latest.csv')
        _after_cnt = update_csv_with_aftermarket(out_path_latest, _snap_latest)
        if _after_cnt > 0:
            import shutil as _sh
            _sh.copy2(out_path_latest, out_path_dated)
            log('After-market: %d stocks updated' % _after_cnt)
        else:
            log('After-market: no changes')
    except ImportError:
        log('naver_aftermarket.py not found - skipped')
    except Exception as e:
        log('After-market failed: ' + str(e))

    # ── 종목명 매핑 저장 ──
    try:
        _snap_path = os.path.join(OUT_DIR, "price_snapshot_latest.csv")
        if os.path.exists(_snap_path):
            _names_df = pd.read_csv(_snap_path, dtype={"종목코드": str}, usecols=["종목코드", "종목명"])
            _names_df["종목코드"] = _names_df["종목코드"].str.zfill(6)
            _names_df = _names_df.drop_duplicates("종목코드")
        else:
            _names_df = df_out[["종목코드", "종목명"]].drop_duplicates("종목코드")
        if ctx.name_map:
            _extra = [{"종목코드": c, "종목명": n} for c, n in ctx.name_map.items()
                      if c not in _names_df["종목코드"].values and c != n and not n.isdigit()]
            if _extra:
                _names_df = pd.concat([_names_df, pd.DataFrame(_extra)], ignore_index=True)
        _names_df = _names_df[_names_df["종목명"].astype(str) != _names_df["종목코드"].astype(str)]
        _names_path = os.path.join(OUT_DIR, "krx_names_latest.csv")
        _names_df.to_csv(_names_path, index=False, encoding=UTF8)
        log(f"📋 종목명 매핑 저장: {len(_names_df)}건 → {_names_path}")
    except Exception as e:
        log(f"⚠️ 종목명 매핑 파일 저장 실패: {e}")

    # ── DB 저장 ──
    try:
        from db_utils import get_db
        db = get_db()
        db.save_recommendations(df_out, trade_ymd)
    except Exception as e:
        log(f"⚠️ DB 저장 실패: {e}")

    # ── Reality Check + Rank Validation ──
    run_reality_check(OUT_DIR, trade_ymd)
    make_rank_validation_report(OUT_DIR, asof_ymd=trade_ymd,
                                methods=["DISPLAY_SCORE", "FINAL_SCORE", "AI_SCORE"])

    # ── 텔레그램 발송 ──
    if ctx.enable_telegram:
        mkt_temp = label_market_temp(ctx.breadth.get("ALL", np.nan))
        summary_text = f"🌡 {mkt_temp} (Breadth: {ctx.breadth.get('ALL', 0)}%)"
        if ctx.macro_msg:
            summary_text += f"\n{ctx.macro_msg}"
        if "SECTOR_RANK" in df_out.columns:
            top_sectors = df_out.sort_values("SECTOR_RS", ascending=False)["업종_대분류"].unique()[:2]
            summary_text += f"\n🚀 주도: {' '.join(top_sectors)}"
        send_telegram_auto(df_out, trade_ymd, market_summary=summary_text, limit_count=ctx.rec_limit)
    else:
        log("✉️ 텔레그램 발송 생략 (옵션 설정)")

    # ── 자동 캘리브레이션 ──
    try:
        from auto_backtest import auto_calibrate
        cal_summary = auto_calibrate(OUT_DIR, trade_ymd)
        log(f"📊 캘리브레이션: {cal_summary.get('n_trades', 0)}건, "
            f"승률={cal_summary.get('overall_winrate', 0):.1%}")
    except Exception as e:
        log(f"⚠️ 자동 캘리브레이션 스킵: {e}")

    # ── 포지션 트래킹 ──
    try:
        from position_tracker import track_open_positions, register_from_recommendations
        register_from_recommendations(OUT_DIR, df_out, trade_ymd, top_n=ctx.rec_limit)
        track_result = track_open_positions(OUT_DIR, trade_ymd)
        log(f"📍 포지션: 체크={track_result.get('checked', 0)}, "
            f"이벤트={track_result.get('events', 0)}, "
            f"청산={track_result.get('closed', 0)}")
    except Exception as e:
        log(f"⚠️ 포지션 트래킹 스킵: {e}")

    # ── 일일 브리핑 ──
    try:
        from daily_briefing import generate_daily_briefing
        briefing = generate_daily_briefing(OUT_DIR, trade_ymd, df_out)
        if briefing["count"] > 0:
            log(f"📝 일일 브리핑: {briefing['count']}종목 "
                f"[{', '.join(briefing.get('names', []))}]")
        else:
            log("📝 일일 브리핑: 대상 없음 (ATTACK/ARMED 0건)")
    except Exception as e:
        log(f"⚠️ 일일 브리핑 생성 스킵: {e}")

    ctx.df_out = df_out
