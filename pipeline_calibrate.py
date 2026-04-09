# -*- coding: utf-8 -*-
"""pipeline_calibrate.py — Stage 5: 캘리브레이션 + 켈리 + 캐리오버 [v20.3-carry-refresh]

[v20.3] CARRY 종목 재분석 패치
 - 기존: 이전 recommend 행을 복사 → 종가만 갱신 → 지표 동결
 - 수정: CARRY 종목도 OHLCV 재수집 → analyze_ticker 재분석 → 지표 신선도 보장
 - CARRY_FROM_DATE: 최초 진입일 기준 고정 (리셋 금지)
 - ROW_BUILD_MODE: FRESH / CARRY_REFRESHED / CARRY_LEGACY 명시
"""
import os, logging, numpy as np, pandas as pd
from pipeline_context import PipelineContext
from shared_log import log, OUT_DIR
from collector_config import Route

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  CARRY 재분석 핵심 함수 (P0 패치 #1)
# ─────────────────────────────────────────────────────────────
def _refresh_carry_rows(ctx: PipelineContext, prev_df: pd.DataFrame,
                        carry_codes: list, *,
                        analyze_fn=None, prepare_ohlcv_fn=None,
                        trigger_fn=None, ml_apply_fn=None,
                        build_score_fn=None, gen_reasons_fn=None,
                        ) -> pd.DataFrame:
    """
    CARRY 대상 종목을 당일 OHLCV 기준으로 재분석한다.
    실패 시 legacy(기존 행 복사) 폴백.

    의존성 주입: 테스트 시 mock 함수를 직접 넘길 수 있음.
    기본값 None이면 실제 모듈에서 import.
    """
    if analyze_fn is None:
        from collector import analyze_ticker as analyze_fn
    if prepare_ohlcv_fn is None:
        from collector import prepare_ohlcv_data as prepare_ohlcv_fn
    if trigger_fn is None:
        from trigger_engine import calculate_trigger_score as trigger_fn
    if build_score_fn is None:
        from scoring_engine import build_global_score as build_score_fn
    if gen_reasons_fn is None:
        from scoring_engine import generate_score_reasons as gen_reasons_fn

    if not carry_codes:
        return pd.DataFrame()

    # 1) OHLCV 재수집 (carry 종목만)
    log(f"🔄 CARRY 재분석: {len(carry_codes)}건 OHLCV 수집 중...")
    try:
        carry_ohlcv = prepare_ohlcv_fn(
            carry_codes, ctx.start_s, ctx.end_s, ctx.trade_ymd
        )
    except Exception as e:
        logger.warning(f"⚠️ CARRY OHLCV 수집 실패: {e}")
        carry_ohlcv = {}

    # 2) 종목별 재분석 — 실패 사유 추적
    rows = []
    legacy_codes = []
    fail_reasons = {}  # code → reason
    for code in carry_codes:
        ohlcv_df = carry_ohlcv.get(code)
        if ohlcv_df is None or ohlcv_df.empty:
            legacy_codes.append(code)
            fail_reasons[code] = "ohlcv_missing"
            continue
        if len(ohlcv_df) < 60:
            legacy_codes.append(code)
            fail_reasons[code] = f"ohlcv_short({len(ohlcv_df)}rows)"
            continue
        try:
            row = analyze_fn(
                code, ohlcv_df, ctx.top_df, ctx.mcap_map,
                ctx.kospi_set, ctx.kosdaq_set, ctx.name_map,
                ctx.sector_map, ctx.bench_map, ctx.inv_maps,
            )
            if row is None:
                legacy_codes.append(code)
                fail_reasons[code] = "analyze_returned_none"
                continue

            # Trigger Score
            row["TRIGGER_SCORE"] = float(trigger_fn(ohlcv_df))
            row["RAW_TRIGGER_SCORE"] = row["TRIGGER_SCORE"]
            row["ROW_BUILD_MODE"] = "CARRY_REFRESHED"
            rows.append(row)
        except Exception as e:
            logger.warning(f"⚠️ CARRY 재분석 실패 ({code}): {e}")
            legacy_codes.append(code)
            fail_reasons[code] = f"exception:{type(e).__name__}"

    # 실패 사유 집계 로그
    if fail_reasons:
        from collections import Counter
        reason_counts = Counter(fail_reasons.values())
        reason_str = ", ".join(f"{r}:{c}" for r, c in reason_counts.most_common())
        log(f"   📋 CARRY 실패 상세: {reason_str}")

    # 3) 재분석 성공분: ML + 스코어링
    refreshed_df = pd.DataFrame()
    if rows:
        refreshed_df = pd.DataFrame(rows)
        # ML Score
        try:
            if ml_apply_fn is not None:
                refreshed_df = ml_apply_fn(refreshed_df, carry_ohlcv)
            else:
                from collector import ml_engine
                refreshed_df = ml_engine.apply_ml_score(refreshed_df, carry_ohlcv)
        except Exception as e:
            logger.warning(f"⚠️ CARRY ML 실패: {e}")
            refreshed_df["ML_SCORE"] = 0.0

        refreshed_df["ML_SCORE"] = pd.to_numeric(
            refreshed_df.get("ML_SCORE", 0.0), errors="coerce"
        ).fillna(0.0).clip(0, 100)

        # 통합 스코어 + 이유
        refreshed_df = build_score_fn(refreshed_df, ctx.macro_risk)
        refreshed_df = gen_reasons_fn(refreshed_df, macro_risk=ctx.macro_risk)

        log(f"✅ CARRY 재분석 성공: {len(refreshed_df)}건")

    # 4) 재분석 실패분: legacy 폴백
    legacy_df = pd.DataFrame()
    if legacy_codes:
        prev_map = prev_df.set_index("종목코드")
        legacy_rows = []
        for code in legacy_codes:
            if code in prev_map.index:
                legacy_rows.append(prev_map.loc[code].to_dict())
        if legacy_rows:
            legacy_df = pd.DataFrame(legacy_rows)
            legacy_df["종목코드"] = legacy_codes[:len(legacy_df)]
            legacy_df["ROW_BUILD_MODE"] = "CARRY_LEGACY"
            legacy_df["DATA_FRESHNESS_OK"] = False
            # Legacy 패널티: DISPLAY_SCORE -15 (강화)
            if "DISPLAY_SCORE" in legacy_df.columns:
                legacy_df["DISPLAY_SCORE"] = (
                    pd.to_numeric(legacy_df["DISPLAY_SCORE"], errors="coerce")
                    .fillna(0) - 15
                ).clip(0, 100)
            # 실패 사유 컬럼 추가
            legacy_df["CARRY_FAIL_REASON"] = legacy_df["종목코드"].map(fail_reasons).fillna("unknown")
            legacy_df["ROUTE_REASON"] = "캐리 재계산 실패: legacy snapshot"
            log(f"⚠️ CARRY legacy 폴백: {len(legacy_df)}건")

    # 5) 합치기
    parts = [df for df in [refreshed_df, legacy_df] if not df.empty]
    if not parts:
        return pd.DataFrame()

    carry_df = pd.concat(parts, ignore_index=True)

    # 6) CARRY 상태 설정
    carry_df["ROUTE"] = Route.CARRY.value
    carry_df["상태"] = Route.CARRY.value
    carry_df["IS_ACTIVE"] = False
    carry_df["IS_NOW_ENTRY"] = False
    carry_df["IS_WATCH"] = False

    # 7) CARRY_FROM_DATE 보존 (P0 패치 #2)
    prev_carry_dates = prev_df.set_index("종목코드").get("CARRY_FROM_DATE")
    if prev_carry_dates is not None:
        carry_df["CARRY_FROM_DATE"] = carry_df["종목코드"].map(prev_carry_dates)
    # 최초 carry인 종목은 이전 기준일 사용
    prev_dates = prev_df.set_index("종목코드").get("기준일")
    if prev_dates is not None:
        carry_df["CARRY_FROM_DATE"] = carry_df["CARRY_FROM_DATE"].where(
            carry_df["CARRY_FROM_DATE"].notna(),
            carry_df["종목코드"].map(prev_dates)
        )
    # 그래도 없으면 오늘 날짜
    carry_df["CARRY_FROM_DATE"] = carry_df["CARRY_FROM_DATE"].fillna(ctx.trade_ymd)

    carry_df["기준일"] = ctx.trade_ymd
    return carry_df


# ─────────────────────────────────────────────────────────────
#  메인 함수
# ─────────────────────────────────────────────────────────────
def run_calibration(ctx: PipelineContext) -> PipelineContext:
    from collector import apply_kelly_betting
    df_out = ctx.df_out
    _sort_col = "DISPLAY_SCORE" if "DISPLAY_SCORE" in df_out.columns else "FINAL_SCORE"
    df_out[_sort_col] = pd.to_numeric(df_out[_sort_col], errors="coerce").fillna(0)
    _am = {Route.ATTACK:1,"ATTACK":1,Route.ARMED:2,"ARMED":2,Route.WAIT:3,"WAIT":3,
           Route.NEUTRAL:4,"NEUTRAL":4,Route.OVERHEAT:5,"OVERHEAT":5,
           Route.EXIT_WARNING:6,"EXIT_WARNING":6,Route.CARRY:7,"CARRY":7}
    df_out["ACTION_PRIORITY"] = df_out["ROUTE"].map(_am).fillna(7).astype(int)
    pm = df_out.index < 120
    sk, sa = ["ACTION_PRIORITY", _sort_col], [True, False]
    df_out = pd.concat([df_out[pm].sort_values(sk, ascending=sa), df_out[~pm].sort_values(sk, ascending=sa)], ignore_index=True)
    df_out["LDY_RANK"] = np.arange(1, len(df_out)+1)
    # UI 호환
    df_out["LDY_SCORE"] = df_out["DISPLAY_SCORE"]; df_out["TOTAL_SCORE"] = df_out["DISPLAY_SCORE"]; df_out["RANK_SCORE"] = df_out["DISPLAY_SCORE"]
    df_out["벤치_60d_KOSPI_%"] = ctx.bench_map.get("KOSPI",{}).get(60, np.nan)
    df_out["벤치_60d_KOSDAQ_%"] = ctx.bench_map.get("KOSDAQ",{}).get(60, np.nan)
    df_out["IS_ACTIVE"] = df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED])
    df_out["IS_NOW_ENTRY"] = df_out["ROUTE"] == Route.ATTACK
    df_out["IS_WATCH"] = df_out["ROUTE"] == Route.WAIT
    # [v20.3] 당일 분석 행에 ROW_BUILD_MODE 태그
    if "ROW_BUILD_MODE" not in df_out.columns:
        df_out["ROW_BUILD_MODE"] = "FRESH"

    # ──── 캘리브레이션 ────
    try:
        from kelly_calibrator import calibrated_win_rate as _cwr, get_calibration_mode as _gcm
        df_out["EST_WIN_RATE"]=np.nan; df_out["CAL_HOLD_REASON"]=""; df_out["LOW_WR_FLAG"]=False
        _cm = _gcm(OUT_DIR, asof_ymd=ctx.trade_ymd)
        df_out["CALIBRATION_MODE"]=_cm["mode"]; df_out["CAL_N_TRADES"]=_cm["n_trades"]
        _is_emp = _cm["mode"] in ("LIGHT","MATURE")
        log(f"📊 캘리브레이션 모드: {_cm['mode']} (트레이드 {_cm['n_trades']}건)")
        _sc2 = "DISPLAY_SCORE" if "DISPLAY_SCORE" in df_out.columns else "FINAL_SCORE"
        for _i in df_out.index:
            _s = float(df_out.at[_i, _sc2]) if pd.notna(df_out.at[_i, _sc2]) else 0
            try:
                _w = _cwr(_s, OUT_DIR, method="RANK_SCORE", horizon=5, asof_ymd=ctx.trade_ymd)
                df_out.at[_i,"EST_WIN_RATE"] = round(_w, 3)
                if _w < 0.45:
                    df_out.at[_i,"LOW_WR_FLAG"] = True
                    if _is_emp and df_out.at[_i,"ROUTE"] in (Route.ATTACK,Route.ARMED):
                        df_out.at[_i,"ROUTE"]=Route.WAIT; df_out.at[_i,"상태"]=Route.WAIT
                        df_out.at[_i,"CAL_HOLD_REASON"]=f"low_wr_{_w:.2f}"
                    elif not _is_emp and df_out.at[_i,"ROUTE"] in (Route.ATTACK,Route.ARMED):
                        df_out.at[_i,"CAL_HOLD_REASON"]=f"fallback_wr_{_w:.2f}"
            except (KeyError,ValueError,FileNotFoundError): pass
            except Exception as e: logging.warning(f"⚠️ 캘리브레이션 오류 ({_i}): {e}")
        _dc = (df_out["CAL_HOLD_REASON"].str.startswith("low_wr")).sum()
        _lt = df_out["LOW_WR_FLAG"].sum()
        if _lt > 0:
            _fc = (df_out["CAL_HOLD_REASON"].str.startswith("fallback")).sum()
            log(f"📊 캘리브레이션: 승률 45%미만 {_lt}건 중 {_dc}건 실제 격하, {_fc}건 fallback(격하 안 함)")
    except ImportError: log("ℹ️ kelly_calibrator 미설치, 캘리브레이션 연동 스킵")
    except Exception as e: log(f"⚠️ 캘리브레이션 연동 에러: {e}")

    # ──── 켈리 ────
    try: df_out = apply_kelly_betting(df_out, total_capital=10_000_000, out_dir=OUT_DIR)
    except Exception as e:
        log(f"⚠️ 켈리 비중 계산 실패: {e}")
        for _kc in ["켈리_수량","켈리_금액(원)","추천수량","추천금액(만원)"]:
            if _kc not in df_out.columns: df_out[_kc] = 0

    # ══════════════════════════════════════════════════════════
    #  캐리오버 — [v20.3] CARRY 재분석 방식으로 전면 교체
    # ══════════════════════════════════════════════════════════
    try:
        _prev = os.path.join(OUT_DIR, "recommend_latest.csv")
        if os.path.exists(_prev):
            _pd = pd.read_csv(_prev, dtype={"종목코드":str})
            _pd["종목코드"] = _pd["종목코드"].str.zfill(6)
            _cc = set(df_out["종목코드"].astype(str).str.zfill(6))
            _ar = {Route.ARMED, Route.ARMED.value, Route.ATTACK, Route.ATTACK.value,
                   Route.CARRY, Route.CARRY.value, "ARMED", "ATTACK", "CARRY"}
            _cm2 = _pd["ROUTE"].isin(_ar) & ~_pd["종목코드"].isin(_cc)
            _carry_prev = _pd[_cm2].copy()

            if not _carry_prev.empty:
                carry_codes = _carry_prev["종목코드"].tolist()

                # ★ 핵심 변경: 복사 대신 재분석
                _cd = _refresh_carry_rows(ctx, _pd, carry_codes)

                if not _cd.empty:
                    # Stale carry 패널티 적용
                    try:
                        _fd = pd.to_datetime(_cd["CARRY_FROM_DATE"], format="%Y%m%d", errors="coerce")
                        _cd["CARRY_AGE_DAYS"] = (pd.Timestamp(ctx.trade_ymd) - _fd).dt.days.fillna(0).astype(int)
                    except Exception:
                        _cd["CARRY_AGE_DAYS"] = 0
                    _cd["IS_STALE_CARRY"] = _cd["CARRY_AGE_DAYS"] >= 7
                    _sp = _cd["CARRY_AGE_DAYS"].clip(0,30).apply(
                        lambda d: min(20.0, max(0, (d-5)*5.0)) if d > 5 else 0.0
                    )
                    _cd["STALE_PENALTY"] = _sp
                    if "DISPLAY_SCORE" in _cd.columns:
                        _cd["DISPLAY_SCORE"] = (
                            pd.to_numeric(_cd["DISPLAY_SCORE"], errors="coerce").fillna(0) - _sp
                        ).clip(0, 100)
                    _sc2 = _cd["IS_STALE_CARRY"].sum()
                    if _sc2 > 0:
                        log(f"   ⏳ stale carry {_sc2}건 (7일+ 경과)")

                    # 캘리브레이션 적용 (CARRY 재분석분에도)
                    try:
                        from kelly_calibrator import calibrated_win_rate as _cwr2, get_calibration_mode as _gcm2
                        _cm3 = _gcm2(OUT_DIR, asof_ymd=ctx.trade_ymd)
                        _sc_col = "DISPLAY_SCORE" if "DISPLAY_SCORE" in _cd.columns else "FINAL_SCORE"
                        for _j in _cd.index:
                            _s = float(_cd.at[_j, _sc_col]) if pd.notna(_cd.at[_j, _sc_col]) else 0
                            try:
                                _w = _cwr2(_s, OUT_DIR, method="RANK_SCORE", horizon=5, asof_ymd=ctx.trade_ymd)
                                _cd.at[_j, "EST_WIN_RATE"] = round(_w, 3)
                                _cd.at[_j, "LOW_WR_FLAG"] = _w < 0.45
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # 랭크 부여 + 합치기
                    _mr = df_out["LDY_RANK"].max() if len(df_out) > 0 else 0
                    _cd["LDY_RANK"] = range(int(_mr)+1, int(_mr)+1+len(_cd))
                    df_out = pd.concat([df_out, _cd], ignore_index=True)

                    _refreshed = (_cd["ROW_BUILD_MODE"] == "CARRY_REFRESHED").sum()
                    _legacy = (_cd["ROW_BUILD_MODE"] == "CARRY_LEGACY").sum()
                    _total_carry = _refreshed + _legacy
                    _rate = _refreshed / _total_carry * 100 if _total_carry > 0 else 0
                    log(f"📌 이전 추천 캐리오버: {_total_carry}건 "
                        f"(재분석 {_refreshed}건, legacy {_legacy}건, "
                        f"refresh_rate={_rate:.0f}%)")
                    if _rate < 50:
                        log(f"   ⚠️ CARRY refresh rate {_rate:.0f}% < 50% — 추천 신선도 주의")
    except Exception as e:
        log(f"⚠️ 캐리오버 처리 실패: {e}")
        import traceback; logger.warning(traceback.format_exc())

    # ──── 재동기화 + 최종 정렬 ────
    df_out["IS_ACTIVE"] = df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED])
    df_out["IS_NOW_ENTRY"] = df_out["ROUTE"] == Route.ATTACK
    df_out["IS_WATCH"] = df_out["ROUTE"] == Route.WAIT
    df_out["ACTION_PRIORITY"] = df_out["ROUTE"].map(_am).fillna(7).astype(int)
    _fsc = "DISPLAY_SCORE" if "DISPLAY_SCORE" in df_out.columns else "FINAL_SCORE"
    _fk, _fa = ["ACTION_PRIORITY", _fsc], [True, False]
    _cm3 = df_out["ACTION_PRIORITY"] == 7
    _nc = df_out[~_cm3]; _cp = df_out[_cm3].sort_values(_fk, ascending=_fa)
    _p = _nc.head(min(120, len(_nc))); _n = _nc.iloc[len(_p):]
    df_out = pd.concat([
        _p.sort_values(_fk, ascending=_fa),
        _n.sort_values(_fk, ascending=_fa),
        _cp
    ], ignore_index=True)
    df_out["LDY_RANK"] = np.arange(1, len(df_out)+1)

    # [v20.3.1] DATA_FRESHNESS_OK / ROW_BUILD_MODE — NaN 확정 채움
    # concat 후 NaN이 섞일 수 있으므로 항상 fillna 실행
    if "DATA_FRESHNESS_OK" not in df_out.columns:
        df_out["DATA_FRESHNESS_OK"] = True
    else:
        df_out["DATA_FRESHNESS_OK"] = df_out["DATA_FRESHNESS_OK"].fillna(True)
    if "ROW_BUILD_MODE" not in df_out.columns:
        df_out["ROW_BUILD_MODE"] = "FRESH"
    else:
        df_out["ROW_BUILD_MODE"] = df_out["ROW_BUILD_MODE"].fillna("FRESH")

    ctx.df_out = df_out
    return ctx
