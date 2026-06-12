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

    # [v24.1 P0-C] carry 종목 OHLCV를 ctx.ohlcv_map에 병합
    # — 보유 중(carry) 종목이야말로 무결성 감사가 가장 중요하므로,
    #   pipeline 말미 data_integrity 감사가 carry 행을 SKIP하지 않게 한다.
    try:
        if isinstance(getattr(ctx, "ohlcv_map", None), dict):
            for _ck, _cdf in carry_ohlcv.items():
                if _cdf is not None and not _cdf.empty:
                    ctx.ohlcv_map[str(_ck).zfill(6)] = _cdf
    except Exception as e:
        logger.debug(f"carry OHLCV → ctx.ohlcv_map 병합 실패 (무해): {e}")

    # [v20.3.5] carry 종목이 top_df에 없으면 임시 추가
    # + 이미 있어도 거래대금이 부족하면 보강 (min_turnover 필터 우회)
    _top_codes = set(ctx.top_df["종목코드"].astype(str).str.zfill(6)) if ctx.top_df is not None and not ctx.top_df.empty else set()
    _missing_in_top = [c for c in carry_codes if c not in _top_codes]
    if ctx.top_df is not None:
        # (A) 이미 top_df에 있는 carry 종목의 거래대금 보강
        _existing_carry = [c for c in carry_codes if c in _top_codes]
        if _existing_carry and "거래대금(원)" in ctx.top_df.columns:
            for _ec in _existing_carry:
                _mask = ctx.top_df["종목코드"].astype(str).str.zfill(6) == _ec
                _cur_tv = pd.to_numeric(ctx.top_df.loc[_mask, "거래대금(원)"], errors="coerce").fillna(0)
                if (_cur_tv < 50e8).any():
                    ctx.top_df.loc[_mask, "거래대금(원)"] = 50e8

        # (B) top_df에 없는 carry 종목 임시 추가
        _patch_rows = []
        for _mc in _missing_in_top:
            _ohlcv = carry_ohlcv.get(_mc)
            if _ohlcv is not None and not _ohlcv.empty:
                _last_c = float(pd.to_numeric(_ohlcv["종가"], errors="coerce").iloc[-1])
                _last_v = float(pd.to_numeric(_ohlcv["거래량"], errors="coerce").iloc[-1]) if "거래량" in _ohlcv.columns else 0
                _tv_won = _last_c * _last_v
                # CARRY 종목은 이미 추천된 종목 → 거래대금 필터 우회
                # min_turnover_eok(약 30억) 이상 보장
                _tv_won = max(_tv_won, 50e8)  # 최소 50억원 보장
                _patch_rows.append({
                    "종목코드": _mc,
                    "거래대금(원)": _tv_won,
                    "시장": "KOSPI" if _mc in ctx.kospi_set else "KOSDAQ",
                })
        if _patch_rows:
            _patch_df = pd.DataFrame(_patch_rows)
            # top_df에 없는 컬럼은 NaN으로 채움
            for _col in ctx.top_df.columns:
                if _col not in _patch_df.columns:
                    _patch_df[_col] = np.nan
            ctx.top_df = pd.concat([ctx.top_df, _patch_df[ctx.top_df.columns]], ignore_index=True)
            log(f"   📋 top_df 임시 보강: {len(_patch_rows)}건 추가")

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
                # [v20.3.5] 실패 진단: 왜 None인지 추적
                _diag = f"ohlcv_len={len(ohlcv_df)}"
                _mcap = ctx.mcap_map.get(code, 0)
                _diag += f",mcap={_mcap:.0f}"
                _in_top = code in set(ctx.top_df["종목코드"].astype(str).str.zfill(6))
                _diag += f",in_top={_in_top}"
                logger.info(f"   🔍 {code} analyze→None: {_diag}")
                legacy_codes.append(code)
                fail_reasons[code] = f"analyze_returned_none({_diag})"
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

    # [v3.9.28] 보유 손익(진입 시점 종가 대비) — carry-stale 가드용. 실패 시 NaN(손실조건 미발동).
    try:
        _from_close = []
        for _, _r in carry_df.iterrows():
            _code = str(_r.get("종목코드", "")).zfill(6)
            _fdt = pd.to_datetime(_r.get("CARRY_FROM_DATE"), format="%Y%m%d", errors="coerce")
            _o = carry_ohlcv.get(_code)
            _fc = np.nan
            if _o is not None and not _o.empty and "종가" in _o.columns and pd.notna(_fdt):
                try:
                    _idx = pd.to_datetime(_o.index, errors="coerce")
                    _sub = _o[_idx <= _fdt]
                    if not _sub.empty:
                        _fc = float(pd.to_numeric(_sub["종가"], errors="coerce").dropna().iloc[-1])
                except Exception:
                    _fc = np.nan
            _from_close.append(_fc)
        carry_df["CARRY_FROM_CLOSE"] = _from_close
        _cur = pd.to_numeric(carry_df.get("종가"), errors="coerce")
        _fcs = pd.to_numeric(carry_df["CARRY_FROM_CLOSE"], errors="coerce")
        carry_df["CARRY_RET_PCT"] = ((_cur - _fcs) / _fcs * 100.0).where(_fcs > 0)
    except Exception as _e:
        logger.debug(f"CARRY_RET_PCT 계산 실패(무시): {_e}")
        if "CARRY_FROM_CLOSE" not in carry_df.columns:
            carry_df["CARRY_FROM_CLOSE"] = np.nan
        if "CARRY_RET_PCT" not in carry_df.columns:
            carry_df["CARRY_RET_PCT"] = np.nan

    carry_df["기준일"] = ctx.trade_ymd
    return carry_df


# ─────────────────────────────────────────────────────────────
#  [v3.9.28] 단계형 보유경과 청산 가드 (Staged Carry-Stale Exit Guard)
#  - GUARD #3/#8 production 승격: 신세계I&C(10일·-9%)류 방치 포지션 경고.
#  - 자동매도 아님. 보유관리 카드에서 '청산 검토 신호'로 강하게 경고하는 표시/감점 레이어.
#  - 신규 진입 산식(TOP_PICK / BUY_NOW_ELIGIBLE / scoring_engine)은 변경하지 않는다.
# ─────────────────────────────────────────────────────────────
def add_carry_stale_guard_columns(df: pd.DataFrame, today_ymd: str = None) -> pd.DataFrame:
    """v3.9.28 보유경과 단계 / 청산검토 신호 / 감점을 부여한다.

    단계:  0~3 FRESH · 4~6 WATCH · 7~9 STALE · 10+ DEAD
    신호:  WATCH=표시만 · STALE=손실이면 경고 · DEAD=청산 검토(CARRY_EXIT_SIGNAL=1)
           DEAD + 손실≤-5% + 회복신호 없음 + 구조/타이밍 악화 → '강한 청산 검토'
    감점:  WATCH 0 · STALE 10/14/18(day7/8/9) · DEAD 22~35(escalation) · stale&손실 +5
    표현 안전: '자동매도 / 팔아라' 금지 — '청산 검토 / 보유관리 주의 / 관찰'만 사용.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    idx = out.index
    n = len(out)

    # 1) 보유 경과일 (없으면 CARRY_FROM_DATE로 계산)
    if "CARRY_AGE_DAYS" in out.columns:
        age = pd.to_numeric(out["CARRY_AGE_DAYS"], errors="coerce")
    else:
        age = pd.Series(np.nan, index=idx)
    if age.isna().all() and "CARRY_FROM_DATE" in out.columns and today_ymd:
        try:
            _fd = pd.to_datetime(out["CARRY_FROM_DATE"], format="%Y%m%d", errors="coerce")
            age = (pd.Timestamp(str(today_ymd)) - _fd).dt.days
        except Exception:
            age = pd.Series(np.nan, index=idx)
    age = age.fillna(0).clip(0, 365).astype(int)
    out["CARRY_AGE_DAYS"] = age

    # 2) 보유 손익 (없으면 NaN — 손실 조건은 NaN이면 미발동)
    if "CARRY_RET_PCT" in out.columns:
        ret = pd.to_numeric(out["CARRY_RET_PCT"], errors="coerce")
    else:
        ret = pd.Series(np.nan, index=idx)

    # 3) 구조/타이밍 (회복신호 없음 / 악화 판정용; 없으면 양호값으로 봐 미발동)
    struct = (pd.to_numeric(out["STRUCT_SCORE"], errors="coerce").fillna(100.0)
              if "STRUCT_SCORE" in out.columns else pd.Series(100.0, index=idx))
    timing = (pd.to_numeric(out["TIMING_SCORE"], errors="coerce").fillna(100.0)
              if "TIMING_SCORE" in out.columns else pd.Series(100.0, index=idx))

    # 4) 단계 산정 (0~3 FRESH · 4~6 WATCH · 7~9 STALE · 10+ DEAD)
    stage = pd.Series("FRESH", index=idx, dtype="object")
    stage[age >= 4] = "WATCH"
    stage[age >= 7] = "STALE"
    stage[age >= 10] = "DEAD"
    out["CARRY_STALE_STAGE"] = stage

    is_stale = stage.eq("STALE")
    is_dead = stage.eq("DEAD")

    # 5) 청산 검토 신호 (DEAD에서만 1) — 자동매도 아님, 보유관리 경고용
    out["CARRY_EXIT_SIGNAL"] = is_dead.astype(int)

    # 6) 기존 호환 — IS_STALE_CARRY = 7일+ (STALE 또는 DEAD)
    out["IS_STALE_CARRY"] = (is_stale | is_dead)

    # 7) 감점 커브
    base = pd.Series(0.0, index=idx)
    base[is_stale] = 10.0 + (age[is_stale] - 7).clip(lower=0) * 4.0          # 10/14/18
    base[is_dead] = (22.0 + (age[is_dead] - 10).clip(lower=0) * 2.0).clip(upper=35.0)  # 22~35
    losing = ret.le(-5.0).fillna(False)                                       # 손실 -5% 이하
    base = base + 5.0 * ((is_stale | is_dead) & losing).astype(float)         # stale&손실 +5
    penalty = base.clip(0, 35)
    out["STALE_PENALTY"] = penalty

    # 8) DISPLAY_SCORE 차감 (표시 점수만 — 진입 산식 무관)
    if "DISPLAY_SCORE" in out.columns:
        out["DISPLAY_SCORE"] = (
            pd.to_numeric(out["DISPLAY_SCORE"], errors="coerce").fillna(0) - penalty
        ).clip(0, 100)

    # 9) '강한 청산 검토' 게이트 (4조건 모두): DEAD & 손실≤-5 & 회복신호 없음 & 구조/타이밍 악화
    no_recovery = timing.lt(40.0)                       # 회복 신호 없음 (타이밍 약화)
    deteriorating = struct.lt(50.0) & timing.lt(40.0)   # 구조/타이밍 동반 악화
    strong_review = is_dead & losing & no_recovery & deteriorating

    # 10) 사유 문자열 (표현 안전)
    ret_list = ret.tolist()
    age_list = age.tolist()
    stage_list = stage.tolist()
    losing_list = losing.tolist()
    strong_list = strong_review.tolist()

    def _ret_txt(i: int) -> str:
        v = ret_list[i] if i < len(ret_list) else np.nan
        return f"{v:+.1f}%" if pd.notna(v) else "—"

    reasons = []
    for i in range(n):
        a = age_list[i]
        st = stage_list[i]
        if st == "FRESH":
            reasons.append("")
        elif st == "WATCH":
            reasons.append(f"보유 {a}일차 · 손익 {_ret_txt(i)} · 관찰")
        elif st == "STALE":
            tail = " · 손실 지속" if losing_list[i] else ""
            reasons.append(f"보유 {a}일차 · 손익 {_ret_txt(i)} · 보유관리 주의{tail}")
        else:  # DEAD
            head = "강한 청산 검토" if strong_list[i] else "청산 검토"
            reasons.append(f"보유 {a}일차 · 손익 {_ret_txt(i)} · {head}")
    out["CARRY_STALE_REASON"] = reasons

    return out


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
    # [v22] Legacy RANK_SCORE 기반 계산은 '기록만'.
    # ROUTE 강등은 ELITE 기반 EST_WIN_RATE 재계산 후 진행 (아래 2-pass 블록).
    try:
        from kelly_calibrator import calibrated_win_rate as _cwr, get_calibration_mode as _gcm
        df_out["EST_WIN_RATE"]=np.nan; df_out["CAL_HOLD_REASON"]=""; df_out["LOW_WR_FLAG"]=False
        # [v22] legacy 기록용 별도 컬럼
        df_out["LEGACY_EST_WIN_RATE"] = np.nan
        df_out["LEGACY_LOW_WR_FLAG"] = False
        _cm = _gcm(OUT_DIR, asof_ymd=ctx.trade_ymd)
        df_out["CALIBRATION_MODE"]=_cm["mode"]; df_out["CAL_N_TRADES"]=_cm["n_trades"]
        _is_emp = _cm["mode"] in ("LIGHT","MATURE")
        log(f"📊 캘리브레이션 모드: {_cm['mode']} (트레이드 {_cm['n_trades']}건)")
        _sc2 = "DISPLAY_SCORE" if "DISPLAY_SCORE" in df_out.columns else "FINAL_SCORE"
        for _i in df_out.index:
            _s = float(df_out.at[_i, _sc2]) if pd.notna(df_out.at[_i, _sc2]) else 0
            try:
                _w = _cwr(_s, OUT_DIR, method="RANK_SCORE", horizon=5, asof_ymd=ctx.trade_ymd)
                # [v22] 초기엔 EST_WIN_RATE에도 seed 값 (ELITE 재계산에서 덮어씀),
                # LEGACY_* 컬럼엔 RANK_SCORE 기준 값 영구 기록
                df_out.at[_i,"EST_WIN_RATE"] = round(_w, 3)
                df_out.at[_i,"LEGACY_EST_WIN_RATE"] = round(_w, 3)
                if _w < 0.45:
                    df_out.at[_i,"LOW_WR_FLAG"] = True
                    df_out.at[_i,"LEGACY_LOW_WR_FLAG"] = True
                    # [v22] ROUTE 강등은 ELITE 기반 재계산 후에만 — 여기서는 금지
            except (KeyError,ValueError,FileNotFoundError): pass
            except Exception as e: logging.warning(f"⚠️ 캘리브레이션 오류 ({_i}): {e}")
        _lt = int(df_out["LEGACY_LOW_WR_FLAG"].sum())
        if _lt > 0:
            log(f"📊 [v22] RANK_SCORE 기준 승률 45%미만 {_lt}건 기록됨 (ROUTE 강등은 ELITE 재계산 후)")
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
                    # [v3.9.28] 단계형 보유경과 청산 가드 (표시/감점/청산검토 신호)
                    #   - 자동매도 아님. DEAD(10일+)·손실 stale에 '청산 검토 신호'를 부여.
                    _cd = add_carry_stale_guard_columns(_cd, today_ymd=ctx.trade_ymd)
                    _sc2 = int(_cd["IS_STALE_CARRY"].sum()) if "IS_STALE_CARRY" in _cd.columns else 0
                    _dead2 = int((_cd.get("CARRY_STALE_STAGE", pd.Series("", index=_cd.index)) == "DEAD").sum())
                    _exit2 = int(pd.to_numeric(_cd.get("CARRY_EXIT_SIGNAL", 0), errors="coerce").fillna(0).sum())
                    if _sc2 > 0 or _dead2 > 0:
                        log(f"   ⏳ carry-stale: STALE+ {_sc2}건 · DEAD {_dead2}건 · 청산검토신호 {_exit2}건")

                    # [v22] 캘리브레이션 적용 (CARRY 재분석분) — LEGACY 기록만
                    # 이 시점엔 ELITE_SCORE가 없으므로 RANK_SCORE 기반으로 seed 값만 기록.
                    # 이후 ELITE 2-pass에서 compute_est_win_rate()로 덮어씀 (SSOT 일관성).
                    try:
                        from kelly_calibrator import calibrated_win_rate as _cwr2, get_calibration_mode as _gcm2
                        _cm3 = _gcm2(OUT_DIR, asof_ymd=ctx.trade_ymd)
                        _sc_col = "DISPLAY_SCORE" if "DISPLAY_SCORE" in _cd.columns else "FINAL_SCORE"
                        # LEGACY 기록용 컬럼 초기화
                        if "LEGACY_EST_WIN_RATE" not in _cd.columns:
                            _cd["LEGACY_EST_WIN_RATE"] = np.nan
                        if "LEGACY_LOW_WR_FLAG" not in _cd.columns:
                            _cd["LEGACY_LOW_WR_FLAG"] = False
                        for _j in _cd.index:
                            _s = float(_cd.at[_j, _sc_col]) if pd.notna(_cd.at[_j, _sc_col]) else 0
                            try:
                                _w = _cwr2(_s, OUT_DIR, method="RANK_SCORE", horizon=5, asof_ymd=ctx.trade_ymd)
                                # seed 값 + LEGACY 기록 (ELITE 2-pass에서 덮어씀)
                                _cd.at[_j, "EST_WIN_RATE"] = round(_w, 3)
                                _cd.at[_j, "LEGACY_EST_WIN_RATE"] = round(_w, 3)
                                if _w < 0.45:
                                    _cd.at[_j, "LEGACY_LOW_WR_FLAG"] = True
                                # [v22] LOW_WR_FLAG는 ELITE 기반 재계산이 세팅 — 여기선 건드리지 않음
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

                    # [v20.3.4] refresh_rate 실질 반영
                    if _rate < 50:
                        log(f"   ⚠️ CARRY refresh rate {_rate:.0f}% < 50% — legacy 추가 제한 적용")
                        _is_leg = df_out["ROW_BUILD_MODE"] == "CARRY_LEGACY"
                        if _is_leg.any():
                            # 추가 패널티: rate에 비례 (0%→-10추가, 50%→-0)
                            _extra = int(10 * (1 - _rate / 50))
                            df_out.loc[_is_leg, "DISPLAY_SCORE"] = (
                                pd.to_numeric(df_out.loc[_is_leg, "DISPLAY_SCORE"], errors="coerce")
                                .fillna(0) - _extra
                            ).clip(0, 100)
                            log(f"   📉 legacy {_is_leg.sum()}건 DISPLAY_SCORE -{_extra} 추가 적용")

                    # ctx에 저장 → Health/UI 연동 가능
                    ctx.breadth["CARRY_REFRESH_RATE"] = round(_rate, 1)
                    ctx.breadth["CARRY_TOTAL"] = int(_total_carry)
                    ctx.breadth["CARRY_REFRESHED"] = int(_refreshed)
                    ctx.breadth["CARRY_LEGACY"] = int(_legacy)
    except Exception as e:
        log(f"⚠️ 캐리오버 처리 실패: {e}")
        import traceback; logger.warning(traceback.format_exc())

    # ──── 재동기화 ────
    # [v22] IS_NOW_ENTRY는 pipeline_finalize.finalize_sort에서 adaptive로 재계산됨.
    # 여기선 IS_ACTIVE/IS_WATCH만 설정 (ROUTE 의미 보존).
    # 중간 concat 정렬은 제거 — finalize_sort가 최종 순서 결정 (SORT_SPEC SSOT).
    df_out["IS_ACTIVE"] = df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED])
    df_out["IS_WATCH"] = df_out["ROUTE"] == Route.WAIT
    df_out["ACTION_PRIORITY"] = df_out["ROUTE"].map(_am).fillna(7).astype(int)

    # [v20.3.1] DATA_FRESHNESS_OK / ROW_BUILD_MODE — NaN 확정 채움
    if "DATA_FRESHNESS_OK" not in df_out.columns:
        df_out["DATA_FRESHNESS_OK"] = True
    else:
        df_out["DATA_FRESHNESS_OK"] = df_out["DATA_FRESHNESS_OK"].fillna(True)
    if "ROW_BUILD_MODE" not in df_out.columns:
        df_out["ROW_BUILD_MODE"] = "FRESH"
    else:
        df_out["ROW_BUILD_MODE"] = df_out["ROW_BUILD_MODE"].fillna("FRESH")

    # ═══════════════════════════════════════════════════
    # [v22] ELITE_SCORE 2-pass: 랭킹 축 = 승률 축 일치
    # ═══════════════════════════════════════════════════
    # Pass 1: compute_elite_score → ELITE_SCORE 산출
    # compute_est_win_rate (SSOT) → ELITE 기반 EST_WIN_RATE + 메타 주입
    # LOW_WR_FLAG / CAL_HOLD_REASON 리셋 → ELITE 기준 재평가
    # ROUTE 강등 (MATURE/LIGHT만) → 상태 동기화
    # Pass 2: compute_elite_score 재호출 → TOP_PICK/STABLE 최종 판정
    try:
        from scoring_engine import compute_elite_score
        from kelly_calibrator import compute_est_win_rate, get_calibration_mode

        # [v22] _cm2 독립 조회 — 앞 legacy try가 실패해도 작동
        _cm2 = get_calibration_mode(OUT_DIR, asof_ymd=ctx.trade_ymd)

        # ── Pass 1 ── ELITE_SCORE 먼저 계산
        df_out, _elite_meta = compute_elite_score(
            df_out, out_dir=OUT_DIR, trade_ymd=ctx.trade_ymd
        )

        # ── [v22] ELITE_SCORE 기반 EST_WIN_RATE + 메타 주입 (SSOT 함수) ──
        df_out = compute_est_win_rate(df_out, OUT_DIR, asof_ymd=ctx.trade_ymd)

        # ── [v22] LOW_WR_FLAG / CAL_HOLD_REASON 리셋 ──
        # RANK_SCORE 기반 판정 결과 제거 — ELITE 기반으로만 다시 세팅.
        # LEGACY_LOW_WR_FLAG / LEGACY_EST_WIN_RATE는 기록용으로 보존.
        df_out["LOW_WR_FLAG"] = False
        df_out["CAL_HOLD_REASON"] = ""

        # ── [v22] ROUTE 강등: ELITE 기반 EST_WIN_RATE가 낮은 종목 HOLD ──
        _is_emp2 = _cm2["mode"] in ("LIGHT", "MATURE")
        _low_elite_wr = (
            (df_out["EST_WIN_RATE"].fillna(0) < 0.45)
            & df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED])
        )
        if _is_emp2 and _low_elite_wr.any():
            _demote_n = int(_low_elite_wr.sum())
            _wr_vals = df_out.loc[_low_elite_wr, "EST_WIN_RATE"].round(2).astype(str)
            df_out.loc[_low_elite_wr, "ROUTE"] = Route.WAIT
            df_out.loc[_low_elite_wr, "상태"] = Route.WAIT
            df_out.loc[_low_elite_wr, "CAL_HOLD_REASON"] = "elite_low_wr_" + _wr_vals
            df_out.loc[_low_elite_wr, "LOW_WR_FLAG"] = True
            log(f"📊 [v22] ELITE 기반 강등: {_demote_n}건 ATTACK/ARMED → WAIT (wr<0.45)")
        elif not _is_emp2:
            # FALLBACK 모드에서는 강등 대신 메모만 (표본 부족 → 보수적)
            _fallback_low = (
                (df_out["EST_WIN_RATE"].fillna(0) < 0.45)
                & df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED])
            )
            if _fallback_low.any():
                _wr_vals = df_out.loc[_fallback_low, "EST_WIN_RATE"].round(2).astype(str)
                df_out.loc[_fallback_low, "CAL_HOLD_REASON"] = "fallback_elite_wr_" + _wr_vals
                df_out.loc[_fallback_low, "LOW_WR_FLAG"] = True

        # ── [v22] ROUTE 변경 이후 상태 컬럼 동기화 ──
        df_out["IS_ACTIVE"] = df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED, "ATTACK", "ARMED"])
        df_out["IS_WATCH"] = df_out["ROUTE"].isin([Route.WAIT, "WAIT"])
        df_out["ACTION_PRIORITY"] = df_out["ROUTE"].map(_am).fillna(7).astype(int)

        # ── Pass 2 ── EST_WIN_RATE_MODE 주입된 상태에서 TOP_PICK/STABLE 재평가
        df_out, _elite_meta = compute_elite_score(
            df_out, out_dir=OUT_DIR, trade_ymd=ctx.trade_ymd
        )

        # funnel 메타를 breadth에 노출 (daily_briefing이 참조)
        ctx.breadth["stable_funnel"] = _elite_meta.get("stable_funnel", {})
        ctx.breadth["aggressive_funnel"] = _elite_meta.get("aggressive_funnel", {})

        _top3 = df_out.nlargest(3, "ELITE_SCORE")
        _names = ", ".join(f"{r['종목명']}({r['ELITE_SCORE']:.0f})" for _, r in _top3.iterrows())
        log(f"🏆 ELITE Top3: {_names}")
        log(f"📊 [v22] EST_WIN_RATE (ELITE 기반): mode={_cm2['mode']}, n={_cm2['n_trades']}")
    except Exception as e:
        logger.warning(f"⚠️ ELITE_SCORE 2-pass 실패: {e}")

    # ═══════════════════════════════════════════════════
    # [v23.0] 통합 GUARD 엔진 — compute_elite_score 직후 · Kelly 직전
    # ═══════════════════════════════════════════════════
    # 8개 GUARD(유동성·RR열화·보유경과·저모멘텀·추세붕괴·시장역행·윗꼬리·사전경고)를
    # 단일 모듈에서 적용. GUARDED_ELITE_SCORE / GUARD_KELLY_MULT / ELITE_LABEL 부여,
    # enforce 모드면 TOP_PICK을 가드 통과분으로 재게이트(TOP_PICK_RAW 보존).
    try:
        from guard_system import apply_guard_system, guard_summary
        from collector_config import DEFAULT_CONFIG as _GCFG

        # 당일 KOSPI 등락률 — ctx에 있으면 전달 (없으면 컬럼/스킵 fallback)
        _kospi_ret = None
        try:
            _kospi_ret = getattr(ctx, "kospi_ret_1d", None)
            if _kospi_ret is None and isinstance(getattr(ctx, "breadth", None), dict):
                _kospi_ret = ctx.breadth.get("KOSPI_RET_1D")
        except Exception:
            _kospi_ret = None

        df_out = apply_guard_system(df_out, config=_GCFG, kospi_ret_1d=_kospi_ret)

        _gs = guard_summary(df_out)
        log(
            "🛡️ [v23.0] GUARD: 차단 {nb}건 · 강제청산경보 {fe}건 · 사전경고 {pw}건 · "
            "ELITE {ne}건 (가드탈락 후보 {gb}건)".format(
                nb=_gs.get("n_block", 0), fe=_gs.get("n_force_exit", 0),
                pw=_gs.get("n_pre_warning", 0), ne=_gs.get("n_elite", 0),
                gb=_gs.get("n_guard_blocked_pick", 0),
            )
        )
    except Exception as e:
        logger.warning(f"⚠️ [v23.0] GUARD 적용 실패 (기존 추천 유지): {e}")

    # ── [v23.1] Momentum Lane — OVERHEAT × GUARD 통과 종목의 별도 추천 레인 ──
    try:
        from momentum_lane import (
            apply_momentum_lane, compute_market_risk_off, momentum_summary,
        )
        from collector_config import DEFAULT_CONFIG as _MCFG
        import os as _os

        _kospi_path = _os.path.join(getattr(_MCFG, "out_dir", "data"), "kospi_daily.csv")
        _risk_off, _regime = compute_market_risk_off(_kospi_path)
        df_out = apply_momentum_lane(df_out, market_risk_off=_risk_off, config=_MCFG)

        _ms = momentum_summary(df_out)
        log(
            "⚡ [v23.1] Momentum Lane: 실전 {a}건 · 관찰 {b}건 (시장 {rg})".format(
                a=_ms.get("tier_a", 0), b=_ms.get("tier_b", 0),
                rg=("위험회피 → 레인 OFF" if _risk_off else "정상 → 레인 ON"),
            )
        )
    except Exception as e:
        logger.warning(f"⚠️ [v23.1] Momentum Lane 적용 실패 (무해): {e}")

    # ── [v23.2] Stop Override — 공식 신호 손절 -10% (베어 시 OFF + 신규차단) ──
    try:
        from stop_override import apply_stop_override, stop_override_summary
        from momentum_lane import compute_market_risk_off as _cmro2
        from collector_config import DEFAULT_CONFIG as _SCFG
        import os as _os2

        _kp2 = _os2.path.join(getattr(_SCFG, "out_dir", "data"), "kospi_daily.csv")
        _ro2, _ = _cmro2(_kp2)
        df_out = apply_stop_override(df_out, market_risk_off=_ro2, config=_SCFG)

        _ss = stop_override_summary(df_out)
        log(
            "\U0001F6E1\uFE0F [v23.2] Stop Override: \uc801\uc6a9 {a}\uac74 \u00b7 \uc2e0\uaddc\ucc28\ub2e8 {b}\uac74 "
            "(\uc190\uc808-{p:.0f}%, \uc2dc\uc7a5 {rg})".format(
                a=_ss.get("active", 0), b=_ss.get("blocked", 0),
                p=_ss.get("stop_pct", 0.10) * 100,
                rg=("\uc704\ud5d8\ud68c\ud53c \u2192 override OFF + \uc2e0\uaddc\ucc28\ub2e8" if _ro2
                    else "\uc815\uc0c1 \u2192 \uc190\uc808 override ON"),
            )
        )
    except Exception as e:
        logger.warning(f"\u26a0\uFE0F [v23.2] Stop Override \uc801\uc6a9 \uc2e4\ud328 (\ubb34\ud574): {e}")

    # ── [v24.1 P0-C] 데이터 무결성 게이트 — OHLC 감사 + 이상 폭등 플래그 (P0-B 흡수) ──
    # 임계값은 collector_config.DataIntegrityConfig(SSOT)에서 주입.
    # 무결성 실패/폭등 종목은 모멘텀 레인에서 제외, 사유는 DATA_INTEGRITY_REASON에 기록.
    try:
        from data_integrity import apply_data_integrity, data_integrity_summary
        from collector_config import DEFAULT_CONFIG as _DICFG

        df_out = apply_data_integrity(
            df_out, ohlcv_map=getattr(ctx, "ohlcv_map", None), config=_DICFG,
        )
        _ds = data_integrity_summary(df_out)
        log(
            "🧪 [v24.1] Data Integrity: 감사 {n}건 · 무결성위반 {b}건 · "
            "폭등플래그 {s}건 · 모멘텀제외 {m}건".format(
                n=_ds.get("n_audited", 0), b=_ds.get("n_integrity_bad", 0),
                s=_ds.get("n_surge", 0), m=_ds.get("n_momentum_excluded", 0),
            )
        )
    except Exception as e:
        # 신규 모듈 실패 시에도 v24 P0-B(폭등주 모멘텀 제외)는 절대 잃지 않는다.
        logger.warning(f"⚠️ [v24.1] Data Integrity 적용 실패 → P0-B 폴백: {e}")
        try:
            if "ret_10d_%" in df_out.columns:
                _r10 = pd.to_numeric(df_out["ret_10d_%"], errors="coerce").fillna(0.0)
            else:
                _r10 = pd.Series(0.0, index=df_out.index)
            _abn_mask = _r10 > 300.0
            df_out["ABNORMAL_SURGE_FLAG"] = _abn_mask
            if "MOMENTUM_LANE" in df_out.columns:
                df_out.loc[_abn_mask, "MOMENTUM_LANE"] = 0
        except Exception as e2:
            logger.warning(f"⚠️ [v24.1] P0-B 폴백도 실패 (플래그 False 고정): {e2}")
            df_out["ABNORMAL_SURGE_FLAG"] = False

    ctx.df_out = df_out
    return ctx
