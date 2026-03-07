# -*- coding: utf-8 -*-
"""
pipeline_calibrate.py — Stage 5: 캘리브레이션 + 켈리 + 캐리오버
═══════════════════════════════════════════════════════════════════
[v20.1] collector.py 분해 — 재정렬 → 캘리브레이션 → 켈리 → 캐리 → 최종정렬
"""

import os
import numpy as np
import pandas as pd

from pipeline_context import PipelineContext


def run_calibration(ctx: PipelineContext) -> PipelineContext:
    """Stage 5: 재정렬 + 캘리브레이션 + 켈리 + 캐리오버 + 최종정렬"""
    from collector import log, apply_kelly_betting, Route, OUT_DIR

    df_out = ctx.df_out

    # ── [Step 7] 최종 점수 재정렬 + 랭크 확정 ──
    _sort_col = "DISPLAY_SCORE" if "DISPLAY_SCORE" in df_out.columns else "FINAL_SCORE"
    df_out[_sort_col] = pd.to_numeric(df_out[_sort_col], errors="coerce").fillna(0)

    _action_map = {
        Route.ATTACK: 1, "ATTACK": 1,
        Route.ARMED: 2, "ARMED": 2,
        Route.WAIT: 3, "WAIT": 3,
        Route.NEUTRAL: 4, "NEUTRAL": 4,
        Route.OVERHEAT: 5, "OVERHEAT": 5,
        Route.EXIT_WARNING: 6, "EXIT_WARNING": 6,
        Route.CARRY: 7, "CARRY": 7,
    }
    df_out["ACTION_PRIORITY"] = df_out["ROUTE"].map(_action_map).fillna(7).astype(int)

    _prime_mask = df_out.index < 120
    _sort_keys = ["ACTION_PRIORITY", _sort_col]
    _sort_asc = [True, False]
    df_prime_re = df_out[_prime_mask].sort_values(_sort_keys, ascending=_sort_asc)
    df_normal_re = df_out[~_prime_mask].sort_values(_sort_keys, ascending=_sort_asc)
    df_out = pd.concat([df_prime_re, df_normal_re], ignore_index=True)
    df_out["LDY_RANK"] = np.arange(1, len(df_out) + 1)

    # ── [Step 7.5] UI 호환성 동기화 ──
    df_out["LDY_SCORE"] = df_out["DISPLAY_SCORE"]
    df_out["TOTAL_SCORE"] = df_out["DISPLAY_SCORE"]
    df_out["RANK_SCORE"] = df_out["DISPLAY_SCORE"]
    df_out["벤치_60d_KOSPI_%"] = ctx.bench_map.get("KOSPI", {}).get(60, np.nan)
    df_out["벤치_60d_KOSDAQ_%"] = ctx.bench_map.get("KOSDAQ", {}).get(60, np.nan)
    df_out["IS_ACTIVE"] = df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED])
    df_out["IS_NOW_ENTRY"] = df_out["ROUTE"] == Route.ATTACK
    df_out["IS_WATCH"] = df_out["ROUTE"] == Route.WAIT

    # ── [Phase 2-2] 캘리브레이션 승률 ──
    try:
        from kelly_calibrator import calibrated_win_rate as _cal_wr, get_calibration_mode as _get_cal_mode
        df_out["EST_WIN_RATE"] = np.nan
        df_out["CAL_HOLD_REASON"] = ""
        df_out["LOW_WR_FLAG"] = False

        _cal_mode = _get_cal_mode(OUT_DIR, asof_ymd=ctx.trade_ymd)
        df_out["CALIBRATION_MODE"] = _cal_mode["mode"]
        df_out["CAL_N_TRADES"] = _cal_mode["n_trades"]
        _is_empirical = _cal_mode["mode"] in ("LIGHT", "MATURE")
        log(f"📊 캘리브레이션 모드: {_cal_mode['mode']} (트레이드 {_cal_mode['n_trades']}건)")

        _score_col_cal = "DISPLAY_SCORE" if "DISPLAY_SCORE" in df_out.columns else "FINAL_SCORE"
        _wr_min = 0.45

        for _idx in df_out.index:
            _sc = float(df_out.at[_idx, _score_col_cal]) if pd.notna(df_out.at[_idx, _score_col_cal]) else 0
            try:
                _wr = _cal_wr(_sc, OUT_DIR, method="RANK_SCORE", horizon=5, asof_ymd=ctx.trade_ymd)
                df_out.at[_idx, "EST_WIN_RATE"] = round(_wr, 3)
                if _wr < _wr_min:
                    df_out.at[_idx, "LOW_WR_FLAG"] = True
                    if _is_empirical and df_out.at[_idx, "ROUTE"] in (Route.ATTACK, Route.ARMED):
                        df_out.at[_idx, "ROUTE"] = Route.WAIT
                        df_out.at[_idx, "상태"] = Route.WAIT
                        df_out.at[_idx, "CAL_HOLD_REASON"] = f"low_wr_{_wr:.2f}"
                    elif not _is_empirical and df_out.at[_idx, "ROUTE"] in (Route.ATTACK, Route.ARMED):
                        df_out.at[_idx, "CAL_HOLD_REASON"] = f"fallback_wr_{_wr:.2f}"
            except (KeyError, ValueError, FileNotFoundError):
                pass
            except Exception as e:
                import logging
                logging.warning(f"⚠️ 캘리브레이션 예기치 않은 오류 ({_idx}): {type(e).__name__}: {e}")

        _demoted_cnt = (df_out["CAL_HOLD_REASON"].str.startswith("low_wr")).sum()
        _low_total = df_out["LOW_WR_FLAG"].sum()
        if _low_total > 0:
            _fallback_cnt = (df_out["CAL_HOLD_REASON"].str.startswith("fallback")).sum()
            log(f"📊 캘리브레이션: 승률 45%미만 {_low_total}건 중 "
                f"{_demoted_cnt}건 실제 격하, {_fallback_cnt}건 fallback(격하 안 함)")
    except ImportError:
        log("ℹ️ kelly_calibrator 미설치, 캘리브레이션 연동 스킵")
    except Exception as _cal_err:
        log(f"⚠️ 캘리브레이션 연동 에러: {_cal_err}")

    # ── [Step 8] 켈리 자금 관리 ──
    try:
        df_out = apply_kelly_betting(df_out, total_capital=10_000_000, out_dir=OUT_DIR)
    except Exception as e:
        log(f"⚠️ 켈리 비중 계산 실패 (0으로 대체): {e}")
        for _kc in ["켈리_수량", "켈리_금액(원)", "추천수량", "추천금액(만원)"]:
            if _kc not in df_out.columns:
                df_out[_kc] = 0

    # ── [Step 7.9] 캐리오버 ──
    try:
        _prev_latest = os.path.join(OUT_DIR, "recommend_latest.csv")
        if os.path.exists(_prev_latest):
            _prev_df = pd.read_csv(_prev_latest, dtype={"종목코드": str})
            _prev_df["종목코드"] = _prev_df["종목코드"].astype(str).str.zfill(6)
            _cur_codes = set(df_out["종목코드"].astype(str).str.zfill(6))

            _active_routes = {Route.ARMED, Route.ARMED.value, Route.ATTACK, Route.ATTACK.value,
                              Route.CARRY, Route.CARRY.value, "ARMED", "ATTACK", "CARRY"}
            _carry_mask_prev = (
                _prev_df["ROUTE"].isin(_active_routes)
                & ~_prev_df["종목코드"].isin(_cur_codes)
            )
            _carry_df = _prev_df[_carry_mask_prev].copy()

            if not _carry_df.empty:
                _carry_df["ROUTE"] = Route.CARRY.value
                _carry_df["상태"] = Route.CARRY.value
                _carry_df["IS_ACTIVE"] = False
                _carry_df["IS_NOW_ENTRY"] = False
                _carry_df["IS_WATCH"] = False

                _carry_df["CARRY_FROM_DATE"] = _carry_df.get("기준일", ctx.trade_ymd)
                try:
                    _from_dates = pd.to_datetime(_carry_df["CARRY_FROM_DATE"], format="%Y%m%d", errors="coerce")
                    _today = pd.Timestamp(ctx.trade_ymd)
                    _carry_df["CARRY_AGE_DAYS"] = (_today - _from_dates).dt.days.fillna(0).astype(int)
                except Exception:
                    _carry_df["CARRY_AGE_DAYS"] = 0

                _carry_df["IS_STALE_CARRY"] = _carry_df["CARRY_AGE_DAYS"] >= 7
                _stale_penalty = _carry_df["CARRY_AGE_DAYS"].clip(0, 30).apply(
                    lambda d: min(20.0, max(0, (d - 5) * 5.0)) if d > 5 else 0.0
                )
                _carry_df["STALE_PENALTY"] = _stale_penalty
                if "DISPLAY_SCORE" in _carry_df.columns:
                    _carry_df["DISPLAY_SCORE"] = (
                        pd.to_numeric(_carry_df["DISPLAY_SCORE"], errors="coerce").fillna(0) - _stale_penalty
                    ).clip(0, 100)

                _stale_cnt = _carry_df["IS_STALE_CARRY"].sum()
                if _stale_cnt > 0:
                    log(f"   ⏳ stale carry {_stale_cnt}건 (7일+ 경과, 점수 감쇠 적용)")

                _snap_path = os.path.join(OUT_DIR, f"price_snapshot_{ctx.trade_ymd}.csv")
                if not os.path.exists(_snap_path):
                    _snap_path = os.path.join(OUT_DIR, "price_snapshot_latest.csv")
                if os.path.exists(_snap_path):
                    try:
                        _snap = pd.read_csv(_snap_path, dtype={"종목코드": str})
                        _snap["종목코드"] = _snap["종목코드"].astype(str).str.zfill(6)
                        _price_map = dict(zip(_snap["종목코드"], pd.to_numeric(_snap["종가"], errors="coerce")))
                        for _idx in _carry_df.index:
                            _code = _carry_df.at[_idx, "종목코드"]
                            if _code in _price_map and pd.notna(_price_map[_code]):
                                _carry_df.at[_idx, "종가"] = _price_map[_code]
                    except Exception:
                        pass

                _max_rank = df_out["LDY_RANK"].max() if len(df_out) > 0 else 0
                _carry_df["LDY_RANK"] = range(int(_max_rank) + 1, int(_max_rank) + 1 + len(_carry_df))
                _carry_df["기준일"] = ctx.trade_ymd

                df_out = pd.concat([df_out, _carry_df], ignore_index=True)
                log(f"📌 이전 추천 캐리오버: {len(_carry_df)}건 (CARRY 상태로 유지)")
    except Exception as e:
        log(f"⚠️ 캐리오버 처리 실패 (무시): {e}")

    # ── [Fix v4] 상태 플래그 재동기화 + 최종 재정렬 ──
    df_out["IS_ACTIVE"] = df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED])
    df_out["IS_NOW_ENTRY"] = df_out["ROUTE"] == Route.ATTACK
    df_out["IS_WATCH"] = df_out["ROUTE"] == Route.WAIT
    df_out["ACTION_PRIORITY"] = df_out["ROUTE"].map(_action_map).fillna(7).astype(int)

    _final_sort_col = "DISPLAY_SCORE" if "DISPLAY_SCORE" in df_out.columns else "FINAL_SCORE"
    _final_keys = ["ACTION_PRIORITY", _final_sort_col]
    _final_asc = [True, False]

    _carry_mask_final = df_out["ACTION_PRIORITY"] == 7
    _non_carry = df_out[~_carry_mask_final]
    _carry_part = df_out[_carry_mask_final].sort_values(_final_keys, ascending=_final_asc)

    _nc_prime = _non_carry.head(min(120, len(_non_carry)))
    _nc_normal = _non_carry.iloc[len(_nc_prime):]
    _nc_prime = _nc_prime.sort_values(_final_keys, ascending=_final_asc)
    _nc_normal = _nc_normal.sort_values(_final_keys, ascending=_final_asc)

    df_out = pd.concat([_nc_prime, _nc_normal, _carry_part], ignore_index=True)
    df_out["LDY_RANK"] = np.arange(1, len(df_out) + 1)

    ctx.df_out = df_out
    return ctx
