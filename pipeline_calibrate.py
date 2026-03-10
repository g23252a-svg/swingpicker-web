# -*- coding: utf-8 -*-
"""pipeline_calibrate.py — Stage 5: 캘리브레이션 + 켈리 + 캐리오버 [v20.2]"""
import os, logging, numpy as np, pandas as pd
from pipeline_context import PipelineContext
from shared_log import log, OUT_DIR
from collector_config import Route

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
    # 캘리브레이션
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
    # 켈리
    try: df_out = apply_kelly_betting(df_out, total_capital=10_000_000, out_dir=OUT_DIR)
    except Exception as e:
        log(f"⚠️ 켈리 비중 계산 실패: {e}")
        for _kc in ["켈리_수량","켈리_금액(원)","추천수량","추천금액(만원)"]:
            if _kc not in df_out.columns: df_out[_kc] = 0
    # 캐리오버
    try:
        _prev = os.path.join(OUT_DIR, "recommend_latest.csv")
        if os.path.exists(_prev):
            _pd = pd.read_csv(_prev, dtype={"종목코드":str}); _pd["종목코드"] = _pd["종목코드"].str.zfill(6)
            _cc = set(df_out["종목코드"].astype(str).str.zfill(6))
            _ar = {Route.ARMED,Route.ARMED.value,Route.ATTACK,Route.ATTACK.value,Route.CARRY,Route.CARRY.value,"ARMED","ATTACK","CARRY"}
            _cm2 = _pd["ROUTE"].isin(_ar) & ~_pd["종목코드"].isin(_cc)
            _cd = _pd[_cm2].copy()
            if not _cd.empty:
                _cd["ROUTE"]=Route.CARRY.value; _cd["상태"]=Route.CARRY.value
                _cd["IS_ACTIVE"]=False; _cd["IS_NOW_ENTRY"]=False; _cd["IS_WATCH"]=False
                _cd["CARRY_FROM_DATE"] = _cd.get("기준일", ctx.trade_ymd)
                try:
                    _fd = pd.to_datetime(_cd["CARRY_FROM_DATE"], format="%Y%m%d", errors="coerce")
                    _cd["CARRY_AGE_DAYS"] = (pd.Timestamp(ctx.trade_ymd) - _fd).dt.days.fillna(0).astype(int)
                except: _cd["CARRY_AGE_DAYS"] = 0
                _cd["IS_STALE_CARRY"] = _cd["CARRY_AGE_DAYS"] >= 7
                _sp = _cd["CARRY_AGE_DAYS"].clip(0,30).apply(lambda d: min(20.0,max(0,(d-5)*5.0)) if d>5 else 0.0)
                _cd["STALE_PENALTY"] = _sp
                if "DISPLAY_SCORE" in _cd.columns:
                    _cd["DISPLAY_SCORE"] = (pd.to_numeric(_cd["DISPLAY_SCORE"],errors="coerce").fillna(0)-_sp).clip(0,100)
                _sc2 = _cd["IS_STALE_CARRY"].sum()
                if _sc2 > 0: log(f"   ⏳ stale carry {_sc2}건 (7일+ 경과)")
                _snp = os.path.join(OUT_DIR, f"price_snapshot_{ctx.trade_ymd}.csv")
                if not os.path.exists(_snp): _snp = os.path.join(OUT_DIR, "price_snapshot_latest.csv")
                if os.path.exists(_snp):
                    try:
                        _sn = pd.read_csv(_snp, dtype={"종목코드":str}); _sn["종목코드"]=_sn["종목코드"].str.zfill(6)
                        _pm2 = dict(zip(_sn["종목코드"], pd.to_numeric(_sn["종가"],errors="coerce")))
                        for _j in _cd.index:
                            _c2 = _cd.at[_j,"종목코드"]
                            if _c2 in _pm2 and pd.notna(_pm2[_c2]): _cd.at[_j,"종가"]=_pm2[_c2]
                    except: pass
                _mr = df_out["LDY_RANK"].max() if len(df_out)>0 else 0
                _cd["LDY_RANK"] = range(int(_mr)+1, int(_mr)+1+len(_cd))
                _cd["기준일"] = ctx.trade_ymd
                df_out = pd.concat([df_out, _cd], ignore_index=True)
                log(f"📌 이전 추천 캐리오버: {len(_cd)}건 (CARRY 상태로 유지)")
    except Exception as e: log(f"⚠️ 캐리오버 처리 실패: {e}")
    # 재동기화 + 최종 정렬
    df_out["IS_ACTIVE"]=df_out["ROUTE"].isin([Route.ATTACK,Route.ARMED])
    df_out["IS_NOW_ENTRY"]=df_out["ROUTE"]==Route.ATTACK
    df_out["IS_WATCH"]=df_out["ROUTE"]==Route.WAIT
    df_out["ACTION_PRIORITY"]=df_out["ROUTE"].map(_am).fillna(7).astype(int)
    _fsc = "DISPLAY_SCORE" if "DISPLAY_SCORE" in df_out.columns else "FINAL_SCORE"
    _fk, _fa = ["ACTION_PRIORITY",_fsc], [True,False]
    _cm3 = df_out["ACTION_PRIORITY"]==7
    _nc = df_out[~_cm3]; _cp = df_out[_cm3].sort_values(_fk,ascending=_fa)
    _p = _nc.head(min(120,len(_nc))); _n = _nc.iloc[len(_p):]
    df_out = pd.concat([_p.sort_values(_fk,ascending=_fa), _n.sort_values(_fk,ascending=_fa), _cp], ignore_index=True)
    df_out["LDY_RANK"] = np.arange(1, len(df_out)+1)
    ctx.df_out = df_out
    return ctx
