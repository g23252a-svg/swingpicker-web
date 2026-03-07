# -*- coding: utf-8 -*-
"""pipeline_score.py — Stage 3: 스코어링 + 라우팅 + 전략 팩토리 [v20.2]"""
import os, gc, numpy as np, pandas as pd
from pipeline_context import PipelineContext
from shared_log import log, OUT_DIR, safe_quantile
from collector_config import Route
from scoring_engine import determine_state_dynamic, build_global_score
from macro_filter import compute_market_breadth, label_market_temp

def run_scoring(ctx: PipelineContext) -> PipelineContext:
    from collector import (fetch_investor_net_buying, classify_big_sector,
        add_sector_momentum, ml_engine, calculate_trigger_score)
    df_raw = ctx.df_out
    # ── 1. 수급 보정 ──
    if ctx.inv_maps:
        map_ant = ctx.inv_maps.get("ant", {})
    else:
        _map_frg, _map_inst, map_ant = fetch_investor_net_buying(ctx.trade_ymd)
        df_raw["외인순매수"] = df_raw["종목코드"].map(_map_frg).fillna(0)
        df_raw["기관순매수"] = df_raw["종목코드"].map(_map_inst).fillna(0)
        df_raw["메이저순매수"] = df_raw["외인순매수"] + df_raw["기관순매수"]
    df_raw["개인순매수"] = df_raw["종목코드"].map(map_ant).fillna(0)
    if "거래대금(원)" not in df_raw.columns:
        tv = pd.to_numeric(df_raw.get("거래대금(억원)", 0), errors="coerce").fillna(0.0)
        df_raw["거래대금(원)"] = (tv * 100_000_000).astype(float)
    # ── 2. 업종 + 온도 ──
    if "업종" in df_raw.columns:
        df_raw["업종_상세"] = df_raw["업종"]
        df_raw["업종_대분류"] = df_raw.apply(lambda r: classify_big_sector(str(r.get("종목명","")), str(r.get("업종",""))), axis=1)
    df_raw, _ = add_sector_momentum(df_raw, "업종_대분류")
    ctx.breadth = compute_market_breadth(df_raw)
    mkt_temp = label_market_temp(ctx.breadth.get("ALL", np.nan))
    log(f"🌡 시장 온도: {mkt_temp} (Breadth: {ctx.breadth.get('ALL', 0)}%) -> 동적 가중치 적용")
    try:
        from stop_logic import get_config as _get_stop_cfg
        _scfg = _get_stop_cfg(); _scfg.market_breadth = ctx.breadth.get('ALL', 50.0)
        log('Stop config: breadth=%.1f adaptive=%s' % (_scfg.market_breadth, _scfg.adaptive_stop))
    except Exception as _e: log('Stop config breadth set failed: ' + str(_e))
    # ── 3. ML + Trigger + 통합 스코어링 ──
    log("🧠 AI 엔진 동기화 및 통합 스코어링 시작...")
    df_out = ml_engine.apply_ml_score(df_raw, ctx.ohlcv_map)
    df_out["ML_SCORE"] = pd.to_numeric(df_out.get("ML_SCORE", 0.0), errors='coerce').fillna(0.0).clip(0, 100)
    trigger_list = []
    for idx, row in df_out.iterrows():
        code = str(row['종목코드']).zfill(6)
        ohlcv_df = ctx.ohlcv_map.get(code)
        ts = calculate_trigger_score(ohlcv_df) if ohlcv_df is not None and not ohlcv_df.empty else 0.0
        trigger_list.append(ts)
    df_out['TRIGGER_SCORE'] = trigger_list
    df_out['RAW_TRIGGER_SCORE'] = df_out['TRIGGER_SCORE']
    ctx.ohlcv_map.clear(); gc.collect()
    df_out = build_global_score(df_out, ctx.macro_risk)
    # ── 4. Hard Block ──
    try:
        from validation import apply_hard_blocks, block_summary
        df_out, df_blocked = apply_hard_blocks(df_out)
        if len(df_blocked) > 0:
            _bs = block_summary(df_blocked)
            log(f"🚫 Hard Block: {_bs['total_blocked']}건 제외 {_bs['by_rule']}")
            df_blocked.to_csv(os.path.join(OUT_DIR, f"blocked_{ctx.trade_ymd}.csv"), index=False, encoding="utf-8-sig")
    except Exception as _hb: log(f"⚠️ Hard Block 스킵: {_hb}")
    # ── 5. ROUTE ──
    th = {"range_q75": safe_quantile(df_out.get("Range_Pos"), 0.75, 0.8),
          "vol_q75": safe_quantile(df_out.get("Vol_Quality"), 0.75, 1.2)}
    df_out["ROUTE"] = df_out.apply(lambda r: determine_state_dynamic(r, th), axis=1)
    df_out["상태"] = df_out["ROUTE"]
    # ── 6. 전략 팩토리 ──
    df_out["STRATEGY"] = "default"
    try:
        from strategies import StrategyFactory
        _ba = ctx.breadth.get("ALL", 50.0)
        _cands = StrategyFactory.select(ctx.macro_risk, _ba)
        log(f"🎯 활성 전략: {_cands} (breadth={_ba:.1f}, macro={ctx.macro_risk})")
        if _cands:
            _all = []
            for _sn, _w in _cands:
                _s = StrategyFactory.create(_sn); _fi = _s.filter(df_out); _sc = _s.score(_fi)
                _k = max(2, round(5 * _w)); _p = _s.rank_and_pick(_sc, top_k=_k); _all.append(_p)
                log(f"   {'✅' if len(_p) else '⬜'} {_sn}(w={_w:.2f}): 필터 {len(_fi)}건 → 선정 {len(_p)}건")
            if _all:
                _sd = pd.concat(_all, ignore_index=True)
                _key = "종목코드" if "종목코드" in _sd.columns else _sd.index.name
                if _key and _key in df_out.columns:
                    _mc = [c for c in ["STRATEGY","STRATEGY_SCORE","STRATEGY_HORIZON"] if c in _sd.columns]
                    if _mc:
                        _sm = _sd.drop_duplicates(_key).set_index(_key)[_mc]
                        for c in _mc: df_out[c] = df_out[_key].map(_sm[c]).fillna(df_out.get(c, "default"))
                log(f"🎯 전략 분류 완료: 매칭 {len(_sd)}건")
    except Exception as _st: log(f"⚠️ 전략 팩토리 스킵: {_st}")
    # ── 7. 정예군 편성 ──
    ebs = pd.to_numeric(df_out.get("EBS", 0), errors='coerce').fillna(0)
    struct = pd.to_numeric(df_out.get("STRUCT_SCORE", 0), errors='coerce').fillna(0)
    ms = ~df_out["ROUTE"].isin([Route.OVERHEAT, Route.EXIT_WARNING])
    mq = (ebs >= ctx.pass_ebs) & (struct >= 60) & ms
    dp = df_out[mq].copy().sort_values(["TIMING_SCORE","AI_SCORE"], ascending=False)
    dn = df_out[~mq].copy().sort_values("FINAL_SCORE", ascending=False)
    df_out = pd.concat([dp.head(120), dn, dp.iloc[120:]], ignore_index=True)
    df_out["기준일"] = ctx.trade_ymd; df_out["시총기준일"] = ctx.mcap_ymd
    ctx.df_out = df_out
    return ctx
