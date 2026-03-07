# -*- coding: utf-8 -*-
"""
pipeline_score.py — Stage 3: 스코어링 + 라우팅 + 전략 팩토리
═══════════════════════════════════════════════════════════════
[v20.1] collector.py 분해 — 수급보정 → 업종/온도 → ML → Hard Block → 라우트 → 전략
"""

import os
import gc
import numpy as np
import pandas as pd

from pipeline_context import PipelineContext


def run_scoring(ctx: PipelineContext) -> PipelineContext:
    """Stage 3: 수급보정, 업종분류, ML스코어, Hard Block, 라우팅, 전략 팩토리"""
    from collector import (
        log, fetch_investor_net_buying, classify_big_sector,
        add_sector_momentum, compute_market_breadth, label_market_temp,
        ml_engine, calculate_trigger_score, safe_quantile,
        determine_state_dynamic, Route, OUT_DIR,
    )
    from scoring_engine import build_global_score

    df_raw = ctx.df_out

    # ── 1. 수급 데이터 보정 ──
    if ctx.inv_maps:
        map_ant = ctx.inv_maps.get("ant", {})
    else:
        _map_frg, _map_inst, map_ant = fetch_investor_net_buying(ctx.trade_ymd)
        df_raw["외인순매수"] = df_raw["종목코드"].map(_map_frg).fillna(0)
        df_raw["기관순매수"] = df_raw["종목코드"].map(_map_inst).fillna(0)
        df_raw["메이저순매수"] = df_raw["외인순매수"] + df_raw["기관순매수"]
    df_raw["개인순매수"] = df_raw["종목코드"].map(map_ant).fillna(0)

    if "거래대금(원)" not in df_raw.columns:
        tv_eok = pd.to_numeric(df_raw.get("거래대금(억원)", 0), errors="coerce").fillna(0.0)
        df_raw["거래대금(원)"] = (tv_eok * 100_000_000).astype(float)

    # ── 2. 업종 분류 + 시장 온도 ──
    if "업종" in df_raw.columns:
        df_raw["업종_상세"] = df_raw["업종"]
        df_raw["업종_대분류"] = df_raw.apply(
            lambda r: classify_big_sector(str(r.get("종목명", "")), str(r.get("업종", ""))), axis=1
        )

    df_raw, _ = add_sector_momentum(df_raw, "업종_대분류")
    ctx.breadth = compute_market_breadth(df_raw)
    mkt_temp = label_market_temp(ctx.breadth.get("ALL", np.nan))
    log(f"🌡 시장 온도: {mkt_temp} (Breadth: {ctx.breadth.get('ALL', 0)}%) -> 동적 가중치 적용")

    try:
        from stop_logic import get_config as _get_stop_cfg
        _scfg = _get_stop_cfg()
        _scfg.market_breadth = ctx.breadth.get('ALL', 50.0)
        log('Stop config: breadth=%.1f adaptive=%s' % (_scfg.market_breadth, _scfg.adaptive_stop))
    except Exception as _e:
        log('Stop config breadth set failed: ' + str(_e))

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

    # OHLCV 메모리 해제
    ctx.ohlcv_map.clear()
    gc.collect()

    df_out = build_global_score(df_out, ctx.macro_risk)

    # ── 4. Hard Block ──
    try:
        from validation import apply_hard_blocks, block_summary
        df_out, df_blocked = apply_hard_blocks(df_out)
        if len(df_blocked) > 0:
            _bs = block_summary(df_blocked)
            log(f"🚫 Hard Block: {_bs['total_blocked']}건 제외 {_bs['by_rule']}")
            blocked_path = os.path.join(OUT_DIR, f"blocked_{ctx.trade_ymd}.csv")
            df_blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")
    except Exception as _hb_err:
        log(f"⚠️ Hard Block 스킵: {_hb_err}")

    # ── 5. ROUTE 결정 ──
    thresholds = {
        "range_q75": safe_quantile(df_out.get("Range_Pos"), 0.75, 0.8),
        "vol_q75": safe_quantile(df_out.get("Vol_Quality"), 0.75, 1.2),
    }
    df_out["ROUTE"] = df_out.apply(lambda r: determine_state_dynamic(r, thresholds), axis=1)
    df_out["상태"] = df_out["ROUTE"]

    # ── 6. 전략 팩토리 ──
    df_out["STRATEGY"] = "default"
    try:
        from strategies import StrategyFactory
        _breadth_all = ctx.breadth.get("ALL", 50.0)
        _candidates = StrategyFactory.select(ctx.macro_risk, _breadth_all)
        log(f"🎯 활성 전략: {_candidates} (breadth={_breadth_all:.1f}, macro={ctx.macro_risk})")
        if _candidates:
            _all_strat_picks = []
            _default_top_k = 5
            for _sname, _weight in _candidates:
                _strat = StrategyFactory.create(_sname)
                _filtered = _strat.filter(df_out)
                _scored = _strat.score(_filtered)
                _adj_k = max(2, round(_default_top_k * _weight))
                _picks = _strat.rank_and_pick(_scored, top_k=_adj_k)
                _all_strat_picks.append(_picks)
                log(f"   {'✅' if len(_picks) else '⬜'} {_sname}(w={_weight:.2f}): "
                    f"필터 {len(_filtered)}건 → 선정 {len(_picks)}건")
            if _all_strat_picks:
                _strat_df = pd.concat(_all_strat_picks, ignore_index=True)
                _key = "종목코드" if "종목코드" in _strat_df.columns else _strat_df.index.name
                if _key and _key in df_out.columns:
                    _merge_cols = ["STRATEGY", "STRATEGY_SCORE", "STRATEGY_HORIZON"]
                    _merge_cols = [c for c in _merge_cols if c in _strat_df.columns]
                    if _merge_cols:
                        _strat_map = _strat_df.drop_duplicates(_key).set_index(_key)[_merge_cols]
                        for _mc in _merge_cols:
                            df_out[_mc] = df_out[_key].map(_strat_map[_mc]).fillna(df_out.get(_mc, "default"))
                log(f"🎯 전략 분류 완료: 매칭 {len(_strat_df)}건")
    except Exception as _st_err:
        log(f"⚠️ 전략 팩토리 스킵: {_st_err}")

    # ── 7. 정예군 편성 ──
    ebs_val = pd.to_numeric(df_out.get("EBS", 0), errors='coerce').fillna(0)
    struct_val = pd.to_numeric(df_out.get("STRUCT_SCORE", 0), errors='coerce').fillna(0)
    mask_safe = ~df_out["ROUTE"].isin([Route.OVERHEAT, Route.EXIT_WARNING])
    mask_qual = (ebs_val >= ctx.pass_ebs) & (struct_val >= 60) & mask_safe

    df_prime = df_out[mask_qual].copy().sort_values(["TIMING_SCORE", "AI_SCORE"], ascending=False)
    df_normal = df_out[~mask_qual].copy().sort_values("FINAL_SCORE", ascending=False)
    df_out = pd.concat([df_prime.head(120), df_normal, df_prime.iloc[120:]], ignore_index=True)

    # ── 8. 날짜 메타 ──
    df_out["기준일"] = ctx.trade_ymd
    df_out["시총기준일"] = ctx.mcap_ymd

    ctx.df_out = df_out
    return ctx
