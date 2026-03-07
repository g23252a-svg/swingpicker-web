# -*- coding: utf-8 -*-
"""pipeline_analyze.py — Stage 2: 종목별 기술 분석 [v20.2]"""
import gc, pandas as pd
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pipeline_context import PipelineContext
from shared_log import log, MAX_WORKERS

def analyze_universe(ctx: PipelineContext) -> PipelineContext:
    from collector import analyze_ticker
    from telegram_sender import send_telegram_auto
    rows: List[Dict[str, Any]] = []
    err_cnt = 0
    if MAX_WORKERS <= 1:
        for t in tqdm(ctx.tickers, desc="Analyzing"):
            code6 = str(t).zfill(6)
            try:
                df_t = ctx.ohlcv_map.get(code6)
                row = analyze_ticker(t, df_t, ctx.top_df, ctx.mcap_map,
                    ctx.kospi_set, ctx.kosdaq_set, ctx.name_map, ctx.sector_map,
                    ctx.bench_map, ctx.inv_maps)
                if row is not None: rows.append(row)
            except Exception as e:
                err_cnt += 1; log(f"⚠️ {code6} 분석 오류: {type(e).__name__}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = []
            for t in ctx.tickers:
                code6 = str(t).zfill(6)
                df_t = ctx.ohlcv_map.get(code6)
                if df_t is None or df_t.empty: continue
                futs.append(ex.submit(analyze_ticker, t, df_t, ctx.top_df, ctx.mcap_map,
                    ctx.kospi_set, ctx.kosdaq_set, ctx.name_map, ctx.sector_map,
                    ctx.bench_map, ctx.inv_maps))
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Analyzing"):
                try:
                    row = fut.result()
                    if row is not None: rows.append(row)
                except Exception as e:
                    err_cnt += 1; log(f"⚠️ 병렬 처리 중 오류: {type(e).__name__}: {e}")
    if err_cnt > 0: log(f"⚠️ 분석 중 오류 발생/데이터 부족 종목 수: {err_cnt}건")
    gc.collect()
    if not rows:
        log("⚠️ 필터를 통과한 종목이 없습니다.")
        if ctx.enable_telegram:
            send_telegram_auto(pd.DataFrame(), ctx.trade_ymd, market_summary="⚠️ 필터 통과 종목 없음", limit_count=0)
        ctx.df_out = None
        return ctx
    ctx.df_out = pd.DataFrame(rows)
    return ctx
