# -*- coding: utf-8 -*-
"""
pipeline_data.py — Stage 1: 참조 데이터 로딩
═══════════════════════════════════════════════════
[v20.1] collector.py 분해 — ML 학습 + 거래일/시총/벤치/종목풀/OHLCV/수급
"""

import os
import gc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from pipeline_context import PipelineContext


def ensure_ml_ready():
    """ML 모델 학습 체크 + 파일 존재 검증"""
    from collector import ml_engine, log

    if ml_engine.is_trained_today():
        log("✅ [SKIP] 오늘 이미 모델 학습이 완료되었습니다.")
    else:
        log("🤖 AI 모델 최적화 진행 중...")
        try:
            ml_engine.train_model()
        except Exception as e:
            log(f"⚠️ 모델 학습 실패: {e}")

    _model_exists = any(
        os.path.exists(p) for p in [
            ml_engine.MODEL_PATH,
            *[fb[0] for fb in ml_engine.FALLBACK_PATHS],
        ]
    )
    if not _model_exists:
        log("🚨 [ML] 모델 파일이 전혀 없습니다! ML_SCORE=0으로 진행됩니다.")
        log("   → data/에 ohlcv_cache_*.pkl 학습 데이터가 있는지 확인하세요.")

    from collector import PYKRX_OK
    if not PYKRX_OK:
        log("⚠️ pykrx 미설치 → FDR 폴백으로 진행합니다.")


def load_reference_data(
    trade_date: Optional[str] = None,
    top_n: Optional[int] = None,
) -> PipelineContext:
    """Stage 1: 거래일, 시총, 벤치마크, 종목풀, OHLCV, 수급 로딩"""
    from collector import (
        log, resolve_trade_date, build_mcap_map, get_mcap_eok_from_map,
        get_benchmark_returns, pick_top_by_trading_value,
        get_market_sets, get_name_map_cached, save_price_snapshot,
        build_sector_map, prepare_ohlcv_data, fetch_investor_net_buying,
        RunContext, OUT_DIR, TOP_N, LOOKBACK_DAYS, MIN_MCAP_EOK,
    )
    from macro_filter import check_macro_env

    ctx = PipelineContext()

    # 1) 거래 기준일
    ctx.trade_ymd = resolve_trade_date(trade_date)

    # 2) 매크로 필터
    ctx.macro_risk, ctx.macro_msg, new_ebs, ctx.rec_limit = check_macro_env(ctx.trade_ymd)
    ctx.pass_ebs = new_ebs
    log(f"⚙️ 매크로 필터 적용: PASS_EBS={ctx.pass_ebs}, Telegram_Limit={ctx.rec_limit}")

    # 3) 시총 맵
    ctx.mcap_map, ctx.mcap_ymd = build_mcap_map(ctx.trade_ymd)
    log(f"📅 거래 기준일: {ctx.trade_ymd} (mcap ref: {ctx.mcap_ymd})")

    # 4) 벤치마크
    ctx.bench_map = get_benchmark_returns(ctx.trade_ymd)

    # [v20.0] OHLCV 캐시 기반 벤치마크 역산 폴백
    if not ctx.bench_map or not ctx.bench_map.get("KOSPI", {}).get(60):
        try:
            from glob import glob as _glob
            _ohlcv_cache_path = os.path.join(OUT_DIR, f"ohlcv_cache_{ctx.trade_ymd}.parquet")
            if not os.path.exists(_ohlcv_cache_path):
                _candidates = sorted(_glob(os.path.join(OUT_DIR, "ohlcv_cache_*.parquet")), reverse=True)
                if _candidates:
                    _ohlcv_cache_path = _candidates[0]

            if os.path.exists(_ohlcv_cache_path):
                _ohlcv_all = pd.read_parquet(_ohlcv_cache_path)
                if "종가" in _ohlcv_all.columns and "종목코드" in _ohlcv_all.columns:
                    _rets = {}
                    for _code, _grp in _ohlcv_all.groupby("종목코드"):
                        _c = pd.to_numeric(_grp["종가"], errors="coerce").dropna()
                        if len(_c) > 60:
                            _rets[_code] = (float(_c.iloc[-1]) / float(_c.iloc[-61]) - 1) * 100
                    if _rets:
                        _median_ret = float(np.median(list(_rets.values())))
                        if "KOSPI" not in ctx.bench_map or not ctx.bench_map.get("KOSPI", {}).get(60):
                            ctx.bench_map["KOSPI"] = {20: round(_median_ret * 0.33, 2), 60: round(_median_ret, 2), 120: round(_median_ret * 1.8, 2)}
                        if "KOSDAQ" not in ctx.bench_map or not ctx.bench_map.get("KOSDAQ", {}).get(60):
                            ctx.bench_map["KOSDAQ"] = {20: round(_median_ret * 0.33, 2), 60: round(_median_ret, 2), 120: round(_median_ret * 1.8, 2)}
                        log(f"📂 [폴백] OHLCV 캐시 기반 벤치마크 역산: 중앙값 {_median_ret:.2f}% ({len(_rets)}종목)")
        except Exception as _be:
            log(f"⚠️ OHLCV 벤치마크 폴백 실패: {_be}")

    def _fmt(v): return f"{v:.2f}%" if isinstance(v, (int, float)) else "N/A"
    k_60 = ctx.bench_map.get("KOSPI", {}).get(60)
    q_60 = ctx.bench_map.get("KOSDAQ", {}).get(60)
    log(f"📈 벤치마크(60d): KOSPI {_fmt(k_60)}, KOSDAQ {_fmt(q_60)}")

    # 5) 거래대금 상위 종목
    use_top_n = top_n or TOP_N
    log(f"📊 거래대금 상위 N: {use_top_n}")
    ctx.top_df = pick_top_by_trading_value(ctx.trade_ymd, use_top_n)

    # 6) 시총 프리필터
    if ctx.mcap_map:
        ctx.top_df["시가총액(억원)"] = ctx.top_df["종목코드"].map(
            lambda c: get_mcap_eok_from_map(ctx.mcap_map, c)
        )
        before_cnt = len(ctx.top_df)
        s_mcap = pd.to_numeric(ctx.top_df["시가총액(억원)"], errors="coerce").fillna(0)
        top_df_f = ctx.top_df[s_mcap >= MIN_MCAP_EOK].copy()
        after_cnt = len(top_df_f)
        log(f"📊 시총 필터 적용: {before_cnt} → {after_cnt}개 (기준 {MIN_MCAP_EOK}억)")
        if after_cnt == 0 and before_cnt > 0:
            relaxed = MIN_MCAP_EOK / 10
            log(f"⚠️ 시총 필터 결과 0개 → 임시 기준 완화 ({relaxed}억)")
            top_df_f = ctx.top_df[s_mcap >= relaxed].copy()
        ctx.top_df = top_df_f
    else:
        log("⚠️ mcap_map 비어 있음 → 시총 사전 필터 생략")
        ctx.top_df["시가총액(억원)"] = 0.0

    # 7) 공통 데이터
    ctx.tickers = ctx.top_df["종목코드"].tolist()
    ctx.kospi_set, ctx.kosdaq_set = get_market_sets(ctx.trade_ymd)
    ctx.name_map = get_name_map_cached(ctx.trade_ymd)
    save_price_snapshot(ctx.trade_ymd, ctx.name_map)
    ctx.sector_map = build_sector_map()

    start_dt = datetime.strptime(ctx.trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS * 2 + 60)
    ctx.start_s = start_dt.strftime("%Y%m%d")
    ctx.end_s = ctx.trade_ymd

    # 8) OHLCV 일괄 수집
    ctx.ohlcv_map = prepare_ohlcv_data(ctx.tickers, ctx.start_s, ctx.end_s, ctx.trade_ymd)

    # 9) 수급 조기 fetch
    try:
        map_frg, map_inst, map_ant = fetch_investor_net_buying(ctx.trade_ymd)
        ctx.inv_maps = {"frg": map_frg, "inst": map_inst, "ant": map_ant}
        log(f"📊 [수급] 조기 로드 완료: 외인 {len(map_frg)}건, 기관 {len(map_inst)}건")
    except Exception as e:
        log(f"⚠️ [수급] 조기 로드 실패: {e} → 수급 보정 없이 진행")
        ctx.inv_maps = None

    return ctx
