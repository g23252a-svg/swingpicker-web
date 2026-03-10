# -*- coding: utf-8 -*-
"""
pipeline_context.py — 파이프라인 공유 데이터 컨테이너
═══════════════════════════════════════════════════════
[v20.1] collector.py 분해의 핵심 — 모든 Stage가 이 객체를 통해 소통
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd


@dataclass
class PipelineContext:
    """파이프라인 전 단계가 공유하는 데이터 컨테이너"""

    # ── Stage 1: 참조 데이터 ──
    trade_ymd: str = ""
    mcap_ymd: str = ""
    macro_risk: str = "NORMAL"
    macro_msg: str = ""
    pass_ebs: int = 4
    rec_limit: int = 5
    mcap_map: Dict[str, float] = field(default_factory=dict)
    bench_map: Dict[str, Dict[int, float]] = field(default_factory=dict)
    tickers: List[str] = field(default_factory=list)
    top_df: Optional[pd.DataFrame] = None
    kospi_set: set = field(default_factory=set)
    kosdaq_set: set = field(default_factory=set)
    name_map: Dict[str, str] = field(default_factory=dict)
    sector_map: Dict[str, str] = field(default_factory=dict)
    ohlcv_map: Dict[str, pd.DataFrame] = field(default_factory=dict)
    inv_maps: Optional[Dict] = None
    start_s: str = ""
    end_s: str = ""

    # ── Stage 2+: 분석/스코어링 결과 ──
    df_out: Optional[pd.DataFrame] = None
    breadth: Dict[str, float] = field(default_factory=dict)

    # ── 실행 메타 ──
    run_health: Optional[Any] = None
    enable_telegram: bool = True
    tag: Optional[str] = None
