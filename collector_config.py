# -*- coding: utf-8 -*-
"""
collector_config.py — 모든 매직넘버/임계치 중앙화
═══════════════════════════════════════════════════
[v14] #11 개선과제: 수십 개의 하드코딩 상수를 단일 Config로 집결.
- 튜닝 시 이 파일만 수정
- 백테스트 시 Config 인스턴스만 교체하면 그리드 서치 가능
- collector.py, scoring_engine.py 등에서 import하여 사용
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class CollectorConfig:
    """collector 파이프라인 전체 설정"""

    # ── 데이터 수집 ──
    lookback_days: int = 250
    bench_lookback_days: int = 60
    top_n: int = 600
    min_turnover_eok: int = 50
    min_mcap_eok: int = 1000
    max_workers: int = 4

    # ── RSI 적정 구간 ──
    rsi_low: float = 45.0
    rsi_high: float = 65.0
    rsi_overheat: float = 75.0

    # ── EBS ──
    pass_ebs: int = 4

    # ── 볼린저밴드 / TTM Squeeze ──
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_bw: float = 10.0
    kc_period: int = 20
    kc_atr_period: int = 20
    kc_mult: float = 1.5
    bonus_bb_squeeze_score: float = 3.0
    bonus_bb_squeeze_entry: float = 4.0

    # ── 가중치 (기존 FINAL_SCORE용) ──
    w_rr: float = 0.25
    w_t1: float = 0.18
    w_sl: float = 0.12
    w_near: float = 0.12
    w_mom: float = 0.10
    w_liq: float = 0.13
    w_tec: float = 0.10

    # ── 패널티 ──
    p_overheat_5d: float = 6.0
    p_overheat_10d: float = 6.0
    p_rsi_out: float = 4.0
    p_macd_neg: float = 4.0
    p_near_far: float = 4.0
    p_liq_low: float = 4.0
    p_vol_spike: float = 2.0
    p_big_sl: float = 3.0

    # ── 섹터 ──
    w_sector: float = 0.05

    # ── Multi-Timeframe (v15) ──
    mtf_struct_bonus: float = 10.0     # 주봉+월봉 모두 상승 시 STRUCT 가점
    mtf_struct_penalty: float = 15.0   # 주봉+월봉 모두 하락 시 STRUCT 감점
    mtf_min_weekly_bars: int = 26      # 주봉 최소 바 수 (6개월)
    mtf_min_monthly_bars: int = 12     # 월봉 최소 바 수 (1년)

    # ── 매크로 환경 ──
    macro_fx_caution: float = 1400.0
    macro_fx_critical: float = 1420.0
    macro_nasdaq_caution: float = -1.5
    macro_nasdaq_critical: float = -2.5

    # ── 추천 수 ──
    rec_limit_default: int = 5
    rec_limit_caution: int = 3

    # ── 텔레그램 ──
    tg_token: str = field(default_factory=lambda: os.environ.get("TG_TOKEN", ""))
    tg_id: str = field(default_factory=lambda: os.environ.get("TG_ID", ""))

    # ── TRIGGER_SCORE ──
    trigger_dist_wick_high: float = 0.35
    trigger_dist_wick_mid: float = 0.25
    trigger_ret_pct_dist: float = 5.0
    trigger_ret_pct_weak: float = 3.0
    trigger_range_pos_weak: float = 0.6
    trigger_vol_ratio_spike: float = 3.0

    # ── 캐시 ──
    cache_format: str = "parquet"  # "pickle" or "parquet"

    # ── 디렉토리 ──
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    @property
    def out_dir(self) -> str:
        return os.path.join(self.base_dir, "data")

    @property
    def rsi_range(self) -> Tuple[float, float]:
        return (self.rsi_low, self.rsi_high)


# ── 싱글턴 기본 인스턴스 ──
DEFAULT_CONFIG = CollectorConfig()

# ── BacktestConfig 참조 (auto_backtest.py SSOT) ──
# from auto_backtest import BacktestConfig, DEFAULT_BT_CONFIG
# 순환 방지: auto_backtest가 collector_config를 import하므로,
# 여기서는 참조 주석만 남김. 실제 사용은 auto_backtest에서 직접.
