# -*- coding: utf-8 -*-
"""
collector_config.py — 도메인별 분리 설정 (SRP 적용)
═══════════════════════════════════════════════════════
[v3.3] God Config 해체 → 7개 도메인 Dataclass + Composition

설계 원칙:
  1. SRP: 각 도메인(데이터/지표/패널티/매크로/슬리피지/시스템/시크릿)이
     독립적인 Dataclass → 그리드 서치 시 IndicatorConfig만 교체 가능
  2. 검증: __post_init__에서 논리적 오류 즉시 차단 (GIGO 방지)
  3. 격리: 민감 정보(SecretsConfig)는 전략 스냅샷에서 완전 배제
  4. 호환: CollectorConfig.__getattr__ 위임으로 기존 _CFG.bb_period 접근 유지

사용 예:
  # 기존과 동일하게 사용
  from collector_config import DEFAULT_CONFIG
  print(DEFAULT_CONFIG.bb_period)  # → 20

  # 그리드 서치: 지표 파라미터만 교체
  custom = CollectorConfig(indicator=IndicatorConfig(bb_period=15, bb_std=2.5))

  # 백테스트: 패널티 조정
  loose = CollectorConfig(scoring=ScoringWeights(p_overheat_5d=3.0, p_rsi_out=2.0))
"""
import os
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════
#  매직 스트링 Enum화 — 오타 방지 + IDE 자동완성
# ═══════════════════════════════════════════════════

class Route(str, Enum):
    """종목 상태 머신(State Machine) — determine_state_dynamic 반환값"""
    ATTACK       = "ATTACK"
    ARMED        = "ARMED"
    WAIT         = "WAIT"
    NEUTRAL      = "NEUTRAL"
    OVERHEAT     = "OVERHEAT"
    EXIT_WARNING = "EXIT_WARNING"
    CARRY        = "CARRY"
    BLOCKED      = "BLOCKED"


class Market(str, Enum):
    """거래소 구분"""
    KOSPI  = "KOSPI"
    KOSDAQ = "KOSDAQ"


# ═══════════════════════════════════════════════════
#  1. DataConfig — 데이터 수집 경계값
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class DataConfig:
    """OHLCV 수집, 필터링, 캐시 관련 설정"""
    lookback_days: int = 250
    bench_lookback_days: int = 60
    top_n: int = 600
    min_turnover_eok: int = 50
    min_mcap_eok: int = 1000
    max_workers: int = 4
    pass_ebs: int = 4
    cache_format: str = "parquet"

    def __post_init__(self):
        if self.lookback_days < 60:
            raise ValueError(f"lookback_days={self.lookback_days}: 최소 60일 필요 (지표 계산)")
        if self.min_turnover_eok < 0:
            raise ValueError(f"min_turnover_eok={self.min_turnover_eok}: 음수 불가")
        if self.min_mcap_eok < 0:
            raise ValueError(f"min_mcap_eok={self.min_mcap_eok}: 음수 불가")
        if self.max_workers < 1:
            raise ValueError(f"max_workers={self.max_workers}: 최소 1")
        if self.cache_format not in ("pickle", "parquet"):
            raise ValueError(f"cache_format='{self.cache_format}': 'pickle' 또는 'parquet'만 허용")


# ═══════════════════════════════════════════════════
#  2. IndicatorConfig — 기술적 지표 파라미터
#     (그리드 서치의 주요 대상)
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class IndicatorConfig:
    """볼린저밴드, RSI, Keltner Channel, Squeeze, Trigger, MTF 파라미터"""
    # RSI 적정 구간
    rsi_low: float = 45.0
    rsi_high: float = 65.0
    rsi_overheat: float = 75.0

    # EBS / Timing / ROUTE 임계치 (scoring_engine.py 연동)
    vol_quality_min: float = 1.1             # EBS Vol_Quality 기준
    rsi_penalty_threshold: float = 75.0      # TIMING_SCORE RSI 패널티 기준
    gap_pct_penalty_threshold: float = 5.0   # TIMING_SCORE 갭 패널티 기준
    timing_attack_threshold: float = 60.0    # ROUTE ATTACK 판정 TIMING_SCORE 기준
    vol_quality_attack: float = 1.3          # ROUTE ATTACK 판정 Vol_Quality 기준

    # 볼린저밴드 / TTM Squeeze
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_bw: float = 10.0
    kc_period: int = 20
    kc_atr_period: int = 20
    kc_mult: float = 1.5
    bonus_bb_squeeze_score: float = 3.0
    bonus_bb_squeeze_entry: float = 4.0

    # Trigger Score 임계치
    trigger_dist_wick_high: float = 0.35
    trigger_dist_wick_mid: float = 0.25
    trigger_ret_pct_dist: float = 5.0
    trigger_ret_pct_weak: float = 3.0
    trigger_range_pos_weak: float = 0.6
    trigger_vol_ratio_spike: float = 3.0

    # Multi-Timeframe (v15)
    mtf_struct_bonus: float = 10.0
    mtf_struct_penalty: float = 15.0
    mtf_min_weekly_bars: int = 26
    mtf_min_monthly_bars: int = 12

    def __post_init__(self):
        if self.rsi_low >= self.rsi_high:
            raise ValueError(
                f"rsi_low({self.rsi_low}) >= rsi_high({self.rsi_high}): "
                f"적정 구간의 하한이 상한보다 클 수 없음"
            )
        if self.rsi_high >= self.rsi_overheat:
            raise ValueError(
                f"rsi_high({self.rsi_high}) >= rsi_overheat({self.rsi_overheat}): "
                f"적정 상한이 과열 기준보다 클 수 없음"
            )
        if self.bb_std <= 0:
            raise ValueError(f"bb_std={self.bb_std}: 양수여야 함")
        if self.bb_period < 5:
            raise ValueError(f"bb_period={self.bb_period}: 최소 5")
        if self.kc_mult <= 0:
            raise ValueError(f"kc_mult={self.kc_mult}: 양수여야 함")
        if self.trigger_dist_wick_high <= self.trigger_dist_wick_mid:
            raise ValueError(
                f"trigger_dist_wick_high({self.trigger_dist_wick_high}) "
                f"<= trigger_dist_wick_mid({self.trigger_dist_wick_mid})"
            )

    @property
    def rsi_range(self) -> Tuple[float, float]:
        return (self.rsi_low, self.rsi_high)


# ═══════════════════════════════════════════════════
#  3. ScoringWeights — 점수 가중치 + 패널티
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class ScoringWeights:
    """FINAL_SCORE 가중치 + 감점 패널티
    
    가중치 합계는 __post_init__에서 1.0 ± 0.02 검증.
    새 패널티 항목 추가 시 p_ 접두사 + __post_init__ 자동 검증.
    """
    # 가중치 (합계 = 1.0)
    w_rr: float = 0.25
    w_t1: float = 0.18
    w_sl: float = 0.12
    w_near: float = 0.12
    w_mom: float = 0.10
    w_liq: float = 0.13
    w_tec: float = 0.10
    w_sector: float = 0.05

    # 패널티 (양수 = 감점)
    p_overheat_5d: float = 6.0
    p_overheat_10d: float = 6.0
    p_rsi_out: float = 4.0
    p_macd_neg: float = 4.0
    p_near_far: float = 4.0
    p_liq_low: float = 4.0
    p_vol_spike: float = 2.0
    p_big_sl: float = 3.0

    def __post_init__(self):
        # 핵심 가중치 합계 검증 (w_sector는 별도 보너스 → 합계에서 제외)
        w_sum = (self.w_rr + self.w_t1 + self.w_sl + self.w_near
                 + self.w_mom + self.w_liq + self.w_tec)
        if abs(w_sum - 1.0) > 0.02:
            raise ValueError(
                f"핵심 가중치 합계={w_sum:.4f} ≠ 1.0 (허용: ±0.02). "
                f"w_rr={self.w_rr}, w_t1={self.w_t1}, w_sl={self.w_sl}, "
                f"w_near={self.w_near}, w_mom={self.w_mom}, w_liq={self.w_liq}, "
                f"w_tec={self.w_tec} (w_sector={self.w_sector}은 별도 보너스)"
            )
        # 패널티 음수 방지 (자동 순회 — 확장 시 추가 코드 불필요)
        for fname in dataclasses.fields(self):
            if fname.name.startswith("p_"):
                val = getattr(self, fname.name)
                if val < 0:
                    raise ValueError(f"패널티 {fname.name}={val}: 음수 불가 (감점은 양수)")


# ═══════════════════════════════════════════════════
#  4. MacroConfig — 매크로 환경 임계치
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class MacroConfig:
    """환율/나스닥 임계치, ML 가중치, 추천 제한"""
    fx_caution: float = 1470.0
    fx_critical: float = 1490.0
    nasdaq_caution: float = -1.5
    nasdaq_critical: float = -2.5

    rec_limit_default: int = 5
    rec_limit_caution: int = 3

    # ML/매크로 동적 가중치
    ml_low: float = 5.0
    ml_high: float = 25.0
    ml_max_weight: float = 0.20
    ml_cov_gate: float = 0.20
    trim_pct: float = 0.10
    ebs_pass_threshold: int = 3
    macro_weights: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "CRITICAL": (0.55, 0.25),
        "HIGH":     (0.50, 0.30),
        "NORMAL":   (0.40, 0.40),
    })

    def __post_init__(self):
        if self.fx_caution >= self.fx_critical:
            raise ValueError(
                f"fx_caution({self.fx_caution}) >= fx_critical({self.fx_critical})"
            )
        if self.nasdaq_caution <= self.nasdaq_critical:
            raise ValueError(
                f"nasdaq_caution({self.nasdaq_caution}) <= nasdaq_critical({self.nasdaq_critical})"
            )
        if self.ml_low >= self.ml_high:
            raise ValueError(f"ml_low({self.ml_low}) >= ml_high({self.ml_high})")
        if not (0 < self.ml_max_weight <= 1.0):
            raise ValueError(f"ml_max_weight={self.ml_max_weight}: (0, 1.0] 범위")
        if self.rec_limit_caution > self.rec_limit_default:
            raise ValueError(
                f"rec_limit_caution({self.rec_limit_caution}) > "
                f"rec_limit_default({self.rec_limit_default})"
            )

    # 기존 접두사 호환 (macro_filter.py: config.macro_fx_caution)
    @property
    def macro_fx_caution(self) -> float: return self.fx_caution
    @property
    def macro_fx_critical(self) -> float: return self.fx_critical
    @property
    def macro_nasdaq_caution(self) -> float: return self.nasdaq_caution
    @property
    def macro_nasdaq_critical(self) -> float: return self.nasdaq_critical


# ═══════════════════════════════════════════════════
#  5. SlippageConfig — 슬리피지/유동성 모델
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class SlippageConfig:
    """거래대금 기반 동적 슬리피지 + 유동성 감점"""
    slippage_base_bps: float = 10.0
    slippage_low_liq_mult: float = 3.0
    slippage_liq_threshold_eok: float = 20.0

    liq_penalty_very_low_eok: float = 5.0
    liq_penalty_very_low_pts: float = 8.0
    liq_penalty_low_eok: float = 10.0
    liq_penalty_low_pts: float = 4.0

    def __post_init__(self):
        if self.slippage_base_bps <= 0:
            raise ValueError(f"slippage_base_bps={self.slippage_base_bps}: 양수여야 함")
        if self.slippage_low_liq_mult < 1.0:
            raise ValueError(f"slippage_low_liq_mult={self.slippage_low_liq_mult}: 1.0 이상")
        if self.liq_penalty_very_low_eok >= self.liq_penalty_low_eok:
            raise ValueError(
                f"liq_penalty_very_low_eok({self.liq_penalty_very_low_eok}) >= "
                f"liq_penalty_low_eok({self.liq_penalty_low_eok})"
            )


# ═══════════════════════════════════════════════════
#  6. TimeStopConfig — 시간 청산 규칙
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class TimeStopConfig:
    """N영업일 무반응 시 청산"""
    time_stop_days: int = 7
    time_stop_min_move_pct: float = 2.0
    time_stop_extend_if_profit: bool = True

    def __post_init__(self):
        if self.time_stop_days < 0:
            raise ValueError(f"time_stop_days={self.time_stop_days}: 0 이상 (0=비활성)")
        if self.time_stop_min_move_pct < 0:
            raise ValueError(f"time_stop_min_move_pct={self.time_stop_min_move_pct}: 음수 불가")


# ═══════════════════════════════════════════════════
#  7. SecretsConfig — 민감 정보 (스냅샷 완전 배제)
# ═══════════════════════════════════════════════════

@dataclass
class SecretsConfig:
    """API 키, 토큰 등 — 전략 파라미터와 생명주기가 다름.
    
    환경변수에서 주입받으며, snapshot()에 절대 포함되지 않음.
    frozen=False: 런타임에 토큰 갱신 가능.
    """
    tg_token: str = field(default_factory=lambda: os.environ.get("TG_TOKEN", ""))
    tg_id: str = field(default_factory=lambda: os.environ.get("TG_ID", ""))


# ═══════════════════════════════════════════════════
#  CollectorConfig — Facade (Composition + 하위 호환)
# ═══════════════════════════════════════════════════

class CollectorConfig:
    """7개 도메인 Config의 조합 (Facade 패턴).
    
    기존 코드 호환:
      _CFG.bb_period      → __getattr__ → indicator.bb_period
      _CFG.tg_token       → __getattr__ → secrets.tg_token
      _CFG.snapshot()     → secrets 제외된 전략 스냅샷
    
    그리드 서치:
      custom = CollectorConfig(indicator=IndicatorConfig(bb_period=15, bb_std=2.5))
    
    백테스트:
      loose = CollectorConfig(scoring=ScoringWeights(p_overheat_5d=3.0, p_rsi_out=2.0))
    """

    __slots__ = (
        "data", "indicator", "scoring", "macro",
        "slippage", "time_stop", "secrets",
        "base_dir", "config_version",
        "_sub_configs",
    )

    def __init__(
        self,
        data: DataConfig = None,
        indicator: IndicatorConfig = None,
        scoring: ScoringWeights = None,
        macro: MacroConfig = None,
        slippage: SlippageConfig = None,
        time_stop: TimeStopConfig = None,
        secrets: SecretsConfig = None,
        base_dir: str = None,
        config_version: str = "2.0.0",
    ):
        self.data = data or DataConfig()
        self.indicator = indicator or IndicatorConfig()
        self.scoring = scoring or ScoringWeights()
        self.macro = macro or MacroConfig()
        self.slippage = slippage or SlippageConfig()
        self.time_stop = time_stop or TimeStopConfig()
        self.secrets = secrets or SecretsConfig()
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.config_version = config_version

        # __getattr__ 탐색 순서 (secrets는 마지막 — 명시적 접근 권장)
        self._sub_configs = (
            self.data, self.indicator, self.scoring,
            self.macro, self.slippage, self.time_stop,
            self.secrets,
        )

    def __getattr__(self, name: str):
        """기존 _CFG.bb_period 접근 패턴 하위 호환.
        
        탐색: data → indicator → scoring → macro → slippage → time_stop → secrets
        """
        for sub in self._sub_configs:
            try:
                return getattr(sub, name)
            except AttributeError:
                continue
        raise AttributeError(f"CollectorConfig에 '{name}' 필드 없음")

    # ── 프로퍼티 (기존 호환) ──

    @property
    def out_dir(self) -> str:
        return os.path.join(self.base_dir, "data")

    @property
    def output_dir(self) -> str:
        return self.out_dir

    @property
    def rsi_range(self) -> Tuple[float, float]:
        return self.indicator.rsi_range

    # ── 스냅샷 (민감 정보 완전 배제) ──

    def snapshot(self) -> dict:
        """전략 파라미터의 재현 가능한 스냅샷 (SecretsConfig, base_dir 배제)."""
        from datetime import datetime
        d = {}
        for sub_name in ("data", "indicator", "scoring", "macro", "slippage", "time_stop"):
            sub = getattr(self, sub_name)
            sub_dict = dataclasses.asdict(sub)
            d.update(sub_dict)
        d["config_version"] = self.config_version
        d["_snapshot_ts"] = datetime.now().isoformat()
        return d

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, default=str)

    def __repr__(self) -> str:
        return (
            f"CollectorConfig(v{self.config_version}, "
            f"data={self.data!r}, indicator={self.indicator!r}, "
            f"scoring={self.scoring!r}, macro={self.macro!r})"
        )


# ═══════════════════════════════════════════════════
#  싱글턴 기본 인스턴스
# ═══════════════════════════════════════════════════

DEFAULT_CONFIG = CollectorConfig()
