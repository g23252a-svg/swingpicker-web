"""
services/backtest_policy.py
===========================
[v3.9.15e + 10] 백테스트 anomaly 정책 SSOT.

전략 검증 임계값을 한 곳에서 관리. tab_backtest UI 컴포넌트에서
import해서 사용하지만, 같은 임계값을 다음에서도 공유 예정:
- v3.9.16 프리셋 비교표 (4프리셋 anomaly 검증)
- v3.9.17 강건성 테스트 (27개 조합 anomaly 비율 추적)
- v3.9.18 Train/Test 분할 (out-of-sample anomaly 비교)
- combo_optimizer.py (백테스트 grid search 시 anomaly 차단)

이전엔 tab_backtest.py 모듈 top-level에 inline 상수로 박혀있어서 다른
모듈에서 import 시 UI 의존성(nicegui)이 따라옴. 이 모듈로 분리하면
service layer만 import해서 정책값 공유 가능.

설계 원칙:
- 상수는 모듈 top-level (튜닝 시 한 곳에서 조정)
- 검출 로직은 함수로 (재사용 가능)
- UI 의존성 0 (nicegui/plotly import 없음)
- check_contract_gate가 임계값 변경을 감지할 수 있도록 명시적 상수명 유지
"""
from __future__ import annotations

from typing import List, Optional


# ═══════════════════════════════════════════════════════════════
# anomaly 절대 임계 (기간 무관)
# ═══════════════════════════════════════════════════════════════
ANOMALY_TOTAL_RET_ABS = 300        # %: 누적 수익률 비정상 임계
ANOMALY_SHARPE_MAX = 5              # Sharpe ratio 비정상 임계 (헤지펀드 최상위도 2~3)
ANOMALY_CAGR_MAX = 300              # %: CAGR 연환산 비정상 임계


# ═══════════════════════════════════════════════════════════════
# anomaly 기간 가중 임계 (단기 백테스트일수록 엄격)
# ═══════════════════════════════════════════════════════════════
ANOMALY_SHORT_DAYS_RET = 120        # 영업일: 6개월 미만 = 단기 (수익률 가중용)
ANOMALY_SHORT_RET = 100             # %: 단기 백테스트의 수익률 의심 임계
ANOMALY_SHORT_DAYS_CAGR = 252       # 영업일: 1년 미만 = 단기 (CAGR 가중용)


# ═══════════════════════════════════════════════════════════════
# TP 포화율 임계 — 익절선(target_pct) tier별 차등
# ═══════════════════════════════════════════════════════════════
# 익절선 낮을수록 (단타) 포화율 높아도 자연스러움
# 익절선 높을수록 (공격) 포화율 낮아도 의심
TP_SAT_THRESH_LOW_TARGET = 80       # target_pct ≤ 5  (단타/소익절): 80%+ 시 경고
TP_SAT_THRESH_MID_TARGET = 70       # 5 < target_pct < 10 (균형): 70%+ 시 경고
TP_SAT_THRESH_HIGH_TARGET = 60      # target_pct ≥ 10 (공격/큰익절): 60%+ 시 경고


def tp_saturation_threshold(target_pct: float) -> int:
    """target_pct tier별 TP 포화율 경고 임계값 반환.

    +3% 단타에서 WIN 80%는 자연스러움 (작은 익절은 자주 도달).
    +20% 공격에서 WIN 70%는 비현실적 — ret_NNd_% 사후 수익률의 한계가
    크게 드러남.

    Args:
        target_pct: 익절선 (% 단위). cfg["target_pct"] 그대로 전달.

    Returns:
        포화율 % 임계값 (60 / 70 / 80 중 하나)
    """
    if target_pct >= 10:
        return TP_SAT_THRESH_HIGH_TARGET
    if target_pct >= 6:
        return TP_SAT_THRESH_MID_TARGET
    return TP_SAT_THRESH_LOW_TARGET


def detect_anomaly_flags(
    *,
    total_ret: float,
    sharpe_val: Optional[float],
    cagr_val: Optional[float],
    trading_days: int,
) -> List[str]:
    """백테스트 결과의 anomaly 사유 목록 반환.

    UI 비의존 — tab_backtest 뿐 아니라 프리셋 비교표(v3.9.16),
    강건성 테스트(v3.9.17), Train/Test 분할(v3.9.18)에서도 같은 로직 공유.

    검출 룰:
    - 절대 임계: total_ret > 300, sharpe > 5, cagr > 300
    - 기간 가중: trading_days < 120 + total_ret > 100 → 단기 과대수익
    - 기간 가중: trading_days < 252 + cagr > 300    → 단기 CAGR 폭주

    Args:
        total_ret:    누적 수익률 (%)
        sharpe_val:   Sharpe ratio (None 가능)
        cagr_val:     CAGR (%, None 가능)
        trading_days: 거래일 수 (백테스트 기간)

    Returns:
        anomaly 사유 문자열 리스트 (중복 제거 후). 비어있으면 정상.
        메시지는 raw 수치 노출 안 함 — "{ANOMALY_X}% 초과" 형식.
    """
    flags: List[str] = []

    # 절대 임계
    if total_ret > ANOMALY_TOTAL_RET_ABS:
        flags.append(f"누적 수익률 비정상 ({ANOMALY_TOTAL_RET_ABS}% 초과)")
    if sharpe_val is not None and sharpe_val > ANOMALY_SHARPE_MAX:
        flags.append(f"Sharpe 비정상 ({ANOMALY_SHARPE_MAX} 초과)")
    if cagr_val is not None and cagr_val > ANOMALY_CAGR_MAX:
        flags.append(f"CAGR 비정상 ({ANOMALY_CAGR_MAX}% 초과)")

    # 기간 가중
    if (
        trading_days > 0
        and trading_days < ANOMALY_SHORT_DAYS_RET
        and total_ret > ANOMALY_SHORT_RET
    ):
        flags.append(
            f"단기 과대수익 ({trading_days}일에 {ANOMALY_SHORT_RET}%+)"
        )
    if (
        trading_days > 0
        and trading_days < ANOMALY_SHORT_DAYS_CAGR
        and cagr_val is not None
        and cagr_val > ANOMALY_CAGR_MAX
    ):
        flags.append(f"단기 CAGR 폭주 ({trading_days}일)")

    # 중복 제거 (순서 보존)
    seen = set()
    return [f for f in flags if not (f in seen or seen.add(f))]
