# -*- coding: utf-8 -*-
"""
market_regime.py — [v28] 시장 레짐 판정 (상승/중립/하락)
═══════════════════════════════════════════════════
목적:
  지수 방향과 시장 내부(시장폭)를 결합해 "지금이 어떤 장인가"를
  하나의 명시적 상태로 판정하고, 그 상태가 신규 진입 허용/차단과
  포지션 사이즈를 결정하게 한다.

설계 근거 (2026-02-26 ~ 2026-07-13, 98거래일 실측):
  - UP 레짐   : 게이트 통과 픽 평균 +5.1%/건 (n=22) — 알파의 대부분
  - NEUTRAL   : 평균 +0.1%/건 (n=14) — 본전 수준 → 사이즈 절반
  - DOWN      : 진입 자체가 손실 구간 → 차단 (v27 시장폭 게이트 일반화)
  - 랭킹은 레짐과 무관하게 POC 근접 순이 우월했다
    (UP 레짐에서도 점수순 +3.0% vs POC순 +5.1%) → 레짐은 GO/NO-GO와
    사이즈만 결정하고 랭킹에는 개입하지 않는다.

판정 규칙 (lookahead 없음 — t일 종가까지만 사용):
  UP      : KOSPI 종가 > MA20 AND 종가 > MA5 AND 시장폭 ≥ 50
  DOWN    : 시장폭 < 35 (지수 위치 무관 — 내부 붕괴 우선)
  NEUTRAL : 그 외
  UNKNOWN : 지수/시장폭 데이터 부족 → 보수적으로 NEUTRAL 취급

═══════════════════════════════════════════════════
[v66] 이 모듈이 화면에 거짓을 적고 있었다 — 문구를 사실에 맞춘다
═══════════════════════════════════════════════════
■ 무엇이 있었나 (2026-08-18·08-19 배치 실측)
    MARKET_REGIME      = DOWN
    REGIME_ALLOW_ENTRY = 0
    REGIME_REASON      = "... 신규 진입 차단 (실측: 이 구간 진입은 손실 우세)"
    REGIME_SIZE_MULT   = 0.3
  그런데 **같은 배치가 공식 매수 1건 + 사이징된 후보 28건**을 냈다.
  다음 날(8/19) 코스피는 -5.80% 폭락했고 사용자는 이 목록을 보고 매수했다.

■ 왜 그렇게 됐나 — 선언만 있고 집행이 없다
  1) `REGIME_ALLOW_ENTRY`는 이 파일에서 **쓰기만** 하고 읽는 코드가 0건이다.
  2) `regime_ok`(DOWN 거부) 검사는 recommendation_quality에 있지만 **레거시
     (비알파) 경로에만** 있다. v32가 알파 경로에서 의도적으로 뺐고 그 사유를
     주석에 남겼다. 알파 엔진이 켜진 지금 이 거부권은 **죽은 게이트**다.
  3) `REGIME_SIZE_MULT`는 `RECOMMENDED_WEIGHT_PCT`(표시용 %) 하나에만 곱한다.
     사용자가 실제로 보고 사는 `켈리_수량`은 이 배수를 보지 않는다.
     실측: 8/18(배수 0.3) 공식픽 실제 투입 **95.0만원** vs
           8/20(배수 1.0) 공식픽 실제 투입 **81.0만원** — 배수가 큰 날이 더 작다.
  전수: 진입 차단 선언 9일 중 사이징 후보가 나온 날 4일, 그중 8/18·8/19가
  28건·24건으로 압도적이다(나머지 7일은 0~3건).

■ 문구의 근거 자체도 틀렸다
  "실측: 이 구간 진입은 손실 우세"를 현재 데이터로 재측정하면 반대다.
  게이트 통과 상위5의 5일 실현수익(일별 평균 · 유니버스 대비 초과):
    DOWN     6일  +4.15%  초과 +2.29%p (t=+0.99, p=0.37) 승률 67%
    NEUTRAL 14일  +1.23%  초과 -0.63%p (t=-0.74, p=0.48) 승률 57%
    UP       1일  (표본 부족)
  둘 다 유의하지 않지만 **부호가 문구와 반대**다. v32의 판단(DOWN 하드블록
  제거)은 이 표본에서 뒤집히지 않는다. 한계: 이 표본은 8/12까지이고 이번
  8/18~19 사건의 선행수익이 아직 확정되지 않았다 — 확정되면 재측정한다.
  UP 문구의 "+5.1%/건"도 v28(2026-02~07) 값이고 현재 표본으로는 재현 불가다.

■ v66이 하는 일 / 하지 않는 일
  하는 일: 문구를 실제 동작에 맞춘다. 집행되지 않는 거부권을 '차단'이라 적지
    않고, 검증되지 않은 수익 수치를 근거로 인용하지 않는다. 죽은 컬럼은
    사실대로 표시하고(`REGIME_VETO_ENFORCED=0`) 재발 방지 테스트를 둔다.
  하지 않는 일: **자금 흐름을 바꾸지 않는다.** DOWN 하드블록을 되살리지도,
    배수를 켈리 수량에 연결하지도 않는다 — 둘 다 검정을 통과해야 하는 별건이고,
    현재 측정은 오히려 복원을 지지하지 않는다.
"""
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REGIME_UP = "UP"
REGIME_NEUTRAL = "NEUTRAL"
REGIME_DOWN = "DOWN"
REGIME_UNKNOWN = "UNKNOWN"

# 레짐별 신규 진입 사이즈 배수 (Kelly/추천 비중에 곱)
# [v32] DOWN 0.0 → 0.3: 알파 전면 게이트 도입으로 하락 레짐에서도
# 알파 최상위 10%만 축소 사이즈로 진입 허용(사용자 선택: 적응형·반감 사이즈).
# 실측 근거: risk_off=False & 내부약세(breadth<35) 알파 픽 +0.84%/승률 51%.
# 진짜 위험한 하락(risk_off=KOSPI<하락MA20)은 NEW_ENTRY_BLOCKED가 별도 하드블록.
#
# [v55 종결 — 평탄화 검토를 닫는다. 수익 근거는 없고, 낙폭 근거로 유지한다.]
#   v52가 '수익 근거 없음'(레짐 라벨 순열 p=0.872 · 플랫 대비 순기여 +0.0018%p
#   p=0.955)으로 판정했으나 켈리·v41 섹터팩터 상호작용 때문에 별건으로 미뤘다.
#   v55는 프로덕션 픽이 하루 1종목이라는 점을 이용해 배수 효과를 직접 계산했다
#   (포트 일수익 = 픽 실현수익 × 배수, 투입자본은 평균배수로 맞춤):
#     v54 현행 계약(15일): 차등 +2.041%/일 복리 +27.4% MDD -11.5%
#                          플랫 +3.387%/일 복리 +48.2% MDD -12.6%
#                          차등-플랫 -1.346%p (t=-1.00, p=0.336) · 순열 p=0.794
#     거부권 없음 계약(29일): 차등 +1.559% 복리 +51.6% MDD **-10.8%**
#                            플랫 +1.778% 복리 +59.4% MDD **-20.3%**
#                            차등-플랫 -0.219%p (t=-0.52, p=0.608) · 순열 p=0.694
#   → 수익은 플랫이 약간 높지만 유의하지 않고, 순열검정상 레짐 라벨은 무작위보다
#     나은 구분력이 없다. 게다가 레짐별 픽 수익 서열이 계약에 따라 완전히 뒤집힌다
#     (거부권 계약 DOWN +19.25% 최고 / 무거부권 계약 DOWN -1.26% 최악).
#     즉 '레짐이 수익을 예측한다'는 전제 자체가 이 표본에서 불안정하다.
#   유지 사유: 낙폭이 일관되게 낫다(-10.8% vs -20.3%). 배수는 예측 도구가 아니라
#     **위험 축소 도구**로만 정당화된다 — 수익 개선을 근거로 인용하면 안 된다.
#   재검증 조건: 레짐별 픽 20에피소드 이상.
REGIME_SIZE_MULT = {
    REGIME_UP: 1.0,
    REGIME_NEUTRAL: 0.5,
    REGIME_DOWN: 0.3,
    REGIME_UNKNOWN: 0.5,
}

# 판정 임계 (v28 검증값)
BREADTH_UP_MIN = 50.0
BREADTH_DOWN_MAX = 35.0

# ── [v66] 집행 범위를 상수로 못 박는다 ──────────────────────────────
# 레짐 거부권이 실제로 진입을 막는가. 현행 알파 경로에서는 **아니다**
# (v32가 recommendation_quality의 알파 분기에서 regime_ok를 제외했다).
# 되살리려면 이 값을 True로 바꾸는 것만으로는 안 되고 실제 배선이 필요하다
# — 그래서 test_v66_*가 이 상수와 배선의 일치를 검사한다.
VETO_ENFORCED = False

# REGIME_SIZE_MULT가 실제로 곱해지는 대상. 켈리 수량(사용자가 사는 수량)은
# 이 배수를 보지 않는다 — 표시용 비중에만 걸린다.
MULT_APPLIES_TO = "RECOMMENDED_WEIGHT_PCT"

_DOWN_TAIL_ENFORCED = "신규 진입 차단."
_DOWN_TAIL_NOT_ENFORCED = (
    "신규 진입은 **차단되지 않는다** — 알파 문턱이 90점으로 오를 뿐이다. "
    "표시 비중은 30%로 줄지만 추천 수량에는 반영되지 않는다. "
    "(하락 레짐 진입이 손실 우세라는 근거는 현재 표본에서 확인되지 않았다: "
    "DOWN 6일 게이트 상위5 +4.15%, 유니버스 대비 +2.29%p, p=0.37)"
)
_MULT_TAIL = (
    f"레짐 배수는 {MULT_APPLIES_TO}(표시용 비중)에만 적용되고 추천 수량은 "
    "바꾸지 않는다. 배수는 수익 근거가 아니라 낙폭 축소 근거로만 유지된다"
    "(v52 순열검정 p=0.872)."
)

_KOSPI_CSV = "kospi_daily.csv"


def load_kospi_daily(data_dir: str = "data") -> Optional[pd.DataFrame]:
    """kospi_daily.csv 로드 → date 오름차순 DataFrame (close, ma5, ma20)."""
    path = os.path.join(data_dir, _KOSPI_CSV)
    if not os.path.exists(path):
        logger.warning(f"KOSPI 일봉 파일 없음: {path}")
        return None
    try:
        k = pd.read_csv(path)
        if "date" not in k.columns or "close" not in k.columns:
            logger.warning("kospi_daily.csv 스키마 불일치 (date/close 필요)")
            return None
        k = k[["date", "close"]].copy()
        k["date"] = k["date"].astype(str).str.replace("-", "").str.slice(0, 8)
        k["close"] = pd.to_numeric(k["close"], errors="coerce")
        k = k.dropna().sort_values("date").reset_index(drop=True)
        k["ma5"] = k["close"].rolling(5).mean()
        k["ma20"] = k["close"].rolling(20).mean()
        return k
    except Exception as e:
        logger.warning(f"KOSPI 일봉 로드 실패: {e}")
        return None


def compute_market_regime(
    trade_ymd: str,
    market_breadth: Optional[float],
    data_dir: str = "data",
    kospi_df: Optional[pd.DataFrame] = None,
) -> dict:
    """레짐 판정.

    Returns dict:
      regime          : UP / NEUTRAL / DOWN / UNKNOWN
      size_mult       : 신규 진입 사이즈 배수 (0.0 / 0.5 / 1.0)
      allow_new_entry : bool — DOWN이면 False
      reason          : 사람이 읽는 판정 근거 (프론트 표시용)
      kospi_close / kospi_ma20 / kospi_ma5 / breadth : 원자료
    """
    ymd = str(trade_ymd).replace("-", "")[:8]
    breadth = None
    try:
        if market_breadth is not None:
            breadth = float(market_breadth)
    except (TypeError, ValueError):
        breadth = None

    k = kospi_df if kospi_df is not None else load_kospi_daily(data_dir)
    close = ma5 = ma20 = None
    if k is not None and not k.empty:
        sub = k[k["date"] <= ymd]
        if not sub.empty:
            last = sub.iloc[-1]
            close = float(last["close"])
            ma5 = float(last["ma5"]) if pd.notna(last["ma5"]) else None
            ma20 = float(last["ma20"]) if pd.notna(last["ma20"]) else None

    def _pack(regime: str, reason: str) -> dict:
        return {
            "regime": regime,
            "size_mult": REGIME_SIZE_MULT[regime],
            # [v66] 이 값은 **집행되지 않는다**. 알파 경로(현행)에서 v32가
            #   레짐 거부권을 뺐기 때문이다. 값을 True로 바꾸면 '차단 안 함'을
            #   말하게 되어 역시 부정확하므로, 판정 자체는 그대로 두고
            #   veto_enforced로 집행 여부를 분리해 함께 내보낸다.
            "allow_new_entry": regime != REGIME_DOWN,
            "veto_enforced": VETO_ENFORCED,
            "reason": reason,
            "kospi_close": close,
            "kospi_ma20": ma20,
            "kospi_ma5": ma5,
            "breadth": breadth,
        }

    # 데이터 부족 → UNKNOWN (보수적: 사이즈 절반)
    if breadth is None and (close is None or ma20 is None):
        return _pack(REGIME_UNKNOWN,
                     "지수/시장폭 데이터 부족 — 레짐 판정 불가. " + _MULT_TAIL)

    # 내부 붕괴 우선 — 지수가 어디 있든 시장폭 붕괴면 DOWN
    if breadth is not None and breadth < BREADTH_DOWN_MAX:
        return _pack(
            REGIME_DOWN,
            f"하락 레짐 — 시장폭 {breadth:.0f}% (<{BREADTH_DOWN_MAX:.0f}%), "
            "상승 종목이 소수인 내부 약세장. "
            + (_DOWN_TAIL_ENFORCED if VETO_ENFORCED else _DOWN_TAIL_NOT_ENFORCED),
        )

    up_trend = (
        close is not None and ma20 is not None and ma5 is not None
        and close > ma20 and close > ma5
    )
    if up_trend and breadth is not None and breadth >= BREADTH_UP_MIN:
        return _pack(
            REGIME_UP,
            f"상승 레짐 — KOSPI 20일선·5일선 위 + 시장폭 {breadth:.0f}%. "
            + _MULT_TAIL,
        )

    parts = []
    if close is not None and ma20 is not None:
        parts.append("KOSPI 20일선 " + ("위" if close > ma20 else "아래"))
    if breadth is not None:
        parts.append(f"시장폭 {breadth:.0f}%")
    return _pack(
        REGIME_NEUTRAL,
        "중립 레짐 — " + ", ".join(parts) + ". " + _MULT_TAIL,
    )


def inject_regime_columns(df: pd.DataFrame, regime_info: dict) -> pd.DataFrame:
    """recommend DataFrame에 레짐 컬럼 주입.

    [v66] REGIME_VETO_ENFORCED / REGIME_MULT_APPLIES_TO를 함께 낸다.
    앞의 둘(ALLOW_ENTRY · SIZE_MULT)만 있으면 화면·소비자가 그 값이 실제로
    집행된다고 읽는다 — 실제로는 둘 다 자금에 걸려 있지 않다. 집행 범위를
    데이터로 함께 내보내야 계약 검사가 가능하다.
    """
    df["MARKET_REGIME"] = regime_info.get("regime", REGIME_UNKNOWN)
    df["REGIME_REASON"] = regime_info.get("reason", "")
    df["REGIME_SIZE_MULT"] = regime_info.get("size_mult", 0.5)
    df["REGIME_ALLOW_ENTRY"] = int(bool(regime_info.get("allow_new_entry", True)))
    df["REGIME_VETO_ENFORCED"] = int(bool(regime_info.get(
        "veto_enforced", VETO_ENFORCED)))
    df["REGIME_MULT_APPLIES_TO"] = MULT_APPLIES_TO
    return df
