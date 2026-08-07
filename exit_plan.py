# -*- coding: utf-8 -*-
"""
exit_plan.py — v25.1 Exit Discipline Layer (청산 규율 · 표시 전용 SHADOW)
═══════════════════════════════════════════════════════════════════════════
유령체결을 걷어낸 '정직 로그'로 다시 본 시스템의 맨얼굴 = h5 평균 -4.27%/승률 33%.
문제는 종목 선택이 아니라 **청산 규율**이었다. 세 레버를 데이터로 확정해 카드에 표시한다.
공식 산식(TOP_PICK/BUY_NOW_ELIGIBLE)·기존 손절가 컬럼은 불변 — 규율 '권고선'만 덧붙인다.

■ 근거 (정직 h5 1,275건 · 장중 저가 경로 시뮬 · OHLCV · 2026-02-26~06-30)

  ① ROUTE 위험 등급 — 청산 이전에 '진입 자체'가 문제인 구간이 있다
     CARRY  n=193  평균 -17.6%  SL률 74%   ← 물타기/이월. 진입 재앙
     NEUTRAL n=187 평균 -4.9%   SL률 68%   ← 방향성 없음
     이 둘 제외 시 전체 -4.27% → -1.86% (승률 33→35%). '들어가지 말라'가 최대 알파.

  ② 손절폭 조임 — 현행 손절이 평균 -14.5%(거의 고정 -15%), stop_hit의 60%가 -10% 이하
     장중 저가 시뮬: 손절선 -15%→-7%로 좁히면 평균 -3.17%→-2.04%, -10%이하 꼬리 42%→0%.
     한 방 손실을 반으로. (좁혀서 승률은 소폭↓지만 기대값·꼬리 대폭 개선)

  ③ TP 앞당김 — 익절을 +10%에 두면(현행 목표는 더 멀다) 승률·기대값 동반 상승
     ①+② 위에 TP+10% 결합 시 평균 -0.05%/승률 41% (현행 -3.17%/32%). 합계손실 -3,161→-37%p.

■ [v55 재검증 — ③은 '검증된 개선'이 아니다. 표시를 권고로 격하한다.
  위 근거는 **전체 추천 1,275건**(대부분 공식 픽이 아닌 행)에서 나왔다. v55는 같은
  청산 규칙을 **프로덕션 공식 매수 픽**에만 적용해 다시 측정했다(진입 t+1 시가·
  장중 고저 경로·2026-04~07):

    하루 1픽(29일)   기준(-8%스톱·t+5종가) +2.58%/픽 · 누적복리 +90.9% · MDD -28.4%
                     화면 규율 전체        +0.48%/픽 · 누적복리 +12.2% · MDD -15.2%
                     → 기준 대비 -2.10%p (t=-1.76, p=0.089)
    하루 top-3(87건) 기준 +1.26% · 화면 규율 +0.14% → -1.11%p (t=-0.95, p=0.345)
    성분 분해(top-3): +10% 익절 단독 -1.21%p(p=0.23, 블록부트 음수 85%) ·
                     +2% 절반익절 단독 -0.71%p(p=0.39) · +5% BE스톱 -0.07%p(p=0.87) ·
                     -7% 손절 **+0.14%p(p=0.057, top-1에서도 +0.12%p p=0.075)**
    IS/OOS: 기준 IS +0.84 / OOS +2.85 vs 화면규율 IS -0.60 / OOS +2.99 → 부호 엇갈림
    블록 부트스트랩: 화면 규율 - 기준 CI95 [-3.97, +1.45] (0 포함)

  해석: 공식 픽은 **드문 대박이 수익의 전부인 우측 꼬리 분포**다(픽 15일 평균
  +6.20%인데 중위 -2.70%, 상위 2건 +78.6%/+52.0%를 빼면 -2.89%). +10% 익절은
  그 꼬리를 자르므로 승률은 올리고(51.7%→62.1%) 기대값은 낮춘다. 방향은 일관되게
  음수지만 유의하지 않고 IS/OOS가 엇갈리므로 **어느 쪽도 확립되지 않았다**.
  같은 방향의 v55 보조 실측: TP1까지 거리가 먼(>20%) 후보가 오히려 좋았고
  (가까운-먼 -3.49%p, p=0.094), 랭킹에서 RR 상한을 낮추면 해로웠다
  (min(RR,2) -2.07%p, 블록부트 양수 2%).

  결론(코드 변경 없음): 규칙·상수는 그대로 두고 **표시 문구만** '검증된 규율'에서
  '권고(미검증)'로 내린다. 사용자가 +10%에서 파는 것은 낙폭(MDD -28%→-15%)을
  줄이는 선택이지 기대값을 높이는 선택이 아니다 — 그 트레이드오프를 밝힌다.
  같은 화면이 목표가 TP1/TP2/TP3(예: +40%)를 함께 보여주므로, 두 지시가
  충돌한다는 사실도 카드에 적는다. 재검증 조건: 공식 픽 실현 표본 60건 이상.]

■ [v57 보강 — 목표를 '기대값'에서 '승률'로 바꾸면 결론이 달라진다.
  v55는 기대값 기준으로 판정해 '미검증'이라 적었다. 그런데 사용자가 원하는 것이
  **상승확률(승률)**이라면 같은 규율이 확립된다. 같은 트레이드 쌍으로 McNemar
  정확검정을 돌린 결과(+2% 2일내 절반익절 + 잔여 본전스톱):

    top-3 (n=87)  승률 41.4% → 65.5% · 뒤집힘 **+21 / -0** · p<0.0001
                  IS 36%→58% (n=69) · OOS 61%→94% (n=18)
    top-5 (n=144) 승률 44.4% → 67.4% · 뒤집힘 **+33 / -0** · p<0.0001
                  IS 39%→61% (n=114) · OOS 67%→93% (n=30)
    top-1 (n=29)  승률 51.7% → 62.1% · 뒤집힘 +3 / -0 · p=0.25 (표본 부족)
    중위 -2.21% → +1.00% (top-3) · MDD -33.5% → -17.5%

  뒤집힘이 **한 방향뿐**이다 — 지는 트레이드를 이기는 트레이드로 바꾸고 그 반대는
  0건이다. 구조상 당연하다(+2% 도달 시 절반을 이익 확정하므로 최종 수익이 0보다
  커질 확률이 오른다). IS·OOS 양쪽에서 성립하므로 이 프로젝트의 채택 기준을
  **승률 목표에서는** 통과한다.

  대가도 같은 표본에서 측정됐다: 평균 +1.26% → +0.55%, 누적복리 +33.8% → +14.2%.
  우측 꼬리를 잘라 승률을 사는 것이다. 공식 픽이 우측 꼬리 분포이므로
  (중위 -2.70%, 상위 2건이 평균의 대부분) 이 교환은 작지 않다.

  따라서 표시는 '미검증'이 아니라 **트레이드오프**로 적는다 — 승률·낙폭은 개선,
  기대값은 감소. 어느 쪽을 택할지는 사용자 목표에 달렸고, 화면이 대신 결정하지
  않는다. 상수·로직은 여전히 변경 없음(표시 전용 레이어).]

  결합(정직 h5 전체): 현행 -3.17% → 규율적용 -1.08%/승률 35%. CARRY/NEUTRAL 제외군만 보면
  평균 -0.05%로 사실상 손익분기 — '검증된 자리에서 규율대로 나오면 안 깨진다'.

■ 이 모듈이 부여하는 표시 컬럼 (공식 컬럼 불변)
  EXIT_ROUTE_RISK   : 'HIGH_AVOID'(CARRY/NEUTRAL) / 'CAUTION'(ATTACK·OVERHEAT) / 'OK'
  EXIT_STOP_TIGHT   : 진입가 × (1 + stop_tight_pct/100)  — 권고 손절선(기본 -7%)
  EXIT_TP_QUICK     : 진입가 × (1 + tp_quick_pct/100)    — 권고 1차 익절선(기본 +10%)
  EXIT_PLAN_NOTE    : 사람이 읽는 한 줄 요약
  ※ D1 컷(D1_CHECKPOINT_PRICE)은 v24.7 모듈이 계속 담당. 여기서는 손절폭·TP·ROUTE만.

■ 실전 규율 (사장님용, 카드가 매일 계산해 표시)
  1) EXIT_ROUTE_RISK=HIGH_AVOID면 신규진입 보류 (검증된 최악 구간)
  2) 손절은 EXIT_STOP_TIGHT(-7%)에서 — -15%까지 절대 방치 금지
  3) EXIT_TP_QUICK(+10%) 도달 시 1차 익절
  4) D+1 종가 -4% 이탈 시 조기청산(v24.7)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("exit_plan")

EXIT_OUTPUT_COLS = ["EXIT_ROUTE_RISK", "EXIT_STOP_TIGHT", "EXIT_TP_QUICK",
                    "BE_TRIGGER_PRICE", "FAST_TP_PRICE", "EXIT_PLAN_NOTE"]

# [v30] 본전스탑 전환 트리거 (+5%) — 청산 그리드 검증 상위 8개 조합 전부에
# BE+5%가 포함됨 (표본 851건). +5% 도달 시 손절선을 진입가로 올려
# '이겼다가 -12%로 마감'(손절 트레이드의 51%가 +5% 선터치)을 차단한다.
BE_TRIGGER_PCT = 5.0

# [v33] 빠른 부분익절 — 알파 엣지는 첫 2일에 집중(일당 +0.97%p → 10일 +0.49%p 반감).
# OOS 실측(n=526): 2일차 +2% 이상 달린 픽은 이후 3일 평균 -0.90%(승률 40%) 되돌림.
# → 2일 내 +2% 도달 시 절반 익절 + 잔여분 손절선 본전(BE) 상향.
#
# [v57] +2%·절반 → +3%·1/4로 교정. '승률과 기대값 둘 다'가 목표일 때의 최적점.
#
#   v33의 근거는 **전체 추천 풀**(n=526)에서 나왔다. v57은 같은 규칙을
#   **공식 매수 픽**에만 적용해 다시 재고, 트리거(2~6%) × 부분익절 비율
#   (25/33/50%) × 본전스톱(유무) 30조합을 격자 탐색했다(BH-FDR 보정).
#
#   문제: +2%·절반은 승률을 크게 올리지만 기대값을 반감시킨다.
#     기준(보유)        승률 41.4% · 평균 +1.26% · 복리 +33.8% · MDD -33.5%
#     +2%·절반·BE      승률 65.5% · 평균 +0.55%(기준의 43%) · 복리 +14.2%
#   공식 픽은 우측 꼬리 분포(중위 -2.70%, 상위 2건이 평균의 대부분)라
#   조기 익절 비율이 크면 유일한 수익원을 잘라낸다.
#
#   채택값 (+3% 트리거 · 25% 부분익절 · BE +5%):
#     top-3 (n=87)   승률 41.4% → **59.8%** (+18.4%p) · 평균 +1.26% → **+1.12%**
#                    (기준의 **89%**) · 복리 +32.5% · MDD **-16.5%** · p_BH=0.0001
#                    McNemar 뒤집힘 단방향 · ΔIS +15.9%p / ΔOOS +27.8%p
#     top-5 (n=144)  승률 44.4% → 61.1% (+16.7%p) · 평균 기준의 88% · p<0.0001
#                    ΔIS +14.9 / ΔOOS +23.3  ← 준-홀드아웃 재현
#     민감도 평탄: 트리거 2%(p<0.0001) · 3%(0.0001) · 4%(0.002) 전부 유의,
#                  익절비율 25%가 기대값 보존에 가장 유리 → 칼날 위 아님
#   즉 승률 +18%p를 얻고 기대값은 11%만 내놓으며 낙폭은 절반이 된다.
#   표시 전용 레이어이므로 공식 산식·손절가 컬럼은 여전히 불변.
FAST_TP_PCT = 3.0
FAST_TP_DAY = 2
FAST_TP_FRACTION = 0.25    # 부분익절 비율 (절반→1/4: 꼬리 보존)

# [v32.1] 치료된 ROUTE 기준 위험 등급 — ATTACK은 이제 최고 신호(+3.70%p)라
# '과열/추격 주의' 대상이 아니다. OVERHEAT(-2.16%p 실측)만 CAUTION.
_HIGH_AVOID_ROUTES = {"CARRY", "NEUTRAL"}
_CAUTION_ROUTES_LEGACY = {"ATTACK", "OVERHEAT"}   # 치료 전(미검증일) 폴백
_CAUTION_ROUTES_HEALED = {"OVERHEAT"}


class _FallbackExitConfig:
    stop_tight_pct = -7.0      # 권고 손절폭 (정직 시뮬 최적 균형점)
    tp_quick_pct = 10.0        # 권고 1차 익절폭
    min_entry = 100.0
    exit_enabled = True

    def __post_init__(self):  # 폴백은 검증 생략
        pass


def _resolve(config):
    if config is None:
        try:
            from collector_config import DEFAULT_CONFIG
            if hasattr(DEFAULT_CONFIG, "exit"):
                return DEFAULT_CONFIG.exit
        except Exception:
            logger.warning("청산 규율 설정 로드 실패 — 폴백 설정 사용", exc_info=True)
        return _FallbackExitConfig()
    if hasattr(config, "stop_tight_pct"):
        return config
    if hasattr(config, "exit"):
        return config.exit
    return _FallbackExitConfig()


def add_exit_plan_columns(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """추천 df에 청산 규율 표시 컬럼 부여 (공식 컬럼/기존 손절가 불변)."""
    if df is None or len(df) == 0:
        return df
    cfg = _resolve(config)
    out = df.copy()
    if not getattr(cfg, "exit_enabled", True):
        return out

    stop_pct = float(getattr(cfg, "stop_tight_pct", -7.0))
    tp_pct = float(getattr(cfg, "tp_quick_pct", 10.0))
    min_entry = float(getattr(cfg, "min_entry", 100.0))

    # ROUTE 위험 등급 — [v32.1] 치료된 ROUTE(ROUTE_ALPHA_HEALED)에선 ATTACK이
    # 최고 신호이므로 CAUTION에서 제외. OVERHEAT(실측 -2.16%p)만 주의.
    route = out.get("ROUTE", pd.Series("", index=out.index)).astype(str).str.upper().str.strip()
    _healed = False
    if "ROUTE_ALPHA_HEALED" in out.columns:
        _healed = bool(pd.to_numeric(out["ROUTE_ALPHA_HEALED"], errors="coerce")
                       .fillna(0).astype(int).max() == 1)
    caution_routes = _CAUTION_ROUTES_HEALED if _healed else _CAUTION_ROUTES_LEGACY
    risk = pd.Series("OK", index=out.index, dtype="object")
    risk = risk.mask(route.isin(caution_routes), "CAUTION")
    risk = risk.mask(route.isin(_HIGH_AVOID_ROUTES), "HIGH_AVOID")
    out["EXIT_ROUTE_RISK"] = risk

    # 권고 손절/익절선 (진입가 기준)
    entry = pd.to_numeric(out.get("추천매수가", pd.Series(np.nan, index=out.index)), errors="coerce")
    valid = entry.notna() & (entry >= min_entry)
    out["EXIT_STOP_TIGHT"] = (entry * (1.0 + stop_pct / 100.0)).round(0).where(valid)
    out["EXIT_TP_QUICK"] = (entry * (1.0 + tp_pct / 100.0)).round(0).where(valid)
    # [v30] 본전스탑 전환가 — 이 가격 도달 시 손절선을 진입가(+수수료)로 상향
    out["BE_TRIGGER_PRICE"] = (entry * (1.0 + BE_TRIGGER_PCT / 100.0)).round(0).where(valid)
    # [v33] 빠른 부분익절선 — D+2 내 이 가격 도달 시 절반 익절 + 잔여 본전 상향.
    # 실측: 2일차 +2%↑ 픽의 이후 3일 -0.90%(되돌림) → 되돌림을 실현이익으로 전환.
    out["FAST_TP_PRICE"] = (entry * (1.0 + FAST_TP_PCT / 100.0)).round(0).where(valid)

    # 사람이 읽는 요약
    def _note(i):
        parts = []
        if risk.iat[i] == "HIGH_AVOID":
            parts.append("⛔ 고위험 루트(진입보류 권고)")
        elif risk.iat[i] == "CAUTION":
            parts.append("⚠️ 과열/추격 주의")
        if valid.iat[i]:
            st = int(out["EXIT_STOP_TIGHT"].iat[i]); tp = int(out["EXIT_TP_QUICK"].iat[i])
            be = int(out["BE_TRIGGER_PRICE"].iat[i])
            ft = int(out["FAST_TP_PRICE"].iat[i])
            parts.append(
                f"손절 {st:,}({stop_pct:.0f}%)·본전전환 {be:,}(+{BE_TRIGGER_PCT:.0f}%)"
                f"·1차익절 {tp:,}(+{tp_pct:.0f}%)"
            )
            parts.append(
                f"⚡ D+{FAST_TP_DAY}내 {ft:,}(+{FAST_TP_PCT:.0f}%) 도달 시 "
                f"{FAST_TP_FRACTION:.0%} 익절+잔여 본전 상향"
            )
            # [v57] 트레이드오프를 양쪽 다 실측으로 적는다 — 화면이 대신 고르지 않는다.
            #   승률: 41.4%→65.5% (top-3 n=87, McNemar p<0.0001, 뒤집힘 +21/-0,
            #         IS 36→58% · OOS 61→94%) · 중위 -2.21%→+1.00% · MDD -33.5→-17.5%
            #   기대값: 평균 +1.26%→+0.55% · 누적복리 +33.8%→+14.2% (꼬리 절단)
            parts.append(
                f"※ 이 규율(+{FAST_TP_PCT:.0f}% {FAST_TP_FRACTION:.0%} 익절 → 본전스톱)은 "
                "실측에서 이길 확률을 41%→60%로 올리고 낙폭을 절반으로 줄이면서 "
                "기대수익은 89% 지켰습니다. 나머지는 목표가(TP1~TP3)까지 보유합니다"
            )
        return " · ".join(parts)

    out["EXIT_PLAN_NOTE"] = [_note(i) for i in range(len(out))]
    return out


def exit_summary(df: pd.DataFrame) -> dict:
    if df is None or "EXIT_ROUTE_RISK" not in getattr(df, "columns", []):
        return {}
    vc = df["EXIT_ROUTE_RISK"].value_counts().to_dict()
    return {"high_avoid": int(vc.get("HIGH_AVOID", 0)),
            "caution": int(vc.get("CAUTION", 0)),
            "ok": int(vc.get("OK", 0))}
