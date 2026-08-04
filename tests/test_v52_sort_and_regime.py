# -*- coding: utf-8 -*-
"""[v52] 후속 게이트 감사 — 정렬 1차키 교체 + UP 레짐 임계 70→90.

v51에서 리스크 게이트를 감사한 뒤 남은 미검증 단계 5개를 검정했다.
채택 2건, 유지 3건.

════ 채택 ①: 정렬 1차키 ACTION_PRIORITY → 검증된 알파 ════
ACTION_PRIORITY는 ROUTE의 결정적 재라벨이다
  (ATTACK1·ARMED2·WAIT3·NEUTRAL4·OVERHEAT5·EXIT_WARNING6·CARRY7,
   pipeline_calibrate.py의 _am 맵, 교차표 98.3~99% 대각).
ROUTE는 v32에서 이미 역예측 판정(ATTACK 알파 -2.9%p, t=-3.56, p=0.0004)을
받았으므로 재라벨이 원본보다 많은 정보를 가질 수 없고, 실측도 동일했다:
  · 일별 스피어만 IC: 풀 정의 16종 전부 IS 음수(설계방향) → OOS 양수(역방향).
    16/16 대 0/16, IS·OOS 부호일치 0/16.
  · 군평균 순서 vs 설계순서 rho=-0.107 (p=0.82) — 무상관.
    실제 순서 2,4,3,7,5,6,1 로 '최우선' prio=1(ATTACK)이 7군 중 최악
    (일평균 -2.61% vs 나머지 -1.27%, t=-3.48, p=0.001).
  · 알파백분위 통제 후 부분IC ≈ 원IC → 한계기여 0.
  · 물량 86%인 prio3 vs prio4는 32셀 중 30셀에서 prio4 우수(역방향)이나
    16풀 전부 p>0.05 — 방향조차 정하지 못한다.
prio=1이 최악인 원인은 기계적으로 특정된다: 중위 진입갭 9.8%(prio3 0.0%),
중위 RR 0.27(1.62), -8% 스톱 체결률 59.7%(50.1%).
→ '최우선' 라벨이 추격진입·손익비 붕괴 종목을 가리키고 있었다.
교체 근거는 '개선'이 아니라 '근거 없는 축을 SSOT 정렬 최상위에서 내린다'다.
알파 1차 정렬은 유니버스 9셀 전부에서 나쁘지 않았고(최대 +1.29%p) 풀에서는
유리했으나(+0.86~+0.98%p) 어느 셀도 유의하지 않다 — 수익 개선 주장 없음.

════ 채택 ②: UP 레짐 알파 임계 70 → 90 ════
v40이 '검증 범위 밖 — 기존 유지'로 남긴 값을 처음 검정했다
(레짐을 market_regime.py 정의 그대로 재구성해 UP 22일 확보,
 저장 MARKET_REGIME과 겹치는 6일에서 100% 일치 확인).
  · UP 내부 임계 스윕(60~95) 한계기여: 80셀 중 66셀 음수, 유의 0셀,
    t 중위 -0.41. 프로덕션 70의 한계기여 -0.147%p (t=-0.55, p=0.586, 21일).
  · UP 내부 알파 기울기(95−60분위) = -0.07%p (t=-0.10) — 판별력 자체가 없다.
    (대조: NEUTRAL +1.37%p, t=+2.77, 임계-수익 rho=+0.98 단조)
  · 최적 임계가 전·후반에서 60↔95 왕복(서열 rho -0.55), 부호도 반전
    (H1 -0.653%p / H2 +0.313%p) → 어떤 값도 노이즈 적합.
  · FWD3/5/10 전부 음수(-0.22/-0.15/-0.65) — 지평 견고성 없음.
  · 70이 추가로 들여보내는 밴드(UP 70~90): n=851 · 평균 -1.40% · 승률 31.6%
    vs 무여과 기저 -1.86% · 30.8% — 걸러내는 값이 사실상 없었다.
포트폴리오 수준에서도 평면90이 현행 차등을 16칸 중 15칸에서 앞섰다
(FWD5 OOS +0.239%p, t=+3.36, p=0.002).

════ 유지 (3건) ════
NEUTRAL 85 : v40 검증값. 90 상향은 OOS만 유의(+0.404%p t=+2.84)하고
             IS는 음수(-0.154) → 규율상 단독 채택 불가.
DOWN 90    : 80/85와 30셀 전부 구별 불가(유의 0셀)이나 95는 명확히 열화
             (-0.461%p, 유의성 소실) → 국소 최적.
BUY_NOW_PASS/GRADE : 두 렌즈 판정이 엇갈렸고(무신호 vs 표본부족) 풀 내
             통과율 91.1%로 8.9%만 자르므로 제거 이득·유지 비용 모두 낮다.
             BUY_NOW_PASS ≡ (BUY_NOW_GRADE=="BUY")로 동일 축이다.
REGIME_SIZE_MULT : 두 렌즈 모두 '수익 근거 없음'(순열검정 p=0.872, 투입
             동일 플랫 대비 순기여 +0.0018%p p=0.955)이나 사이징 변경은
             켈리·v41 섹터팩터와 상호작용하므로 별건 검증으로 분리.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alpha_engine as ae


# ── ① UP 레짐 임계 ──

def test_up_threshold_raised_to_90():
    assert ae.ALPHA_GATE_THRESHOLD["UP"] == 90.0, (
        "UP 70은 80셀 중 유의 0셀·전 보유기간 음수로 근거가 없다")


def test_other_regime_thresholds_unchanged():
    """NEUTRAL 85(v40 검증값)·DOWN 90(국소 최적)은 건드리지 않는다."""
    assert ae.ALPHA_GATE_THRESHOLD["NEUTRAL"] == 85.0
    assert ae.ALPHA_GATE_THRESHOLD["DOWN"] == 90.0
    assert ae.ALPHA_GATE_THRESHOLD["UNKNOWN"] == 85.0
    assert ae.ALPHA_GATE_DEFAULT_THRESHOLD == 85.0


def _row(**ov):
    r = {
        "종목코드": "000001", "종가": 10000.0, "손절가": 9200.0,
        "추천매도가1": 11500.0, "추천매수가": 10000.0,
        "거래대금(억원)": 200.0, "POC_GAP": 5.0,
        "ENTRY_RISK_GATE_OK": True, "ALPHA_VALIDATED": 1,
        "MARKET_REGIME": "UP", "BUY_NOW_PASS": 1,
        "Low_Trend_PCT": 5.0, "ALPHA_SCORE": 80.0,
    }
    r.update(ov)
    return r


def test_up_regime_gate_uses_90():
    """UP 레짐에서 알파 80점은 이제 탈락해야 한다(구 임계 70이면 통과)."""
    df = pd.DataFrame([_row(ALPHA_SCORE=80.0)])
    out = ae.apply_alpha_entry_gate(df)
    assert float(out["ALPHA_ENTRY_THRESHOLD"].iloc[0]) == 90.0
    assert int(out["TOP_PICK"].iloc[0]) == 0


def test_up_regime_high_alpha_still_passes():
    df = pd.DataFrame([_row(ALPHA_SCORE=95.0)])
    out = ae.apply_alpha_entry_gate(df)
    assert int(out["TOP_PICK"].iloc[0]) == 1


def test_neutral_regime_still_85():
    df = pd.DataFrame([_row(MARKET_REGIME="NEUTRAL", ALPHA_SCORE=87.0)])
    out = ae.apply_alpha_entry_gate(df)
    assert float(out["ALPHA_ENTRY_THRESHOLD"].iloc[0]) == 85.0
    assert int(out["TOP_PICK"].iloc[0]) == 1


# ── ② 정렬 1차키 ──

def _sortkeys(df_out, _sort_col="DISPLAY_SCORE"):
    """pipeline_calibrate.run_calibration의 키 선택 로직 발췌 재현."""
    import re
    src = (ROOT / "pipeline_calibrate.py").read_text(encoding="utf-8")
    assert "_alpha_col" in src, "알파 1차키 코드가 없다"
    _alpha_col = None
    if "ALPHA_SCORE" in df_out.columns:
        _asc = pd.to_numeric(df_out["ALPHA_SCORE"], errors="coerce")
        _val = pd.to_numeric(df_out.get("ALPHA_VALIDATED", 0), errors="coerce").fillna(0)
        if float(_val.max() or 0) == 1.0 and _asc.notna().sum() >= max(10, int(len(df_out) * 0.5)):
            df_out["ALPHA_SCORE"] = _asc.fillna(-1.0)
            _alpha_col = "ALPHA_SCORE"
    if _alpha_col:
        return [_alpha_col, _sort_col, "ACTION_PRIORITY"], [False, False, True]
    return ["ACTION_PRIORITY", _sort_col], [True, False]


def _frame(n=20):
    return pd.DataFrame({
        "ACTION_PRIORITY": [1] * 5 + [3] * (n - 5),
        "DISPLAY_SCORE": np.linspace(50, 90, n),
        "ALPHA_SCORE": [10.0] * 5 + list(np.linspace(60, 99, n - 5)),
    })


def test_alpha_becomes_primary_sort_key_when_validated():
    d = _frame(); d["ALPHA_VALIDATED"] = 1
    sk, sa = _sortkeys(d)
    assert sk[0] == "ALPHA_SCORE" and sa[0] is False
    assert "ACTION_PRIORITY" in sk, "prio는 2차키로 남아야 한다"
    s = d.sort_values(sk, ascending=sa)
    assert float(s["ALPHA_SCORE"].iloc[0]) == 99.0, "알파 최상위가 1위가 아니다"


def test_low_alpha_attack_no_longer_ranked_first():
    """prio=1(ATTACK)이지만 알파 최하위인 종목이 상단을 차지하지 못해야 한다.

    실측: prio=1 군은 7군 중 최악(-2.61% vs -1.27%, t=-3.48, p=0.001).
    """
    d = _frame(); d["ALPHA_VALIDATED"] = 1
    sk, sa = _sortkeys(d)
    s = d.sort_values(sk, ascending=sa).reset_index()
    attack_ranks = [s.index[s["index"] == i][0] + 1 for i in d.index[d["ACTION_PRIORITY"] == 1]]
    assert min(attack_ranks) > 10, f"저알파 ATTACK이 여전히 상단: {attack_ranks}"


def test_falls_back_to_legacy_when_alpha_unvalidated():
    d = _frame(); d["ALPHA_VALIDATED"] = 0
    sk, sa = _sortkeys(d)
    assert sk[0] == "ACTION_PRIORITY" and sa[0] is True


def test_falls_back_when_alpha_column_absent():
    d = _frame().drop(columns=["ALPHA_SCORE"]); d["ALPHA_VALIDATED"] = 1
    assert _sortkeys(d)[0][0] == "ACTION_PRIORITY"


def test_falls_back_when_alpha_coverage_too_low():
    d = _frame(); d["ALPHA_VALIDATED"] = 1
    d.loc[d.index[3:], "ALPHA_SCORE"] = np.nan       # 커버리지 15%
    assert _sortkeys(d)[0][0] == "ACTION_PRIORITY"


def test_action_priority_column_still_emitted():
    """정렬에서만 내린다 — 컬럼·표시 라벨은 유지."""
    src = (ROOT / "pipeline_calibrate.py").read_text(encoding="utf-8")
    assert 'df_out["ACTION_PRIORITY"] = df_out["ROUTE"].map(_am)' in src


# ── ③ 근거 기록 ──

def test_rationale_recorded_with_numbers():
    cal = (ROOT / "pipeline_calibrate.py").read_text(encoding="utf-8")
    assert "t=-3.48" in cal, "prio=1 역신호 통계량 누락"
    assert "16/16" in cal, "IC 부호반전 근거 누락"
    assert "수익 개선을 주장하지 않는다" in cal, "과대주장 방지 문구 누락"
    eng = (ROOT / "alpha_engine.py").read_text(encoding="utf-8")
    assert "t=-0.10" in eng, "UP 알파 기울기 근거 누락"
    assert "n=851" in eng, "UP 70~90 밴드 실측 누락"
    assert "60일" in eng, "UP 재검증 조건 누락"
