# -*- coding: utf-8 -*-
"""[v55] '왜 추천이 없냐'의 진짜 원인 — risk_off 차단 상태를 숫자로.

■ v55에서 밝혀진 사실
  사용자는 수 주간 "왜 추천이 한 번도 없냐"고 물었고 v45~v54는 라벨·UI·사유
  배선을 고쳤다. 그런데 프로덕션 CSV를 직접 파보니 원인은 표시가 아니었다:

    최근 16영업일 중 15일이 NEW_ENTRY_BLOCKED = 전 종목
    사유 "risk_off: 추천손절 유지 + 신규진입 차단"
    코스피 20일선 대비 이탈: 7/29 **-20.1%** … 8/4 -5.5% … 8/5 **-1.1%(해제)**
    8/5에 풀리자 그날 처음 TOP_PICK 20개가 나왔다 — 사용자가 본 첫 추천종목

  즉 제품이 고장난 게 아니라 폭락 방어가 설계대로 작동했다. 문제는 화면이
  '얼마나 깊은 차단이고 언제 풀리는지'를 말하지 않아 '추천 없음'과 구별되지
  않았던 것이다. (v34가 스트립을 넣었지만 df 컬럼 의존이고 수치가 없었다.)

■ 차단이 옳은지는 단정하지 않는다 (실측 기록)
  패널 내 차단일 6일(전부 OOS)의 반사실 픽은 오히려 +3.15%·승률 66.7%로
  허용일 +2.43%보다 나빴다는 증거가 없다
  (Welch -0.72%p t=-0.20 p=0.85 · Mann-Whitney p=0.75 · 순열 p=0.60 ·
   블록부트 CI95 [-6.49,+6.59] · FWD3 +0.71%p p=0.83).
  깊은 폭락 구간(-8~-20%, 7/28~8/4)은 아직 선행수익이 없어 평가 불가이고,
  v49 기록대로 이 축은 창 길이에 따라 부호가 뒤집힌 이력이 있다
  (105일 창 +3.07% t=+4.05 → 397일 창 -1.37%).
  → 규칙은 손대지 않고 상태만 정직하게 노출한다. '검증됨' 주장은 철회한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import momentum_lane as ml  # noqa: E402
from services import entry_block_status as ebs  # noqa: E402


# ── ① 엔진과 화면이 같은 상수를 쓰는가 ──

def test_constants_match_production_engine():
    """다르면 화면이 '차단 아님'인데 엔진은 차단하는 상태가 생긴다."""
    assert ebs.MA_WINDOW == ml.REGIME_MA_WINDOW
    assert ebs.MA_SLOPE_LOOKBACK == ml.REGIME_MA_SLOPE_LOOKBACK
    assert ebs.DEVIATION_FLOOR == ml.REGIME_DEVIATION_FLOOR


def _kospi(tmp_path, closes):
    dates = pd.bdate_range("2026-01-01", periods=len(closes))
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": closes}).to_csv(
        tmp_path / "kospi_daily.csv", index=False, encoding="utf-8-sig")
    return str(tmp_path)


# ── ② 판정이 프로덕션 함수와 일치하는가 ──

def test_agrees_with_production_on_crash(tmp_path):
    """하락 전환 + MA20 아래 -3% 이상 이탈 → 양쪽 모두 차단."""
    closes = list(np.linspace(3000, 3300, 40)) + list(np.linspace(3300, 2600, 25))
    d = _kospi(tmp_path, closes)
    st = ebs.compute_entry_block_status(d)
    prod, _info = ml.compute_market_risk_off(str(Path(d) / "kospi_daily.csv"))
    assert st["ok"] and st["risk_off"] is True
    assert st["risk_off"] == prod


def test_agrees_with_production_on_uptrend(tmp_path):
    closes = list(np.linspace(2600, 3400, 60))
    d = _kospi(tmp_path, closes)
    st = ebs.compute_entry_block_status(d)
    prod, _ = ml.compute_market_risk_off(str(Path(d) / "kospi_daily.csv"))
    assert st["risk_off"] is False
    assert st["risk_off"] == prod


# ── ③ 숫자가 실제로 담기는가 ──

def test_reports_actionable_numbers_when_blocked(tmp_path):
    closes = list(np.linspace(3000, 3300, 40)) + list(np.linspace(3300, 2600, 25))
    st = ebs.compute_entry_block_status(_kospi(tmp_path, closes))
    assert st["deviation_pct"] < ebs.DEVIATION_FLOOR * 100
    assert st["unlock_gap_pct"] > 0, "해제까지 남은 거리를 말해야 한다"
    assert st["streak_days"] >= 1
    assert st["block_rate_60d_pct"] is not None
    assert st["ma20_falling"] is True
    line = ebs.block_reason_line(st)
    for token in ["신규진입 차단", "20일선", "해제선"]:
        assert token in line, f"{token} 누락: {line}"


def test_unlock_gap_is_zero_when_not_blocked(tmp_path):
    st = ebs.compute_entry_block_status(_kospi(tmp_path, list(np.linspace(2600, 3400, 60))))
    assert st["unlock_gap_pct"] == 0.0
    assert "차단 없음" in ebs.block_reason_line(st)


def test_missing_data_does_not_raise(tmp_path):
    st = ebs.compute_entry_block_status(str(tmp_path))
    assert st["ok"] is False and st["risk_off"] is None
    assert ebs.block_reason_line(st) == ""


def test_short_history_is_treated_as_no_block(tmp_path):
    st = ebs.compute_entry_block_status(_kospi(tmp_path, [3000, 3010, 3020]))
    assert st["ok"] is False


# ── ④ 결정 센터가 그 숫자를 사용하는가 ──

def test_decision_center_uses_market_truth():
    src = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")
    assert "compute_entry_block_status" in src, "코스피 원본 기반 상태 미배선"
    assert "unlock_gap_pct" in src, "해제까지 남은 거리 미표시"
    assert "streak_days" in src


def _code_only(path: Path) -> str:
    """주석 줄을 걷어낸 소스 — 화면에 나가는 문구만 검사하기 위해."""
    return "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                     if not ln.lstrip().startswith("#"))


def test_user_facing_text_retracts_the_unverified_claim():
    """'승률 17%' 단정은 v55가 확인하지 못했다 — 화면 문구에서 빠져야 한다.

    주석에는 철회 근거로 남는다(재도입 방지). 검사 대상은 렌더 문구뿐이다.
    """
    for rel in [("components", "decision_center.py"), ("components", "tab_stocks.py")]:
        code = _code_only(ROOT.joinpath(*rel))
        assert "승률 17%" not in code, f"{rel[-1]}에 미검증 단정이 남아 있다"
    src = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")
    assert "통계로 확립되지 않았습니다" in src, "미검증임을 밝히는 문구 필요"
    assert "수익 효과는 미검증" in src


def test_batch_logs_block_and_pick_counts():
    """#29 — 차단 상태와 픽 수량을 배치 로그 한 줄로 남긴다."""
    src = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")
    assert "진입차단 계측" in src
    for token in ["NEW_ENTRY_BLOCKED", "TOP_PICK", "PRODUCTION_BUY", "risk_off"]:
        assert token in src


# ── ⑤ 청산 규율 문구 정직화 ──

def test_exit_plan_states_the_measured_tradeoff():
    """[v57] '미검증'이 아니라 트레이드오프로 적는다 — 양쪽 다 실측이다.

    v55는 기대값 기준으로 '미검증'이라 적었다. v57에서 목표를 승률로 바꿔
    같은 트레이드 쌍 McNemar 검정을 하니 확립됐다:
      top-3 n=87  41.4%→65.5% 뒤집힘 +21/-0 p<0.0001 (IS 36→58 · OOS 61→94)
      top-5 n=144 44.4%→67.4% 뒤집힘 +33/-0 p<0.0001
    대가도 같은 표본에서 측정됐다: 평균 +1.26%→+0.55%, 복리 +33.8%→+14.2%.
    화면은 어느 쪽도 대신 고르지 않고 둘 다 말한다.
    """
    src = (ROOT / "exit_plan.py").read_text(encoding="utf-8")
    assert "t=-1.76" in src, "v55 기대값 실측 근거 미기록"
    assert "McNemar" in src, "v57 승률 검정 근거 미기록"
    assert "p<0.0001" in src
    # [v57] 문구가 '충돌 경고'에서 '트레이드오프 수치 제시'로 바뀌었다 —
    #       두 선택지를 함께 언급하는지만 검사한다(구체 문구는 v57 테스트가 고정).
    assert "목표가(TP1~TP3)까지 보유" in src
    # 화면 문구: 승률 개선과 기대값 보존을 수치로 함께 말해야 한다
    # [v57] 파라미터 재조정(+3%·1/4)으로 '기대값 감소' 서술이 '89% 보존'으로 바뀌었다.
    assert "이길 확률을 41%→60%로 올리고" in src
    assert "기대수익은 89% 지켰습니다" in src
    assert "익절 권고는 미검증" not in src, "승률 근거가 확립된 뒤에도 미검증이라 적고 있다"


def test_rr_warning_no_longer_reads_as_bad_stock():
    """v55 실측은 먼 목표가 후보가 오히려 나았다고 말한다."""
    src = (ROOT / "components" / "stock_detail_v2.py").read_text(encoding="utf-8")
    assert "목표가가 먼 후보가 오히려 나았습니다" in src
    assert "p=0.094" in src or "+3.5%p" in src


# ── ⑥ 종결 기록 ──

def test_regime_size_mult_decision_recorded():
    src = (ROOT / "market_regime.py").read_text(encoding="utf-8")
    assert "v55 종결" in src
    assert "위험 축소 도구" in src, "수익 근거로 인용 금지 기록 필요"
    assert "순열 p=0.694" in src or "p=0.694" in src


def test_route_veto_v55_measurement_recorded():
    src = (ROOT / "services" / "recommendation_quality.py").read_text(encoding="utf-8")
    assert "v55 재측정" in src
    assert "재검증 조건" in src
    assert "20에피소드" in src
