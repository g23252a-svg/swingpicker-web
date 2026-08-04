# -*- coding: utf-8 -*-
"""[v53] UI 배선 완결 — 엔진 변경(v44~v52)이 화면에 반영 안 된 것 수정.

엔진 패치는 병합됐는데 화면 배선이 빠져 사용자가 볼 수 없던 것들:

■ v45 '침묵의 0주 금지'가 미완이었다
  v45는 KELLY_ZERO_REASON·KELLY_NEED_WIN_RATE를 만들어 0주 사유를 분류했다
  (실측 승률 vs 필요 승률 / 배분액<1주 가격 / 상태 미달 / 가격정보 불완전).
  그런데 UI 배선이 없어 화면은 여전히 '신규 부적합(0주)'로만 끝났다.
  사용자가 "왜 추천이 없냐"고 반복해 물은 지점이 정확히 이것이다.
  → 액션레일에 사유 블록 추가, 목록 사유에도 승격.

■ v47 위험구간이 상세 페이지에만 있었다
  목록(특히 보유관리 섹션)에 없어 보유자가 스캔으로 정리 신호를 놓친다.
  실측: 알파 하위10% + 저점추세 하위30% → 5일 -2.88% · 승률 21.5%,
        20일 승률 10.8% (엣지 -1.59%p, t=-4.43, IS·OOS 양쪽 유의).
  → 목록 payload·사유 최상위로 노출.

■ v44 낙폭추격 경고 미노출
  이격 -15% 이하 구간 실측 손실(10일 -3.6% · 승률 41%)을 화면에서 못 봤다.

■ 목록 사유가 게이트 실제 임계와 어긋났다
  RR 임계 1.2로 표시했는데 게이트 실제값은 1.0 → 통과 종목을 '부족'이라
  적는 거짓 사유. POC 임계 30인데 게이트는 20 (v27 검증: 20% 초과 구간
  승률 20%/12%) → 실제로 차단되는 21~30% 구간을 '양호'로 표시.

■ ACTION_PRIORITY를 '우선순위'로 강조 표시했다
  v52 실측에서 이 축은 ROUTE의 결정적 재라벨이고 예측력이 없다
  (IC 부호 IS/OOS 완전 반전 16/16 대 0/16, 군평균 vs 설계순서 rho -0.107).
  특히 '최우선' prio=1(ATTACK)이 7군 중 최악이다
  (-2.61%/일 vs 나머지 -1.27%, t=-3.48, p=0.001 — 중위 진입갭 9.8%,
   중위 RR 0.27, 스톱 체결률 59.7%). 정렬키에서 내렸으니 표시도 정직해야 한다.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import components.tab_stocks as ts


def _item(**ov):
    """게이트를 모두 통과한 기본 후보."""
    d = {"rr": 2.0, "poc_gap": 0.0, "vwap_gap": 0.0,
         "top_pick": True, "eligible": True, "qty": 5,
         "danger_zone": False, "kelly_zero_reason": ""}
    d.update(ov)
    return d


# ── ① 게이트 실제 임계와 사유 문구 정합 ──

def test_rr_threshold_matches_gate_1_0_not_1_2():
    """게이트는 RR>=1.0이다. 1.2로 표시하면 통과 종목에 거짓 사유가 붙는다."""
    assert "부족" in ts._candidate_watch_reason(_item(rr=0.9, top_pick=False))
    r = ts._candidate_watch_reason(_item(rr=1.1, top_pick=False))
    assert "손익비 부족" not in r, f"게이트 통과 RR을 부족이라 표기: {r}"


def test_poc_threshold_matches_gate_20_not_30():
    """게이트는 POC_GAP<=20 (v27: 20% 초과 구간 승률 20%/12%)."""
    r25 = ts._candidate_watch_reason(_item(poc_gap=25.0, top_pick=False))
    assert "POC 과열" in r25, f"실제로 차단되는 25%를 양호로 표기: {r25}"
    r15 = ts._candidate_watch_reason(_item(poc_gap=15.0, top_pick=False))
    assert "POC 과열" not in r15


# ── ② v47 위험구간이 목록에 노출되는가 ──

def test_danger_zone_is_top_priority_reason():
    r = ts._candidate_watch_reason(_item(danger_zone=True, top_pick=False,
                                         rr=0.5, poc_gap=99.0))
    assert r.startswith("⛔ 위험구간"), f"위험구간이 최우선이 아니다: {r}"


def test_danger_zone_absent_when_flag_off():
    assert "위험구간" not in ts._candidate_watch_reason(_item(top_pick=False))


def test_row_payload_carries_danger_zone():
    row = pd.Series({"종목명": "테스트", "종목코드": "000001",
                     "DANGER_ZONE": 1, "켈리_수량": 0})
    p = ts._candidate_row_payload(row)
    assert p["danger_zone"] is True or p["danger_zone"] == 1


# ── ③ v45 0주 사유가 목록에 노출되는가 ──

def test_zero_qty_reason_surfaces_in_list():
    """게이트를 통과했는데 0주면 그 사유가 표시돼야 한다."""
    r = ts._candidate_watch_reason(
        _item(qty=0, kelly_zero_reason="실측 승률 41% < 필요 승률 36%"))
    assert "실측 승률 41%" in r


def test_zero_qty_has_fallback_reason_without_column():
    r = ts._candidate_watch_reason(_item(qty=0, kelly_zero_reason=""))
    assert "켈리 0주" in r, f"사유 없는 0주: {r}"


def test_positive_qty_has_no_zero_reason():
    r = ts._candidate_watch_reason(_item(qty=7))
    assert "0주" not in r


def test_row_payload_carries_kelly_zero_reason():
    row = pd.Series({"종목명": "테스트", "종목코드": "000001",
                     "켈리_수량": 0, "KELLY_ZERO_REASON": "상태 WAIT — 신규진입 상태 아님"})
    p = ts._candidate_row_payload(row)
    assert p["kelly_zero_reason"] == "상태 WAIT — 신규진입 상태 아님"
    assert p["qty"] == 0


# ── ④ 상세 dict 배선 ──

def test_detail_dict_exposes_new_fields():
    src = (ROOT / "components" / "stock_detail_v2.py").read_text(encoding="utf-8")
    for key in ['"kelly_zero_reason"', '"kelly_need_win_rate"', '"crash_chase_warn"']:
        assert key in src, f"상세 dict에 {key} 미배선"


def test_action_priority_no_longer_labeled_as_priority():
    """v52에서 무예측 판정된 축을 '우선순위'로 강조하면 오해를 부른다."""
    src = (ROOT / "components" / "stock_detail_v2.py").read_text(encoding="utf-8")
    assert "ACTION_PRIORITY <strong>" not in src, "여전히 우선순위로 강조 표시"
    assert "priority_html" in src
    assert "t=-3.48" in src, "prio=1 역신호 근거 미기록"


def test_crash_chase_warning_rendered():
    src = (ROOT / "components" / "stock_detail_v2.py").read_text(encoding="utf-8")
    assert "chase_html" in src
    assert "낙폭 과대" in src


# ── ⑤ 근거 기록 ──

def test_stale_thresholds_documented():
    src = (ROOT / "components" / "tab_stocks.py").read_text(encoding="utf-8")
    assert "게이트 실제값은" in src, "임계 정정 근거 미기록"
    assert "v53" in src


def test_missing_qty_key_does_not_claim_zero_shares():
    """수량 정보가 없는 payload에 '켈리 0주'를 붙이면 없는 사실을 단정한다.

    구 호출부·부분 dict는 qty 키가 아예 없다. 부재를 0으로 읽으면
    게이트를 통과한 종목에 거짓 사유가 생긴다.
    """
    item = {"rr": 2.0, "poc_gap": 5.0, "vwap_gap": 1.0,
            "top_pick": True, "eligible": True}     # qty 키 없음
    assert "qty" not in item
    assert ts._candidate_watch_reason(item) == "공식 신규매수 조건 미충족"


def test_qty_present_and_zero_does_claim():
    item = {"rr": 2.0, "poc_gap": 5.0, "vwap_gap": 1.0,
            "top_pick": True, "eligible": True, "qty": 0}
    assert "켈리 0주" in ts._candidate_watch_reason(item)


def test_non_numeric_qty_is_ignored():
    item = {"rr": 2.0, "poc_gap": 5.0, "vwap_gap": 1.0,
            "top_pick": True, "eligible": True, "qty": "N/A"}
    assert ts._candidate_watch_reason(item) == "공식 신규매수 조건 미충족"
