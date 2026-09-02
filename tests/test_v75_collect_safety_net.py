# -*- coding: utf-8 -*-
"""v75 — 야간 배치가 조용히 안 도는 것을 막는다.

■ 무엇이 있었나 (2026-08-28 실측)

사용자: "오늘탭에 뜬 추천종목이 2일전과 똑같은데?" → "현재가도 27일 종가랑 안맞아"

확인해보니 **2026-08-27(목) 정시 실행이 아예 생성되지 않았다.**
실패가 아니라 스케줄 자체가 안 떴다 — auto_collect 실행 기록이
  #672  2026-08-26 11:35Z  schedule  success
  #673  2026-08-27 19:27Z  workflow_dispatch  (사람이 수동으로 부름)
로 8/27 정시(11:05Z) 실행이 통째로 비어 있다.

GitHub Actions 의 cron 은 best-effort 라 부하가 몰리면 조용히 드롭된다.
그 결과 8/28 새벽까지 화면이 8/26 배치를 띄웠고, **아무도 몰랐다** —
실패 알림도 없다. 실행이 없으면 실패도 없기 때문이다.

■ 처방

평일 22:40 KST 에 안전망 실행을 하나 더 둔다. 오늘자 산출물이 이미 있으면
스스로 건너뛰므로 정상적인 날에는 아무 일도 하지 않는다.
"""
import os
import re

import pytest

yaml = pytest.importorskip("yaml")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(ROOT, ".github", "workflows", "auto_collect.yml")


@pytest.fixture(scope="module")
def wf():
    with open(WF, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def src():
    return open(WF, encoding="utf-8").read()


def _on(wf):
    # PyYAML 은 `on:` 을 불리언 True 로 파싱한다.
    return wf.get("on") or wf.get(True)


PRIMARY_CRON = "23 11 * * 1-5"   # [v80] 11:05 → 11:23 — 정각 근처 혼잡 슬롯 회피


def test_primary_schedule_unchanged(wf):
    """정시 cron이 존재하고 평일 11시대(UTC)여야 한다 — 분은 v80에서 옮겼다."""
    crons = [c["cron"] for c in _on(wf)["schedule"]]
    assert PRIMARY_CRON in crons, "정시(20:23 KST)를 없애면 안 된다"


def test_safety_net_schedule_exists(wf):
    crons = [c["cron"] for c in _on(wf)["schedule"]]
    assert len(crons) >= 2, "안전망 스케줄이 없다"
    extra = [c for c in crons if c != PRIMARY_CRON]
    assert extra, "안전망 cron 없음"
    m = re.fullmatch(r"(\d+) (\d+) \* \* 1-5", extra[0])
    assert m, f"안전망 cron 형식이 예상과 다름: {extra[0]}"
    minute, hour = int(m.group(1)), int(m.group(2))
    # 정시(11:05Z)보다 뒤여야 하고, KRX 야간 차단(대략 UTC 15시 이후)은 피한다
    assert 11 < hour < 15, f"안전망 시각 {hour}:{minute:02d}Z 가 부적절"


def test_manual_dispatch_still_available(wf):
    assert "workflow_dispatch" in _on(wf)


def test_guard_step_exists_right_after_checkout(wf):
    steps = wf["jobs"]["build"]["steps"]
    assert "Checkout" in steps[0].get("name", "")
    assert steps[1].get("id") == "guard", "가드는 체크아웃 바로 뒤여야 한다"


def test_guard_skips_when_today_csv_exists(src):
    i = src.index("id: guard")
    blk = src[i:i + 2600]     # [v80] 지연 발화 스킵 분기 둘이 늘어 창을 넓혔다
    assert 'TZ=Asia/Seoul date +%Y%m%d' in blk, "거래일은 KST 기준이어야 한다"
    assert 'data/recommend_${TODAY}.csv' in blk
    assert 'skip=true' in blk and 'skip=false' in blk
    assert "$GITHUB_OUTPUT" in blk


def test_manual_dispatch_bypasses_guard(src):
    i = src.index("id: guard")
    blk = src[i:i + 2600]     # [v80] 지연 발화 스킵 분기 둘이 늘어 창을 넓혔다
    assert "github.event_name" in blk and "workflow_dispatch" in blk, (
        "사람이 일부러 부른 실행은 건너뛰면 안 된다")


def test_every_working_step_is_guarded(wf):
    steps = wf["jobs"]["build"]["steps"]
    unguarded = [s.get("name") for s in steps[2:]
                 if s.get("if") != "steps.guard.outputs.skip != 'true'"]
    assert unguarded == [], f"가드 없이 도는 스텝: {unguarded}"


def test_commit_step_is_guarded(wf):
    steps = wf["jobs"]["build"]["steps"]
    commit = [s for s in steps if "Commit" in (s.get("name") or "")]
    assert commit, "커밋 스텝을 못 찾음"
    assert commit[0].get("if") == "steps.guard.outputs.skip != 'true'"


def test_concurrency_group_prevents_double_run(wf):
    c = wf.get("concurrency") or {}
    assert c.get("group"), "동시 실행 방지 그룹이 없다"
    assert c.get("cancel-in-progress") is False, (
        "진행 중인 수집을 취소하면 반쪽 산출물이 커밋될 수 있다")


def test_reason_is_documented(src):
    assert "2026-08-27" in src and "best-effort" in src, (
        "왜 안전망을 뒀는지가 파일에 남아 있어야 한다")
