# -*- coding: utf-8 -*-
"""[v59] 서드파티 모듈을 sys.modules에서 지우는 테스트가 세션 전체를 오염시키는 것을 막는다.

■ 무엇이 있었나 (2026-08-11 실측)
  tests/test_v59_sector_gap.py는 **단독 실행 25/25 통과**인데 전체 스위트에서
  9건 실패했다. 실패 메시지는 차트와 무관해 보였다:
    ValueError: Invalid value of type
                'plotly.graph_objs.layout._template.Template'
                received for the 'template' property of layout
  즉 "Template에 Template을 넣을 수 없다"는 말이었다.

  원인은 **같은 클래스가 두 벌 존재**하는 것이었다. 몇몇 테스트 픽스처가
  격리를 위해 이렇게 했다:

      for mod in list(sys.modules):
          if mod == "plotly" or mod.startswith("plotly."):
              del sys.modules[mod]          # ← monkeypatch가 아닌 raw del

  plotly를 sys.modules에서 지워도, 이미 그것을 import해 둔 모듈
  (chart_components 등)은 **옛 클래스 객체를 계속 들고 있다**. 이후 누군가
  plotly를 다시 import하면 모듈 코드가 재실행되어 Template·Figure 클래스가
  **새로 만들어진다**. 그러면 옛 Figure의 검증기가 새 Template 인스턴스를
  "모르는 타입"으로 거부한다. 실행 순서에만 의존하므로 단독 실행은 통과하고,
  파일을 몇 개 같이 돌릴 때만 터진다 — 가장 찾기 어려운 형태였다.

  (raw del이 monkeypatch.delitem보다 나쁜 이유가 하나 더 있다. 픽스처가
   지운 뒤에 `monkeypatch.setitem(sys.modules, "plotly", 가짜)`를 하면
   monkeypatch는 "원래 없었음"을 기록하므로, teardown에서 **원복이 아니라
   삭제**를 한다. 진짜 plotly가 세션에서 영구히 사라진다.)

■ 이 가드가 하는 일
  1. 세션 시작 시 plotly 모듈 집합을 스냅샷한다.
  2. 매 테스트 teardown 후(픽스처 원복이 끝난 뒤) 스냅샷과 어긋난 항목을
     **원복**한다 → 뒤에 도는 테스트는 오염되지 않는다.
  3. 어긋난 사실을 **조용히 넘기지 않고**, 원인이 된 테스트의 nodeid를 달아
     그 테스트를 실패시킨다. 조용히 고쳐주면 누출이 계속 새로 생긴다.
     (v55.4 ml_engine 스텁 · v56 죽은 게이트와 같은 취급이다.)

■ 왜 prefix를 화이트리스트로 두는가
  대상은 "클래스 동일성이 동작에 영향을 주는 서드파티 라이브러리"다.
  프로젝트 자체 모듈(services/components/…)은 테스트가 의도적으로 다시
  import해야 하므로 보호 대상이 아니다.
"""
from __future__ import annotations

import logging
import sys

import pytest

_log = logging.getLogger("tests.isolation_guard")

# 보호 대상 서드파티.
#   plotly : 클래스 동일성이 검증 로직에 직접 쓰인다(위 사례).
#   nicegui: 지워지면 다른 테스트가 "없으니 만들자"며 가짜를 영구 설치한다
#            (tests/test_pick_top1_eligible_hotfix.py가 그랬다).
PROTECTED_PREFIXES: tuple[str, ...] = ("plotly", "nicegui")

_snapshot: dict[str, object] = {}
_leaks: list[str] = []


def _is_protected(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in PROTECTED_PREFIXES)


def _current() -> dict[str, object]:
    return {k: v for k, v in list(sys.modules.items()) if _is_protected(k)}


def pytest_configure(config):
    """세션 시작 시 보호 대상을 완전히 로드해 두고 스냅샷한다.

    plotly는 서브모듈을 지연 로딩하므로 Figure를 한 번 만들어
    검증기 서브모듈까지 실제로 import되게 한다.
    """
    try:
        import plotly.express  # noqa: F401
        import plotly.graph_objects as go

        go.Figure()
    except Exception as e:      # 라이브러리 미설치 환경 — 보호할 것이 없다
        _log.info("plotly 보호 대상 로드 실패 — 격리 가드 비활성: %s", e)
    try:
        import nicegui  # noqa: F401
    except Exception as e:
        _log.info("nicegui 보호 대상 로드 실패 — 격리 가드 비활성: %s", e)
    _snapshot.update(_current())
    if not _snapshot:
        _log.warning("보호 대상 서드파티가 없어 테스트 격리 가드가 검사하지 않는다")


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """픽스처 원복이 모두 끝난 뒤 검사한다(trylast).

    monkeypatch가 정상 원복하는 경우를 누출로 오판하지 않기 위해 순서가 중요하다.
    """
    if not _snapshot:
        return
    drifted = [k for k, v in _snapshot.items() if sys.modules.get(k) is not v]
    if not drifted:
        # 지연 로딩으로 새로 들어온 서브모듈도 이후부터 보호 대상에 넣는다.
        for k, v in _current().items():
            _snapshot.setdefault(k, v)
        return

    for k in drifted:                       # 뒤에 도는 테스트부터 구제
        _snapshot_val = _snapshot[k]
        sys.modules[k] = _snapshot_val
    _leaks.append(item.nodeid)
    shown = ", ".join(sorted(drifted)[:6])
    more = "" if len(drifted) <= 6 else f" (외 {len(drifted) - 6}개)"
    raise RuntimeError(
        f"[테스트 격리 위반] 이 테스트가 서드파티 모듈 {len(drifted)}개를 "
        f"sys.modules에서 교체/삭제한 채로 끝났다: {shown}{more}\n"
        "  → 같은 클래스가 두 벌 생겨 뒤에 도는 테스트가 엉뚱한 곳에서 죽는다"
        " (예: \"Invalid value of type ...layout._template.Template\").\n"
        "  → 서드파티는 지우지 말고, 필요하면 monkeypatch.setitem으로만 덮어라"
        " (raw del sys.modules 금지). 자세한 내용은 tests/conftest.py 참고."
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if _leaks:
        terminalreporter.write_line(
            f"[테스트 격리] 서드파티 모듈 누출 {len(_leaks)}건 — "
            + ", ".join(_leaks[:5]), red=True)
