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
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

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
    _check_data_dirty(item)
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


def _check_data_dirty(item) -> None:
    """[v71] data/ 를 건드린 테스트를 그 자리에서 지목한다.

    별도 훅으로 두지 않는다 — 같은 이름의 pytest 훅을 한 모듈에 두 번 정의하면
    뒤엣것이 앞엣것을 덮어써서 모듈 누출 가드가 통째로 사라진다.
    """
    if not _DATA_GUARD_ON or _DATA_DIR is None:
        return
    global _data_sig
    now = _scan_data()
    if not _data_sig:
        _data_sig = now
        return
    changed = sorted(k for k in set(_data_sig) | set(now)
                     if _data_sig.get(k) != now.get(k))
    if not changed:
        return
    _data_sig = now                       # 뒤 테스트가 같은 걸로 또 죽지 않게
    _data_dirty.append((item.nodeid, changed))
    shown = ", ".join(changed[:5])
    more = "" if len(changed) <= 5 else f" (외 {len(changed) - 5}개)"
    raise RuntimeError(
        f"[data/ 오염] 이 테스트가 저장소의 data/ 파일 {len(changed)}개를 "
        f"바꿨다: {shown}{more}\n"
        "  → 프로덕션 코드는 data_dir 에 캐시를 쓴다. 테스트가 진짜 data/ 를 "
        "넘기면 그 캐시가 저장소에 떨어진다.\n"
        "  → real_data_mirror 픽스처를 써라: "
        "d = real_data_mirror(\"ohlcv_cache_*.parquet\", ...)\n"
        "  → 자세한 내용은 tests/conftest.py 참고."
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if _leaks:
        terminalreporter.write_line(
            f"[테스트 격리] 서드파티 모듈 누출 {len(_leaks)}건 — "
            + ", ".join(_leaks[:5]), red=True)
    if _data_dirty:
        terminalreporter.write_line(
            f"[data/ 오염] {len(_data_dirty)}건 — "
            + ", ".join(n for n, _ in _data_dirty[:5]), red=True)


# ══════════════════════════════════════════════════════════════════
#  [v71] 테스트가 저장소의 data/ 를 건드리는 것을 막는다
# ══════════════════════════════════════════════════════════════════
"""■ 무엇이 있었나 (2026-08-27 실측)

  `git status` 에 `data/ohlcv_union_hl.parquet` 이 수정된 채로 남았다.
  내용은 HEAD와 **완전히 동일**했다(`equals()` True) — 데이터는 그대로인데
  파일 바이트만 다시 쓰인 것이다.

  범인은 v65에서 내가 쓴 실데이터 회귀 테스트였다:

      def test_real_data_skips_known_holidays(self):
          res = pr.compute_pick_reliability(str(DATA))   # ← 진짜 data/

  `pick_reliability._build_hl_union(data_dir)` 는 설계상 합집합을
  `<data_dir>/ohlcv_union_hl.parquet` 에 캐싱한다. 프로덕션에서는 옳은 동작이다.
  테스트가 진짜 `data/` 를 넘겼기 때문에 캐시가 저장소에 떨어졌다.

■ 왜 그냥 커밋하면 안 되나

  내용이 같은 바이너리를 커밋하면 diff 소음만 남고, "테스트가 저장소를
  더럽힌다"는 사실이 묻힌다. 언젠가 내용까지 바뀌는 사고가 나도 알아채지
  못한다. 실제로 이 세션 앞부분에서 `data/feature_cache_schema.json` 이
  n_stocks 672→595 로 덮여 있었고 그때는 범인을 못 찾았다.

■ 처방

  1. `real_data_mirror` — 실데이터를 심링크로 미러링한 임시 디렉터리.
     읽기는 그대로 되고, 새로 만들어지는 산출물은 tmp 에 떨어진다.
  2. 아래 훅 — 테스트 하나가 끝날 때마다 data/ 의 (이름, mtime, 크기)
     서명을 비교해 **더럽힌 테스트를 그 자리에서 지목**한다.
     심링크를 따라가 원본을 덮는 경우까지 잡힌다.
     끄려면 SWINGPICKER_SKIP_DATA_GUARD=1.
"""

_DATA_DIR = _ROOT / "data" if (_ROOT / "data").is_dir() else None
_data_sig: dict = {}
_data_dirty: list = []
_DATA_GUARD_ON = os.environ.get("SWINGPICKER_SKIP_DATA_GUARD", "") != "1"


def _scan_data() -> dict:
    if _DATA_DIR is None:
        return {}
    out = {}
    try:
        with os.scandir(_DATA_DIR) as it:
            for e in it:
                if not e.is_file(follow_symlinks=False):
                    continue
                st = e.stat()
                out[e.name] = (st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    return out


def build_data_mirror(dest, *patterns: str) -> str:
    """실데이터를 심링크로 미러링한 디렉터리를 dest 에 만든다.

    픽스처가 아니라 평범한 함수인 것은 의도다 — 클래스/모듈 스코프가 필요한
    테스트(예: 14초짜리 CD.measure 를 클래스당 한 번만 돌리는 곳)가
    tmp_path_factory 로 자기 스코프의 디렉터리를 잡아 쓸 수 있어야 한다.
    세션 공유 미러를 하나 두면 A가 쓴 캐시를 B가 읽어 순서 의존이 생긴다.

    패턴을 명시하게 한 것도 의도다 — 테스트가 무엇을 읽는지 드러난다.
    """
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    if _DATA_DIR is None:
        return str(d)
    for pat in patterns:
        for src in sorted(_DATA_DIR.glob(pat)):
            dst = d / src.name
            if not dst.exists():
                dst.symlink_to(src)
    return str(d)


@pytest.fixture
def real_data_mirror(tmp_path):
    """함수 스코프용 래퍼.

    사용:
        def test_x(real_data_mirror):
            d = real_data_mirror("ohlcv_cache_*.parquet", "recommend_*.csv")
            res = pr.compute_pick_reliability(d)
    """
    def _mk(*patterns: str) -> str:
        return build_data_mirror(tmp_path / "data_mirror", *patterns)
    return _mk
