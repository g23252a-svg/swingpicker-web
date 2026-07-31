# -*- coding: utf-8 -*-
"""[v48] 시장 국면 서술이 실측과 어긋나지 못하게 고정.

v46 창고에 사실과 다른 서술 두 개가 배포됐다:
  ① "누적 105일 전 구간이 하락장이다"
     → 틀렸다. 당일 등락 오른 날 56/105일(53%), 당일 평균 +0.03%(보합).
       4월은 상승장(당일 +1.26%/일 · 향후5일 +3.02% · 19/22 양수), 3월 보합.
       -1.28%는 '향후 5일 수익의 평균'이며 음수인 날은 63%였다.
  ② "UP 레짐은 단 하루도 없다"
     → MARKET_REGIME 컬럼이 105일 중 12일에만 존재. 12일을 105일로 일반화한 오류.

정정된 성격 규정 = 방어형 알파
  시장 상승일 28일: 전략 +3.91% vs 유니버스 +4.06%  엣지 -0.15 (t=-0.39)
  시장 하락일 48일: 전략 -2.54% vs 유니버스 -4.40%  엣지 +1.85 (t=+6.56)
  4월 단독: +2.68%/5일 · 승률 58.7% (절대수익 플러스)
→ 상승장에서는 시장만큼 벌고, 하락장에서 덜 잃는다.

문서에 거짓이 남으면 나중 분석이 그것을 전제로 또 틀린다. 그래서 테스트로 박는다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import version_info as vi

BANNED = [
    "105일 전 구간이 하락장이다",
    "UP 레짐은 단 하루도 없다",
    "어떤 조합도 플러스가 아니었다",
]


CORRECTION_VERSION = "48.0.0"


def _all_changelog_text() -> str:
    parts = []
    for e in vi.CHANGELOG:
        parts.append(str(e.get("title", "")))
        parts.extend(str(x) for x in e.get("items", []))
    return "\n".join(parts)


def _text_excluding_correction() -> str:
    """정정 항목은 틀린 문장을 '인용'해야 하므로 스캔에서 제외한다."""
    parts = []
    for e in vi.CHANGELOG:
        if str(e.get("version")) == CORRECTION_VERSION:
            continue
        parts.append(str(e.get("title", "")))
        parts.extend(str(x) for x in e.get("items", []))
    return "\n".join(parts)


def test_false_regime_claims_not_asserted_anywhere():
    """정정 항목 밖에서는 틀린 서술이 주장되어선 안 된다."""
    txt = _text_excluding_correction()
    for bad in BANNED:
        assert bad not in txt, f"실측과 어긋난 서술 잔존: {bad!r}"


def test_correction_entry_marks_claims_as_false():
    """정정 항목은 인용만 하고 끝내지 말고 '틀렸다'를 명시해야 한다."""
    e = next(x for x in vi.CHANGELOG if str(x["version"]) == CORRECTION_VERSION)
    body = " ".join(str(i) for i in e["items"])
    assert "틀렸다" in body
    assert "철회" in body, "잘못된 단정을 철회한다는 표시가 없음"


def test_correction_entry_present_and_quantified():
    txt = _all_changelog_text()
    # 정정 사실과 핵심 수치가 문서에 남아 있어야 한다
    assert "56/105" in txt, "오른 날 비율(53%) 실측치 누락"
    assert "방어형 알파" in txt, "정정된 성격 규정 누락"
    assert "12일에만 존재" in txt, "MARKET_REGIME 커버리지 한계 누락"
    assert "+2.68%" in txt, "4월 절대수익 실측치 누락"


def test_defensive_alpha_numbers_locked():
    """국면별 성능 수치가 문서에서 조용히 바뀌지 못하게 고정."""
    txt = _all_changelog_text()
    for token in ["+3.91%", "-2.54%", "+1.85", "t=+6.56", "t=-0.39"]:
        assert token in txt, f"국면별 실측 수치 누락: {token}"


def test_v48_entry_exists_with_correct_shape():
    e = next((x for x in vi.CHANGELOG if str(x.get("version")) == "48.0.0"), None)
    assert e is not None, "v48 정정 항목 없음"
    assert e["type"] in ("minor", "major")
    assert len(e["items"]) >= 4
    assert "정정" in e["title"]


def test_changelog_versions_are_descending():
    """정정 항목을 끼워넣다 순서가 깨지지 않았는지."""
    def key(v):
        # 39.1.0 · 22.3.5c 처럼 접미사가 붙은 버전도 안전하게 정렬.
        out = []
        for part in str(v).split("."):
            digits = "".join(c for c in part if c.isdigit())
            suffix = "".join(c for c in part if c.isalpha())
            out.append((int(digits or 0), suffix))
        return tuple(out)
    vs = [key(e["version"]) for e in vi.CHANGELOG]
    assert vs == sorted(vs, reverse=True), f"버전 순서 역전: {vs[:5]}"
