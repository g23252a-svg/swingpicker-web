# -*- coding: utf-8 -*-
"""[v49] 검증된 설정 고정 + 막다른 길 재시도 방지.

v49 대규모 탐색은 채택 후보가 하나도 없었다. 남는 가치는 두 가지다:
  ① 검증된 현행 설정이 근거 없이 흔들리지 않게 고정
  ② 실패한 경로를 수치와 함께 남겨 재시도를 막음

탐색 결과 요약
  · 인버스 타이밍: 397일·에피소드 8~24건에서 규칙 9개 전부 마이너스
      risk_off 3조건 -1.37%/5일(9에피) — 105일 창의 +3.07%(t=4.05)는 에피소드 1건의 착시
  · 지수 헤지: 5일 헤지오차 표준편차 6.68%p vs 얻을 알파 0.94% → 신호/잡음 0.14
      (동일가중 vs 코스피200 R²=0.52 · 베타 0.50)
  · 청산 격자: 손절 1~4ATR · 익절 1.5~5ATR · 보유 2~15일 · 트레일 1~2.5ATR
      → 현행 대비 유의한 개선 0건(전부 |t|<1.1)
      → 유의한 악화: 보유 10일 -1.01%p(t=-2.24) · 15일 -2.49%p(t=-4.82)
  · 알파 모델 변형 6종: 전부 기준선 이하. 분류 타깃 -0.63%p(t=-2.98)
  · v41 섹터 모멘텀: 제거 제안 철회 — 상호작용 재현 +0.165%p(t=+3.13, p=0.0025)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import collector_config as cc
import kelly_calibrator as kc
import version_info as vi


# ── ① TIME_STOP: 검증 안전구간 고정 ──

def test_time_stop_within_validated_range():
    """보유 10일 -1.01%p(t=-2.24) · 15일 -2.49%p(t=-4.82) → 상한 7일."""
    ts = cc.CollectorConfig().time_stop_days
    assert ts > 0, "타임스탑 비활성 — 장기보유 손실구간 노출"
    assert 3 <= ts <= 7, (
        f"time_stop_days={ts}: 검증 구간(3~7일) 밖. "
        "10일 이상은 실측에서 유의하게 열위(t=-2.24)")


# ── ② 섹터 모멘텀: 유지 근거와 함께 상수 고정 ──

def test_sector_momentum_retained_with_validated_bounds():
    """v46의 제거 제안은 잘못된 대상(원시 축)을 검정한 결과 — 철회됐다."""
    assert kc._SECMOM_LO == 0.5
    assert kc._SECMOM_HI == 1.5
    assert kc._COMBINED_CLIP == (0.3, 3.0)


def test_sector_momentum_rationale_records_decay():
    """효과가 초판 주장의 절반이고 후반이 미유의하다는 사실이 주석에 남아야 한다."""
    src = (ROOT / "kelly_calibrator.py").read_text(encoding="utf-8")
    assert "v49 재검증" in src
    assert "t=+3.13" in src, "재현 검정 통계량 누락"
    assert "미유의" in src, "후반 구간 한계 미기록"
    assert "잘못된 대상을 검정했다" in src, "오판 경위 미기록"


# ── ③ 알파 사이징 상수 (5개 대안이 못 넘은 구조) ──

def test_alpha_tilt_bounds_unchanged():
    assert kc._ALPHA_TILT_LO == 0.5
    assert kc._ALPHA_TILT_HI == 2.0
    assert kc._ALPHA_TILT_BASE == 1.0


# ── ④ 막다른 길 문서화 — 재시도 방지 ──

def _changelog_text() -> str:
    parts = []
    for e in vi.CHANGELOG:
        parts.append(str(e.get("title", "")))
        parts.extend(str(x) for x in e.get("items", []))
    return "\n".join(parts)


def test_dead_ends_documented_with_numbers():
    """수치 없는 '안 됨' 기록은 재시도를 막지 못한다."""
    txt = _changelog_text()
    # 인버스 타이밍
    assert "-1.37%" in txt, "인버스 규칙 실측치 누락"
    assert "9개가 전부 마이너스" in txt or "9개 전부 마이너스" in txt
    # 지수 헤지 베이시스
    assert "6.68%p" in txt, "헤지 오차 실측치 누락"
    # 청산·모델
    assert "t=-4.82" in txt, "장기보유 악화 통계량 누락"
    assert "t=-2.98" in txt, "분류 타깃 열위 통계량 누락"


def test_v41_removal_proposal_retracted():
    """v46의 잘못된 중립화 제안이 철회 표시 없이 남아있으면 안 된다."""
    txt = _changelog_text()
    assert "기대수익 없이 분산만 키우는 상태" not in txt, "철회되지 않은 오판 잔존"
    assert "철회" in txt and "+0.165%p" in txt


def test_v49_entry_declares_no_runtime_change():
    e = next((x for x in vi.CHANGELOG if str(x.get("version")) == "49.0.0"), None)
    assert e is not None, "v49 항목 없음"
    body = " ".join(str(i) for i in e["items"])
    assert "런타임 동작 변경 없음" in body, (
        "채택 후보가 없었다는 사실이 명시되어야 한다 — "
        "없는 엣지를 붙인 것처럼 읽히면 안 된다")
