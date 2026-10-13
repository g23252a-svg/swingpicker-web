# -*- coding: utf-8 -*-
"""[v84.2] 배치 대상 세션 — scripts/batch_target.sh 를 가짜 시계로 실제 실행해 검증한다.

사고: v84.1이 prefetch 체인에 '16시(KST) 전이면 생략'을 넣었다. 10/12(월) prefetch
cron이 10/13 00:55 KST에 늦게 발화 → 체인 생략, auto_collect cron(05:20·06:50 KST)도
'16시 전' 가드로 생략 → 10/12 배치가 통째로 빠지고 화면이 10/8 픽(휴림로봇)을 계속
보여줬다(사용자 신고). v84.1의 전제('자정 넘긴 실행은 다음 날 장전 배치를 만든다')는
틀렸다 — collector 는 데이터가 있는 최신 세션을 고른다(10/6 00:41 실행 →
recommend_20261005 생성으로 확인).
"""
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "batch_target.sh"
KST = timezone(timedelta(hours=9))
REAL_DATE = shutil.which("date")


def _run(kst_str: str, tmp_path: Path) -> dict:
    """KST 시각을 흉내 낸 date 로 스크립트를 실행하고 출력 변수를 돌려준다."""
    epoch = int(datetime.strptime(kst_str, "%Y-%m-%d %H:%M").replace(tzinfo=KST).timestamp())
    fake = tmp_path / "date"
    fake.write_text(
        "#!/bin/bash\n"
        'for a in "$@"; do case "$a" in -d|-d*|--date*) exec ' + REAL_DATE + ' "$@";; esac; done\n'
        "exec " + REAL_DATE + ' -d "@$FAKE_EPOCH" "$@"\n', encoding="utf-8")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "FAKE_EPOCH": str(epoch)}
    out = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env, check=True).stdout
    return dict(line.split("=", 1) for line in out.strip().splitlines())


@pytest.mark.skipif(REAL_DATE is None or not Path("/usr/share/zoneinfo/Asia/Seoul").exists(),
                    reason="date/tzdata 없음")
class TestWindows:
    @pytest.mark.parametrize("now,window,target", [
        ("2026-10-12 21:00", "evening", "20261012"),     # 월 저녁 — 정상 체인
        ("2026-10-13 00:55", "overnight", "20261012"),   # ★ 10/12 사고 시각 — 직전 평일 회수
        ("2026-10-13 05:20", "overnight", "20261012"),   # auto_collect 새벽 cron
        ("2026-10-13 07:59", "overnight", "20261012"),
        ("2026-10-13 08:00", "market", ""),              # 장 시작 전후 — 실행 안 함
        ("2026-10-13 09:53", "market", ""),
        ("2026-10-13 15:35", "market", ""),              # prefetch 정시 발화 — 데이터 미확정
        ("2026-10-13 16:00", "evening", "20261013"),
        ("2026-10-17 01:00", "overnight", "20261016"),   # 토 새벽 → 금 회수
        ("2026-10-18 01:00", "overnight", "20261016"),   # 일 새벽 → 금
        ("2026-10-19 03:00", "overnight", "20261016"),   # 월 새벽 → 금 (월 장전이므로 유효)
        ("2026-10-17 20:00", "weekend", ""),
        ("2026-10-18 23:00", "weekend", ""),
    ])
    def test_window_and_target(self, tmp_path, now, window, target):
        got = _run(now, tmp_path)
        assert got == {"BATCH_WINDOW": window, "BATCH_TARGET": target}, now


class TestWiring:
    def _wf(self, name):
        return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")

    def test_chain_uses_script_and_target_not_calendar_today(self):
        y = self._wf("prefetch_flow.yml")
        i = y.index("Chain auto_collect"); blk = y[i:i + 2500]
        assert 'eval "$(bash scripts/batch_target.sh)"' in blk
        assert "recommend_${BATCH_TARGET}.csv" in blk and "holiday_skip_${BATCH_TARGET}.json" in blk
        assert 'date +%H)" -lt 16' not in blk, "v84.1 오진 가드가 되살아났다"
        # 장중·주말·이미 있음·휴장 확정이 dispatch 앞에서 모두 걸러진다
        i_disp = blk.index("gh workflow run auto_collect.yml")
        for key in ('"market"', '"weekend"', "recommend_${BATCH_TARGET}.csv", "holiday_skip_${BATCH_TARGET}.json"):
            assert blk.index(key) < i_disp, key

    def test_batch_guard_uses_same_script(self):
        w = self._wf("auto_collect.yml")
        i = w.index("id: guard"); blk = w[i:i + 3000]
        assert 'eval "$(bash scripts/batch_target.sh)"' in blk
        assert 'data/recommend_${BATCH_TARGET}.csv' in blk and 'data/holiday_skip_${BATCH_TARGET}.json' in blk
        i_disp = blk.index('"workflow_dispatch" ]')
        assert i_disp < blk.index('"market"') < blk.index('data/recommend_${BATCH_TARGET}.csv'), \
            "dispatch 우선 → 장중 → 대상 CSV 순"

    def test_guard_runs_after_checkout_so_script_exists(self):
        d = yaml.safe_load(self._wf("auto_collect.yml"))
        steps = d["jobs"]["build"]["steps"]
        assert "Checkout" in steps[0]["name"] and steps[1].get("id") == "guard"

    def test_holiday_marker_detected_by_mtime_not_today(self):
        """새벽 회수 실행은 직전 평일이 대상이라 마커 이름이 오늘이 아니다."""
        w = self._wf("auto_collect.yml")
        i = w.index("Run Collector"); blk = w[i:i + 1800]
        assert 'touch "$RUNNER_TEMP/collect_start"' in blk
        assert blk.index('touch "$RUNNER_TEMP/collect_start"') < blk.index("python collector.py")
        assert "-newer \"$RUNNER_TEMP/collect_start\"" in blk
        assert "holiday_skip_${TODAY}" not in blk
