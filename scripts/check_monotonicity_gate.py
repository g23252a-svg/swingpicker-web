#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_monotonicity_gate.py — monotonicity_report의 CI HARD 게이트 판정.

[v56] 이 로직은 종전에 .github/workflows/v22_monotonicity.yml **안에 인라인
python으로** 박혀 있었다. YAML 안의 코드는 테스트할 수 없고, 그래서 아래 두
구멍이 오래 남아 있었다:

  1) **스테일 리포트 수용.** 워크플로가 `python daily_briefing.py … || echo
     "::warning::"`로 실패를 삼킨다. data/monotonicity_report_latest.json은
     git에 커밋돼 있으므로, 생성이 실패해도 체크아웃된 **과거 리포트**가 남고
     게이트는 그것을 검증해 통과했다. 날짜를 비교하는 코드가 없었다.
     → asof_ymd가 기대 날짜와 다르면 push/schedule에서는 FAIL, PR에서는 WARN.

  2) **공허한 PASS를 검증된 PASS와 동일하게 표시.** TOP_PICK·PRODUCTION_BUY가
     0건인 날에는 하드 게이트가 '검사할 대상이 없어' 통과한다. 종전 출력은
     "최종: HARD=PASS ✅"만 찍어서, 실제로는 4개 중 3개가 공허하거나 SKIP인
     상태를 강한 보증처럼 보이게 했다.
     → subject_n(대상 수)을 읽어 `PASS(대상 0건)`으로 구분 표시하고,
       "검증된 하드게이트 n/m"을 요약에 남긴다.

판정 기준은 바꾸지 않았다(대상 0건을 FAIL로 만들면 정상적인 무픽 구간에 CI가
막힌다). 바뀐 것은 **정직성**과 **스테일 차단**이다.

사용:
    python scripts/check_monotonicity_gate.py \
        --report data/monotonicity_report_latest.json \
        --event "$GITHUB_EVENT_NAME" --expect-ymd 20260807
"""
import argparse
import json
import os
import sys

# 리포트 부재를 통과시켜도 되는 이벤트 (작업자가 collector를 못 돌리는 맥락)
LENIENT_EVENTS = ("pull_request", "workflow_dispatch")


def evaluate(report: dict, event: str = "", expect_ymd: str = "") -> tuple[bool, list[str]]:
    """(통과 여부, 출력 줄들)."""
    lines = []
    lenient = event in LENIENT_EVENTS
    fatal = []

    asof = str(report.get("asof_ymd") or "")
    lines.append("=" * 60)
    lines.append(f"📊 monotonicity_report ({asof or '날짜없음'})")
    lines.append(f"   event={event}")
    lines.append("=" * 60)

    # ── 스테일 검사 ──
    if expect_ymd and asof != str(expect_ymd):
        msg = (f"리포트가 기대 날짜와 다르다: asof_ymd={asof or '없음'} "
               f"≠ 기대 {expect_ymd} — daily_briefing 생성이 실패해 "
               f"과거 리포트를 검증하고 있을 수 있다")
        if lenient:
            lines.append(f"  ⚠️ 스테일 리포트 (event={event} → 경고): {msg}")
        else:
            lines.append(f"  ❌ 스테일 리포트: {msg}")
            fatal.append(msg)

    for key, label in (
        ("top_pick_count", "TOP_PICK count"),
        ("declared_wr_top_pick", "declared_wr_top_pick"),
        ("declared_wr_active", "declared_wr_active"),
        ("realized_wr", "realized_wr"),
        ("declared_vs_realized_gap_top_pick", "gap (top_pick)"),
        ("declared_vs_realized_gap_active", "gap (active)"),
        ("avg_ret_excess_pct", "avg_ret_excess_pct"),
        ("market_map_coverage", "market_map_coverage"),
        ("top_pick_stable_7day_avg", "stable_7day_avg"),
    ):
        lines.append(f"  {label}: {report.get(key)}")
    lines.append("")

    # ── HARD ──
    hard = report.get("ci_hard", []) or []
    verified = 0
    vacuous = 0
    skipped = 0
    unknown = 0
    lines.append("🛡️ HARD Gates:")
    for g in hard:
        status = g.get("status")
        n = g.get("subject_n")
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(status, "❓")
        if status == "SKIP":
            skipped += 1
            suffix = " [검사 안 됨]"
        elif status == "PASS" and n == 0:
            vacuous += 1
            suffix = " [대상 0건 — 공허한 통과]"
        elif status == "PASS" and n is None:
            # v56 이전 산출물 — 대상 수를 모른다. '검증됨'으로 셈하면 안 된다.
            unknown += 1
            suffix = " [대상 수 미기록 — 검증 여부 불명]"
        elif status == "PASS":
            verified += 1
            suffix = f" [대상 {n}건 검증]"
        else:
            suffix = ""
        lines.append(f"  {icon} {g.get('gate')}: {status} ({g.get('detail', '')}){suffix}")

    # ── SOFT ──
    lines.append("")
    lines.append("💡 SOFT Gates:")
    for g in report.get("ci_soft", []) or []:
        icon = {"OK": "✅", "WARN": "⚠️"}.get(g.get("status"), "❓")
        lines.append(f"  {icon} {g.get('gate')}: {g.get('status')} ({g.get('detail', '')})")

    hard_pass = bool(report.get("ci_hard_all_pass", False))
    soft_warn = bool(report.get("ci_soft_any_warn", False))

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"최종: HARD={'PASS ✅' if hard_pass else 'FAIL ❌'}, "
                 f"SOFT={'WARN ⚠️' if soft_warn else 'OK ✅'}")
    # 정직성 요약 — '통과'가 '검증됨'을 뜻하지 않을 수 있다는 사실을 숨기지 않는다
    lines.append(f"검증된 하드게이트: {verified}/{len(hard)} "
                 f"(공허한 통과 {vacuous} · SKIP {skipped}"
                 + (f" · 대상수 미기록 {unknown}" if unknown else "") + ")")
    if unknown:
        lines.append(f"ℹ️ {unknown}건은 subject_n이 없는 v56 이전 산출물이다 — "
                     f"다음 배치가 리포트를 다시 쓰면 자동 해소된다")
    if hard_pass and verified == 0 and hard:
        lines.append("⚠️ 하드게이트가 전부 공허/SKIP이다 — 이 PASS는 "
                     "'위반 없음'이 아니라 '검사 대상 없음'을 뜻한다")
    lines.append("=" * 60)

    ok = hard_pass and not fatal
    return ok, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="data/monotonicity_report_latest.json")
    ap.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    ap.add_argument("--expect-ymd", default="")
    args = ap.parse_args(argv)

    if not os.path.exists(args.report):
        if args.event in LENIENT_EVENTS:
            print(f"⏭️  monotonicity_report 없음 — SKIP (event={args.event})")
            print("   PR 작업자는 보통 collector를 못 돌리므로 통과시킴.")
            return 0
        print(f"❌ monotonicity_report 없음 (event={args.event})")
        print("   main push 또는 schedule 실행에서는 collector가 돌고")
        print("   data/monotonicity_report_latest.json이 생성되어야 합니다.")
        return 1

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    ok, lines = evaluate(report, event=args.event, expect_ymd=args.expect_ymd)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
