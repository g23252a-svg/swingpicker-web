# -*- coding: utf-8 -*-
"""[v86] 일봉 CSV(date 컬럼) 두 개를 비교 — 첫 파일의 마지막 날짜가 둘째보다 같거나 늦으면 exit 0.

collect_kospi_daily.yml의 rebase 충돌 재시도에서 쓴다: 원격(auto_collect 커밋)이 더 새로운
kospi_daily/us_daily를 이미 갖고 있으면 묵은 로컬 결과로 덮지 않는다(exit 1).
둘째 파일이 없거나 읽을 수 없으면 덮어도 된다(exit 0).

    python scripts/daily_csv_newer.py <candidate.csv> <current.csv>
"""
from __future__ import annotations

import logging
import sys

import pandas as pd

_logger = logging.getLogger("daily_csv_newer")


def last_date(path: str) -> str:
    """date 컬럼의 최댓값(YYYYMMDD 문자열). 읽기 실패 시 ''."""
    try:
        d = pd.read_csv(path, usecols=["date"], dtype={"date": str})["date"].dropna()
    except Exception as e:  # 파일 없음·컬럼 없음
        _logger.warning("[v86] %s 날짜 읽기 실패: %s", path, e)
        return ""
    d = d.astype(str).str.replace("-", "").str[:8]
    return str(d.max()) if len(d) else ""


def is_newer_or_equal(candidate: str, current: str) -> bool:
    """candidate의 마지막 날짜가 current 이상이면 True(current를 못 읽으면 True)."""
    cand, cur = last_date(candidate), last_date(current)
    if not cand:
        return False
    return (not cur) or cand >= cur


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(argv) != 2:
        print("usage: daily_csv_newer.py <candidate.csv> <current.csv>", file=sys.stderr)
        return 2
    ok = is_newer_or_equal(argv[0], argv[1])
    print(f"candidate={last_date(argv[0]) or '-'} current={last_date(argv[1]) or '-'} -> "
          f"{'use candidate' if ok else 'keep current'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
