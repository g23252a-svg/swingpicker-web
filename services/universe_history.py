# -*- coding: utf-8 -*-
"""universe_history.py — 시점별(point-in-time) 상장 유니버스 SSOT [v71]

## 왜 필요한가

2026-08-27 선택 로직 탐색에서 확인된 것:
`data/ohlcv_cache_*.parquet`는 593종목이고 **중도 소멸 흔적이 0건**이다.
상장폐지·관리종목 편입으로 사라진 종목이 통째로 빠져 있다는 뜻이다.
그 상태에서는 편입 게이트(`top_n=600`, `min_turnover_eok=50`,
`min_mcap_eok=1000`) 완화를 검정할 수 없다 — 저유동·소형 구간이야말로
생존편향이 가장 심한 구간이기 때문이다.

이 모듈은 `data/krx_codes_YYYYMMDD.csv` 일별 스냅샷(130개, 합집합 2,923종목)에서
**시점별 상장 여부**를 복원한다. 시세는 복원하지 않는다 — 그건
`scripts/backfill_universe_ohlcv.py`가 네트워크가 허용된 환경에서 채운다.

## 붕괴 스냅샷 방어

스냅샷 중 14개가 수집 실패로 잘려 있다(20260227 107종목, 20260813 381종목 등).
그대로 쓰면 하루에 2,400종목이 상장폐지된 것으로 잡힌다 — v65에서 잡은
팬텀 세션과 같은 부류의 결함이다. 롤링 중위 대비 비율로 걸러낸다.

## 소멸 판정

'유효 스냅샷에 있다가 이후 유효 스냅샷 CONFIRM_SNAPSHOTS개 연속으로 없음'만
소멸로 인정한다. 관측 구간 끝에 가까워 확인 창이 모자라면
`status=UNCONFIRMED`로 남기고 소멸로 세지 않는다 — 모르는 것을 안다고
말하지 않기 위해서다.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

logger = logging.getLogger(__name__)

# ── 정책 상수 (SSOT) ────────────────────────────────────────────
SNAPSHOT_GLOB = "krx_codes_*.csv"
CODE_COL = "종목코드"
NAME_COL = "종목명"

#: 기준 크기 대비 이 비율 미만이면 수집 실패로 간주하고 버린다.
MIN_VALID_RATIO = 0.5
#: 기준 크기를 구할 때 반복 절사 횟수. 오염된 관측을 떨어내고 수렴시킨다.
TRIM_PASSES = 3
#: 이 비율을 넘게 버리면 기준 자체를 의심해야 한다 — 경고를 남긴다.
DEGENERATE_ALARM_RATIO = 0.30
#: 소멸을 인정하기 위해 연속으로 부재해야 하는 유효 스냅샷 수.
CONFIRM_SNAPSHOTS = 3

STATUS_LISTED = "LISTED"
STATUS_DELISTED = "DELISTED"
STATUS_UNCONFIRMED = "UNCONFIRMED"

CACHE_NAME = "universe_history_latest.json"


@dataclass
class Snapshot:
    ymd: str
    path: str
    codes: Set[str]
    names: Dict[str, str] = field(default_factory=dict)
    valid: bool = True
    reason: str = ""


def _ymd_of(path: str) -> Optional[str]:
    m = re.search(r"(\d{8})", os.path.basename(path))
    return m.group(1) if m else None


def load_snapshots(data_dir: str = "data") -> List[Snapshot]:
    """krx_codes_*.csv 를 날짜순으로 읽어 Snapshot 목록을 만든다.

    읽기 실패한 파일은 valid=False 로 남긴다 — 조용히 건너뛰면
    '스냅샷이 원래 없었다'와 구분이 되지 않는다.
    """
    out: List[Snapshot] = []
    for p in sorted(glob.glob(os.path.join(data_dir, SNAPSHOT_GLOB))):
        ymd = _ymd_of(p)
        if ymd is None:
            logger.warning("[v71] 날짜를 못 읽은 스냅샷 건너뜀: %s", p)
            continue
        try:
            d = pd.read_csv(p, dtype=str)
        except Exception as e:  # 파일 손상 / 인코딩 등
            logger.warning("[v71] 스냅샷 읽기 실패 %s: %s", p, e)
            out.append(Snapshot(ymd=ymd, path=p, codes=set(), valid=False,
                                reason=f"read_error:{type(e).__name__}"))
            continue
        if CODE_COL not in d.columns:
            out.append(Snapshot(ymd=ymd, path=p, codes=set(), valid=False,
                                reason="missing_code_column"))
            continue
        codes_s = d[CODE_COL].astype(str).str.strip().str.zfill(6)
        codes = set(codes_s[codes_s.str.fullmatch(r"\d{6}")])
        names: Dict[str, str] = {}
        if NAME_COL in d.columns:
            names = dict(zip(codes_s, d[NAME_COL].astype(str)))
        out.append(Snapshot(ymd=ymd, path=p, codes=codes, names=names))
    return out


def reference_size(sizes: Sequence[float],
                   min_ratio: float = MIN_VALID_RATIO,
                   passes: int = TRIM_PASSES) -> float:
    """붕괴 스냅샷을 걸러낼 기준 크기 — 반복 절사 중위.

    롤링 중위는 쓰지 않는다. 수집 장애가 **연속으로** 나면 그 구간의
    롤링 중위 자체가 오염되어 붕괴를 못 잡는다(실제로 2026-02-27~03-06
    6일 연속 장애 구간이 그렇게 빠져나갔다). 전역 중위에서 시작해
    기준 미만을 떨어내고 다시 중위를 잡는 것을 반복하면, 오염 비율이
    과반 미만인 한 정상 모집단의 중위로 수렴한다.
    """
    v = pd.Series(list(sizes), dtype=float)
    v = v[v > 0]
    if v.empty:
        return 0.0
    ref = float(v.median())
    for _ in range(max(0, passes)):
        kept = v[v >= ref * min_ratio]
        if kept.empty:
            break
        nxt = float(kept.median())
        if nxt == ref:
            break
        ref = nxt
    return ref


def mark_degenerate(snaps: Sequence[Snapshot],
                    min_ratio: float = MIN_VALID_RATIO,
                    passes: int = TRIM_PASSES) -> List[Snapshot]:
    """수집이 잘린 스냅샷을 valid=False 로 표시한다."""
    sizes = [float(len(s.codes)) for s in snaps]
    if not sizes:
        return list(snaps)
    ref = reference_size(sizes, min_ratio=min_ratio, passes=passes)
    if ref <= 0:
        return list(snaps)
    dropped = 0
    for s, n in zip(snaps, sizes):
        if not s.valid:
            continue
        if n < ref * min_ratio:
            s.valid = False
            s.reason = f"degenerate:{int(n)}<{min_ratio:.0%}x{int(ref)}"
            dropped += 1
    if dropped > len(snaps) * DEGENERATE_ALARM_RATIO:
        logger.warning("[v71] 스냅샷 %d/%d를 붕괴로 버렸다 — 기준 크기 %d가 타당한지 확인 필요",
                       dropped, len(snaps), int(ref))
    return list(snaps)


def membership(snaps: Sequence[Snapshot]) -> Dict[str, Set[str]]:
    """유효 스냅샷만으로 {ymd: 종목코드 집합} 을 만든다."""
    return {s.ymd: s.codes for s in snaps if s.valid}


def build(data_dir: str = "data",
          confirm: int = CONFIRM_SNAPSHOTS) -> pd.DataFrame:
    """종목별 상장 이력 표를 만든다.

    반환 컬럼:
      종목코드 / 종목명 / first_ymd / last_ymd / n_snapshots /
      status(LISTED|DELISTED|UNCONFIRMED) / gap_max / delisted_ymd
    """
    snaps = mark_degenerate(load_snapshots(data_dir))
    valid = [s for s in snaps if s.valid]
    if not valid:
        logger.warning("[v71] 유효 스냅샷이 없다 — 빈 표 반환")
        return pd.DataFrame(columns=["종목코드", "종목명", "first_ymd", "last_ymd",
                                     "n_snapshots", "status", "gap_max", "delisted_ymd"])
    order = [s.ymd for s in valid]
    pos = {y: i for i, y in enumerate(order)}
    tail = len(order) - 1

    seen: Dict[str, List[int]] = {}
    names: Dict[str, str] = {}
    for s in valid:
        for c in s.codes:
            seen.setdefault(c, []).append(pos[s.ymd])
        for c, nm in s.names.items():
            if c in s.codes and nm and nm != "nan":
                names[c] = nm

    rows = []
    for code, idxs in seen.items():
        idxs.sort()
        first_i, last_i = idxs[0], idxs[-1]
        # 중간 공백의 최대 길이 — 스냅샷 품질 진단용
        gap_max = 0
        for a, b in zip(idxs, idxs[1:]):
            gap_max = max(gap_max, b - a - 1)
        after = tail - last_i          # 마지막 관측 이후 남은 유효 스냅샷 수
        if after == 0:
            status, delisted = STATUS_LISTED, ""
        elif after >= confirm:
            status, delisted = STATUS_DELISTED, order[last_i]
        else:
            # 확인 창이 모자란다 — 소멸이라고 단정하지 않는다
            status, delisted = STATUS_UNCONFIRMED, ""
        rows.append(dict(종목코드=code, 종목명=names.get(code, ""),
                         first_ymd=order[first_i], last_ymd=order[last_i],
                         n_snapshots=len(idxs), status=status,
                         gap_max=gap_max, delisted_ymd=delisted))
    df = pd.DataFrame(rows).sort_values("종목코드").reset_index(drop=True)
    return df


def coverage(data_dir: str = "data",
             hist: Optional[pd.DataFrame] = None) -> dict:
    """유니버스 대비 OHLCV 보유율. 백필이 얼마나 남았는지 보여준다."""
    if hist is None:
        hist = build(data_dir)
    have: Set[str] = set()
    caches = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.parquet")))
    if caches:
        try:
            px = pd.read_parquet(caches[-1], columns=["종목코드"])
            have = set(px["종목코드"].astype(str).str.zfill(6).unique())
        except Exception as e:
            logger.warning("[v71] OHLCV 캐시 읽기 실패 %s: %s", caches[-1], e)
    store = os.path.join(data_dir, "universe_ohlcv")
    if os.path.isdir(store):
        for p in glob.glob(os.path.join(store, "*.parquet")):
            c = _code_of_store_file(p)
            if c:
                have.add(c)
    uni = set(hist["종목코드"]) if not hist.empty else set()
    delisted = set(hist.loc[hist["status"] == STATUS_DELISTED, "종목코드"]) if not hist.empty else set()
    return dict(
        universe=len(uni),
        have_ohlcv=len(uni & have),
        missing=len(uni - have),
        delisted=len(delisted),
        delisted_with_ohlcv=len(delisted & have),
        source_cache=os.path.basename(caches[-1]) if caches else "",
    )


def _code_of_store_file(path: str) -> Optional[str]:
    m = re.search(r"(\d{6})", os.path.basename(path))
    return m.group(1) if m else None


def missing_codes(data_dir: str = "data",
                  hist: Optional[pd.DataFrame] = None) -> List[str]:
    """OHLCV가 아직 없는 유니버스 종목코드 (백필 대상)."""
    if hist is None:
        hist = build(data_dir)
    if hist.empty:
        return []
    have: Set[str] = set()
    caches = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.parquet")))
    if caches:
        try:
            px = pd.read_parquet(caches[-1], columns=["종목코드"])
            have = set(px["종목코드"].astype(str).str.zfill(6).unique())
        except Exception as e:
            logger.warning("[v71] OHLCV 캐시 읽기 실패 %s: %s", caches[-1], e)
    store = os.path.join(data_dir, "universe_ohlcv")
    if os.path.isdir(store):
        for p in glob.glob(os.path.join(store, "*.parquet")):
            c = _code_of_store_file(p)
            if c:
                have.add(c)
    return sorted(set(hist["종목코드"]) - have)


def snapshot_report(data_dir: str = "data") -> pd.DataFrame:
    """스냅샷 품질 표 — 붕괴로 버려진 것이 무엇인지 보이게 한다."""
    snaps = mark_degenerate(load_snapshots(data_dir))
    return pd.DataFrame([dict(ymd=s.ymd, n=len(s.codes), valid=s.valid, reason=s.reason)
                         for s in snaps])


def summary_line(data_dir: str = "data") -> str:
    hist = build(data_dir)
    cov = coverage(data_dir, hist)
    snaps = snapshot_report(data_dir)
    nbad = int((~snaps["valid"]).sum()) if not snaps.empty else 0
    if hist.empty:
        return "[v71] 유니버스 이력 없음 — krx_codes 스냅샷을 찾지 못함"
    n_del = int((hist["status"] == STATUS_DELISTED).sum())
    n_unc = int((hist["status"] == STATUS_UNCONFIRMED).sum())
    return (f"[v71] 유니버스 {cov['universe']:,}종목 "
            f"(소멸 확인 {n_del} · 미확인 {n_unc}) · "
            f"OHLCV 보유 {cov['have_ohlcv']:,} / 결손 {cov['missing']:,} · "
            f"소멸종목 시세 보유 {cov['delisted_with_ohlcv']}/{cov['delisted']} · "
            f"스냅샷 {len(snaps)}개 중 붕괴 {nbad}개 제외")
