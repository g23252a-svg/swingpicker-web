# -*- coding: utf-8 -*-
"""ohlcv_store.py — 일별 OHLCV 캐시의 추가 전용(append-only) 버전 저장소 [v85]

## 왜

`data/ohlcv_cache_YYYYMMDD.parquet` · `quiet_lane_ohlcv_YYYYMMDD.parquet` 는 매일 밤
~1년치 **전체 스냅샷**을 새 파일로 쓴다. 164+30개 파일 3,400만 행 가운데 97.5%가
앞 파일과 같은 값이고, 밤마다 깃이 ~7.4MB(합집합 재작성까지 ~16.9MB)씩 커진다.

같은 정보를 두 조각으로 나눠 담으면 밤마다 수십 KB면 된다.

    v (versions)   그날 파일에서 **처음 보였거나 값이 바뀐** (종목, 날짜) 행만.
                   rev=True 는 이미 있던 키의 값이 바뀐 것 — 액면분할·병합 후 수정주가
                   재작성, 당일 봉(시가 0·거래량 과소)의 다음 날 확정, 거래량 보정.
    m (membership) 그날 파일의 구성: 종목 블록 순서(pos)·종목·[dmin, dmax]·행수(n).
                   원본 arrow 스키마(메타데이터 포함)를 파일 메타데이터에 싣는다.

어떤 날짜 파일이든 '그날까지의 v 를 밤 순서로 접어 얻은 최신 값'에서 m 의 구간을
잘라 pos·날짜 순으로 이으면 **원본과 정확히 같은 프레임**(인덱스·dtype·컬럼 순서·
행 순서·스키마 메타데이터)이 나온다. 실측: main 164/164, lane 30/30 밤 정확 복원.

## 레이아웃 (설계 [0] freeze-and-delta)

    data/ohlcv_store/{main,lane}/v/YYYYMMDD.parquet   zstd · code, date, 값 컬럼…, rev
    data/ohlcv_store/{main,lane}/m/YYYYMMDD.parquet   zstd · pos, code, dmin, dmax, n
    data/ohlcv_store/verify_log.jsonl                 야간 그림자 검증 기록(한 줄 = 한 밤·한 종류)

- 파일 이름은 `ohlcv_cache_*` · `quiet_lane_ohlcv_*` · `recommend_*` 글롭에 절대 걸리지
  않는다(하위 폴더 + 숫자 이름). 최신 파일을 `sorted(glob)[-1]` 로 고르는 소비자가 7곳 넘는다.
- **m 파일이 커밋 마커다.** v 를 먼저 쓰고 m 을 마지막에 쓴다(둘 다 임시 파일 + os.replace).
  m 이 없는 밤의 v 는 읽을 때 무시한다 — 중간에 죽어도 저장소는 일관된다.
- 같은 밤을 다시 수집하면(수동 재실행) 그 밤의 조각을 **교체**한다. 더 오래된 밤은
  기본적으로 건드리지 않는다(`ingest(ymd="all")` 만 그 밤부터 뒤를 다시 계산한다).
- 값은 비트 단위로 비교한다(NaN 끼리는 같음, -0.0 ≠ 0.0). 값이 예전 값으로 돌아가도
  새 버전으로 다시 적는다 — '최신 값'만 보면 되므로 되돌림 판정이 필요 없다.

## 이 단계(v85-2a)에서 하는 것 / 하지 않는 것

- 하는 것: 야간 배치가 날짜 파일을 **그대로 커밋하면서** 저장소에도 같은 밤을 적고(그림자
  이중 기록), 그 밤을 다시 만들어 원본과 대조한 결과를 verify_log.jsonl 에 남긴다.
- 하지 않는 것: 소비자 전환·날짜 파일 커밋 중단·hydrate 워크플로 연결. 전환은 깨끗한
  검증 ≥ 5밤 뒤 별도 PR 이다(docs/OHLCV_STORE_V85.md).

이 모듈은 numpy·pandas·pyarrow 만 쓴다(앱 모듈 의존 없음 — prefetch 워크플로의 최소
의존 환경에서도 import 된다). pyarrow 는 함수 안에서 늦게 import 한다.

CLI:
    python -m services.ohlcv_store ingest  --data-dir data [--ymd all|YYYYMMDD]
    python -m services.ohlcv_store verify  --data-dir data [--ymd all|newest|YYYYMMDD]
    python -m services.ohlcv_store hydrate --data-dir data --target-dir DIR [--ymd ...]
    python -m services.ohlcv_store stats   --data-dir data
    python -m services.ohlcv_store shadow  --data-dir data --log data/ohlcv_store/verify_log.jsonl
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import zlib
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ohlcv_store")

#: 저장소 폴더 이름 (data_dir 아래). 날짜 캐시 글롭에 걸리지 않는 이름이어야 한다.
STORE_DIRNAME = "ohlcv_store"
STORE_VERSION = 1
#: 종류 → 날짜 파일 접두사.
KINDS: Dict[str, str] = {"main": "ohlcv_cache_", "lane": "quiet_lane_ohlcv_"}
CODE_COL = "종목코드"
VERIFY_LOG = "verify_log.jsonl"
#: m 메타데이터 키.
META_KEY = b"ohlcv_store"
SCHEMA_KEY = b"ohlcv_store.source_schema"
#: m 메타데이터에 남기는 개정 종목 목록 상한 (분할이 몰린 밤에도 m 이 커지지 않게).
REV_CODES_CAP = 500

_DATED_RX = {k: re.compile(r"^" + re.escape(p) + r"(\d{8})\.parquet$") for k, p in KINDS.items()}
_PART_RX = re.compile(r"^(\d{8})\.parquet$")
_TMP_TAG = ".ohlcv_store.tmp"

# (종목, 날짜) → int64 키: 종목 id << 34 | (날짜 초 + 2^33). 같은 종목의 키는 날짜 순으로 연속.
_DATE_BITS = 34
_DATE_OFF = 1 << 33
_DATE_MASK = (1 << _DATE_BITS) - 1
_NS = 1_000_000_000


class StoreError(RuntimeError):
    """저장소가 표현할 수 없는 입력(중복 키·비수치 컬럼 등) 또는 손상된 조각."""


# ══════════════════════════════════════════════════════════════
#  경로
# ══════════════════════════════════════════════════════════════
def store_path(data_dir: str, store_dir: Optional[str] = None) -> str:
    return store_dir or os.path.join(data_dir, STORE_DIRNAME)


def _kinds(kind: Optional[str]) -> List[str]:
    if kind in (None, "all"):
        return list(KINDS)
    if kind not in KINDS:
        raise ValueError(f"kind 는 main|lane|all: {kind!r}")
    return [kind]


def _part(store: str, kind: str, sub: str, ymd: str) -> str:
    return os.path.join(store, kind, sub, f"{ymd}.parquet")


def dated_name(kind: str, ymd: str) -> str:
    return f"{KINDS[kind]}{ymd}.parquet"


def dated_files(data_dir: str, kind: str = "main") -> Dict[str, str]:
    """data_dir 의 날짜 파일 {ymd: 경로}. 접두사 + 숫자 8자리 + .parquet 만 (latest 별칭 제외)."""
    rx = _DATED_RX[kind]
    if not os.path.isdir(data_dir):
        return {}
    names = os.listdir(data_dir)
    out = {}
    for n in names:
        m = rx.match(n)
        if m:
            out[m.group(1)] = os.path.join(data_dir, n)
    return dict(sorted(out.items()))


def nights(data_dir: str, kind: str = "main", store_dir: Optional[str] = None) -> List[str]:
    """m 파일(커밋 마커)이 있는 밤, 오름차순."""
    d = os.path.join(store_path(data_dir, store_dir), kind, "m")
    if not os.path.isdir(d):
        return []
    names = os.listdir(d)
    return sorted(m.group(1) for m in map(_PART_RX.match, names) if m)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _tmp_path(path: str) -> str:
    return os.path.join(os.path.dirname(path), f".{os.path.basename(path)}{_TMP_TAG}{os.getpid()}")


def _atomic_write_table(table, path: str, compression: str) -> None:
    """임시 파일 + os.replace — pandas.to_parquet 과 같은 기본 옵션(ARROW:schema 포함)."""
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = _tmp_path(path)
    try:
        pq.write_table(table, tmp, compression=compression)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_write_part(table, path: str, kv: Optional[dict] = None) -> None:
    """저장소 조각 쓰기 (zstd). ARROW:schema 는 싣지 않는다 — 실으면 m 메타데이터(원본
    스키마)가 그 안에 한 번 더 들어가 m 이 두 배가 된다. 메타데이터는 파일 key-value 로."""
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = _tmp_path(path)
    try:
        t = table.replace_schema_metadata(None)
        with pq.ParquetWriter(tmp, t.schema, compression="zstd", store_schema=False) as w:
            w.write_table(t)
            if kv:
                w.add_key_value_metadata(kv)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _clean_tmp(store: str, kind: str) -> None:
    """앞선 실행이 죽으며 남긴 임시 파일 정리 — `git add -f data/ohlcv_store` 에 섞이면 안 된다."""
    for sub in ("v", "m"):
        d = os.path.join(store, kind, sub)
        if not os.path.isdir(d):
            continue
        for n in os.listdir(d):
            if _TMP_TAG in n:
                try:
                    os.remove(os.path.join(d, n))
                except OSError as e:
                    logger.warning("[v85] 임시 파일 정리 실패 %s: %s", n, e)


# ══════════════════════════════════════════════════════════════
#  원본 정규화
# ══════════════════════════════════════════════════════════════
def _dict_codes(col) -> Tuple[np.ndarray, List[str]]:
    """문자열 컬럼 → (행별 사전 인덱스, 사전). 20만 개 파이썬 문자열을 만들지 않는다."""
    import pyarrow as pa
    arr = col.cast(pa.string())
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    de = arr.dictionary_encode()
    return (de.indices.to_numpy(zero_copy_only=False).astype(np.int64),
            de.dictionary.to_pylist())


class _Src:
    """날짜 파일 하나 — 원본 arrow 테이블 + 정규화 배열(행 순서 그대로)."""

    __slots__ = ("table", "schema", "date_col", "cols", "code_idx", "code_dict", "dates", "vals")

    def __init__(self, table):
        import pyarrow as pa
        self.table = table
        self.schema = table.schema
        names = self.schema.names
        if CODE_COL not in names:
            raise StoreError(f"'{CODE_COL}' 컬럼 없음")
        ts = [f.name for f in self.schema if pa.types.is_timestamp(f.type)]
        if len(ts) != 1:
            raise StoreError(f"날짜(timestamp) 컬럼이 정확히 하나여야 한다: {ts}")
        self.date_col = ts[0]
        if self.schema.field(self.date_col).type.tz is not None:
            raise StoreError("시간대가 붙은 날짜 컬럼은 지원하지 않는다")
        self.cols = [n for n in names if n not in (CODE_COL, self.date_col)]
        for c in self.cols:
            t = self.schema.field(c).type
            if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                raise StoreError(f"값 컬럼 '{c}' 이 수치형이 아니다: {t}")
        code = table.column(CODE_COL)
        date = table.column(self.date_col)
        if code.null_count or date.null_count:
            raise StoreError("종목코드/날짜에 결측이 있다")
        self.code_idx, self.code_dict = _dict_codes(code)
        d = date.cast(pa.timestamp("ns")).to_numpy().view(np.int64)
        if len(d) and np.any(d % _NS):
            raise StoreError("날짜에 초 미만 성분이 있다")
        self.dates = d
        n = table.num_rows
        self.vals = np.empty((n, len(self.cols)), dtype=np.float64)
        for j, c in enumerate(self.cols):
            self.vals[:, j] = table.column(c).cast(pa.float64()).to_numpy(zero_copy_only=False)

    @classmethod
    def read(cls, path: str) -> "_Src":
        import pyarrow.parquet as pq
        return cls(pq.read_table(path))


# ══════════════════════════════════════════════════════════════
#  최신 값 상태 — 정렬된 int64 키 + 값 행렬
# ══════════════════════════════════════════════════════════════
class _Latest:
    """(종목, 날짜) → 지금까지 저장된 최신 값. 키는 정렬·유일, 배열은 제자리 수정하지 않는다."""

    def __init__(self):
        self.code_id: Dict[str, int] = {}
        self.code_names: List[str] = []
        self.keys = np.empty(0, dtype=np.int64)
        self.cols: List[str] = []
        self.vals = np.empty((0, 0), dtype=np.float64)

    # ── 키 ──
    def encode(self, code_idx: np.ndarray, code_dict: Sequence[str], dates_ns: np.ndarray) -> np.ndarray:
        ids = np.empty(len(code_dict), dtype=np.int64)
        for i, c in enumerate(code_dict):
            j = self.code_id.get(c)
            if j is None:
                j = len(self.code_names)
                self.code_id[c] = j
                self.code_names.append(c)
            ids[i] = j
        secs = dates_ns // _NS + _DATE_OFF
        if len(secs) and (secs.min() < 0 or secs.max() > _DATE_MASK):
            raise StoreError("날짜 범위 초과")
        return (ids[code_idx] << _DATE_BITS) | secs

    @staticmethod
    def dates_of(keys: np.ndarray) -> np.ndarray:
        return ((keys & _DATE_MASK) - _DATE_OFF) * _NS

    def names_of(self, keys: np.ndarray) -> np.ndarray:
        return np.asarray(self.code_names, dtype=object)[keys >> _DATE_BITS]

    def codes_pa(self, keys: np.ndarray):
        """키 → 종목코드 arrow 배열 (사전 take — 파이썬 문자열을 만들지 않는다)."""
        import pyarrow as pa
        return pa.array(self.code_names, type=pa.string()).take(pa.array(keys >> _DATE_BITS))

    def lookup(self, keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(self.keys, keys)
        if len(self.keys) == 0:
            return idx, np.zeros(len(keys), dtype=bool)
        found = self.keys.take(np.minimum(idx, len(self.keys) - 1)) == keys
        return idx, found

    def col_matrix(self, cols: Sequence[str], rows: Optional[np.ndarray] = None) -> np.ndarray:
        """요청 컬럼 순서의 값 행렬 (없는 컬럼은 NaN). rows 를 주면 그 행만."""
        n = len(self.keys) if rows is None else len(rows)
        out = np.full((n, len(cols)), np.nan)
        for j, c in enumerate(cols):
            if c in self.cols:
                src = self.vals[:, self.cols.index(c)]
                out[:, j] = src if rows is None else src[rows]
        return out

    # ── 갱신 (새 객체를 돌려준다 — 실패한 밤이 상태를 오염시키지 않게) ──
    def applied(self, keys: np.ndarray, cols: Sequence[str], vals: np.ndarray) -> "_Latest":
        new = _Latest()
        new.code_id = self.code_id           # 사전은 단조 증가 — 공유해도 안전
        new.code_names = self.code_names
        new.cols = list(self.cols) + [c for c in cols if c not in self.cols]
        if len(new.cols) == len(self.cols):
            base = self.vals.copy()
        else:
            base = np.full((len(self.keys), len(new.cols)), np.nan)
            base[:, :len(self.cols)] = self.vals
        if len(keys) > 1:
            sk = np.sort(keys)
            if bool(np.any(sk[1:] == sk[:-1])):
                raise StoreError("한 조각 안에 중복 (종목, 날짜)")
        idx, found = self.lookup(keys)
        cj = [new.cols.index(c) for c in cols]
        if found.any():
            # 개정 — 이 조각에 있는 컬럼만 덮는다(없는 컬럼은 예전 값 유지)
            rows = idx[found]
            for j_src, j_dst in enumerate(cj):
                base[rows, j_dst] = vals[found, j_src]
        add = ~found
        if add.any():
            ak = keys[add]
            o = np.argsort(ak, kind="stable")
            ak = ak[o]
            av = np.full((len(ak), len(new.cols)), np.nan)
            sub = vals[add][o]
            for j_src, j_dst in enumerate(cj):
                av[:, j_dst] = sub[:, j_src]
            ins = np.searchsorted(self.keys, ak)
            new.keys = np.insert(self.keys, ins, ak)
            new.vals = np.insert(base, ins, av, axis=0)
        else:
            new.keys, new.vals = self.keys, base
        return new

    # ── 구간 ──
    def rank_in_code(self, idx: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """키가 그 종목의 알려진 날짜들 중 몇 번째인가."""
        lo = np.searchsorted(self.keys, (keys >> _DATE_BITS) << _DATE_BITS)
        return idx - lo


def _bits(a: np.ndarray) -> np.ndarray:
    """float64 비트 패턴 (NaN 은 하나로 정규화)."""
    a = np.where(np.isnan(a), np.nan, a)
    return np.ascontiguousarray(a, dtype=np.float64).view(np.int64)


# ══════════════════════════════════════════════════════════════
#  조각 읽기/쓰기
# ══════════════════════════════════════════════════════════════
def _read_v(path: str):
    """v 조각 → (code_idx, code_dict, dates_ns, cols, vals)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    cols = [c for c in t.schema.names if c not in ("code", "date", "rev")]
    code_idx, code_dict = _dict_codes(t.column("code"))
    dates = t.column("date").cast(pa.timestamp("ns")).to_numpy().view(np.int64)
    vals = np.empty((t.num_rows, len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        vals[:, j] = t.column(c).cast(pa.float64()).to_numpy(zero_copy_only=False)
    return code_idx, code_dict, dates, cols, vals


def _read_m(path: str):
    """m 조각 → (m DataFrame, meta dict, source arrow schema)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    md = t.schema.metadata or {}
    if META_KEY not in md or SCHEMA_KEY not in md:
        raise StoreError(f"m 메타데이터 없음: {path}")
    meta = json.loads(md[META_KEY].decode("utf-8"))
    schema = pa.ipc.read_schema(pa.py_buffer(zlib.decompress(base64.b64decode(md[SCHEMA_KEY]))))
    m = pd.DataFrame({
        "pos": t.column("pos").to_numpy().astype(np.int64),
        "code": np.asarray(t.column("code").to_pylist(), dtype=object),
        "dmin": t.column("dmin").cast(pa.timestamp("ns")).to_numpy().view(np.int64),
        "dmax": t.column("dmax").cast(pa.timestamp("ns")).to_numpy().view(np.int64),
        "n": t.column("n").to_numpy().astype(np.int64),
    })
    return m, meta, schema


def _v_table(state: "_Latest", keys: np.ndarray, cols: Sequence[str], vals: np.ndarray, rev: np.ndarray):
    """v 조각 테이블 — (code, date) 문자열·날짜 순 정렬(상태의 종목 id 와 무관하게 결정적).
    정수로 정확히 표현되는 컬럼은 int64 로 담는다(-0.0·NaN·소수·2^53 이상이 있으면 float64)."""
    import pyarrow as pa
    names = np.asarray(state.code_names, dtype=object)
    rank = np.empty(len(names), dtype=np.int64)
    rank[np.argsort(names.astype(str), kind="stable")] = np.arange(len(names))
    ids = keys >> _DATE_BITS
    order = np.lexsort((keys & _DATE_MASK, rank[ids])) if len(keys) else np.empty(0, dtype=np.int64)
    k = keys[order]
    arrays = [state.codes_pa(k),
              pa.array(_Latest.dates_of(k).view("datetime64[ns]"), type=pa.timestamp("ns"))]
    out_names = ["code", "date"]
    for j, c in enumerate(cols):
        x = vals[order, j]
        exact_int = (len(x) > 0 and bool(np.all(np.isfinite(x)))
                     and bool(np.all(np.abs(x) < 2.0 ** 53)) and bool(np.all(x == np.round(x)))
                     and not bool(np.any((x == 0) & np.signbit(x))))
        arrays.append(pa.array(x.astype(np.int64)) if exact_int
                      else pa.array(x, type=pa.float64(), from_pandas=False))
        out_names.append(c)
    arrays.append(pa.array(rev[order], type=pa.bool_()))
    out_names.append("rev")
    return pa.Table.from_arrays(arrays, names=out_names)


def _m_table(m: pd.DataFrame, meta: dict, schema):
    import pyarrow as pa
    t = pa.Table.from_arrays([
        pa.array(m["pos"].to_numpy().astype(np.int32), type=pa.int32()),
        pa.array(list(m["code"]), type=pa.string()),
        pa.array(m["dmin"].to_numpy().astype(np.int64).view("datetime64[ns]"), type=pa.timestamp("ns")),
        pa.array(m["dmax"].to_numpy().astype(np.int64).view("datetime64[ns]"), type=pa.timestamp("ns")),
        pa.array(m["n"].to_numpy().astype(np.int32), type=pa.int32()),
    ], names=["pos", "code", "dmin", "dmax", "n"])
    md = {META_KEY: json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8"),
          SCHEMA_KEY: base64.b64encode(zlib.compress(schema.serialize().to_pybytes(), 9))}
    return t.replace_schema_metadata(md)


# ══════════════════════════════════════════════════════════════
#  한 밤 계산 / 복원
# ══════════════════════════════════════════════════════════════
def _membership(state: _Latest, keys: np.ndarray) -> pd.DataFrame:
    """파일 행 순서를 '같은 종목 + 알려진 날짜에서 연속'인 구간들로 자른다.

    정상 파일은 종목마다 구간 하나다(실측 194/194). 중간에 날짜가 빠진 종목은 구간이
    둘 이상으로 나뉠 뿐 여전히 정확히 복원된다.
    """
    if len(keys) == 0:
        return pd.DataFrame({"pos": np.empty(0, np.int64), "code": np.empty(0, object),
                             "dmin": np.empty(0, np.int64), "dmax": np.empty(0, np.int64),
                             "n": np.empty(0, np.int64)})
    idx, found = state.lookup(keys)
    if not found.all():
        raise StoreError("상태에 없는 키 (내부 오류)")
    k = state.rank_in_code(idx, keys)
    cid = keys >> _DATE_BITS
    brk = np.ones(len(keys), dtype=bool)
    brk[1:] = (cid[1:] != cid[:-1]) | (k[1:] != k[:-1] + 1)
    start = np.flatnonzero(brk)
    end = np.append(start[1:], len(keys)) - 1
    dates = _Latest.dates_of(keys)
    return pd.DataFrame({"pos": np.arange(len(start), dtype=np.int64),
                         "code": state.names_of(keys[start]), "dmin": dates[start],
                         "dmax": dates[end], "n": (end - start + 1).astype(np.int64)})


def _select(state: _Latest, m: pd.DataFrame) -> np.ndarray:
    """m 구간들 → 상태 행 인덱스 (pos·날짜 순). 행수가 n 과 다르면 StoreError."""
    if len(m) == 0:
        return np.empty(0, dtype=np.int64)
    m = m.sort_values("pos", kind="stable")
    ids = np.array([state.code_id.get(c, -1) for c in m["code"]], dtype=np.int64)
    if (ids < 0).any():
        bad = m["code"].to_numpy()[ids < 0][:5]
        raise StoreError(f"상태에 없는 종목: {list(bad)}")
    lo_k = (ids << _DATE_BITS) | (m["dmin"].to_numpy() // _NS + _DATE_OFF)
    hi_k = (ids << _DATE_BITS) | (m["dmax"].to_numpy() // _NS + _DATE_OFF)
    lo = np.searchsorted(state.keys, lo_k, side="left")
    hi = np.searchsorted(state.keys, hi_k, side="right")
    cnt = hi - lo
    n = m["n"].to_numpy()
    if not np.array_equal(cnt, n):
        j = int(np.flatnonzero(cnt != n)[0])
        raise StoreError(f"구간 행수 불일치: {m['code'].iloc[j]} 기대 {n[j]} 실제 {cnt[j]}")
    if cnt.sum() == 0:
        return np.empty(0, dtype=np.int64)
    # 구간들을 이어 붙인 인덱스 (벡터화 arange 연결)
    rep_lo = np.repeat(lo, cnt)
    offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return rep_lo + offs


def _materialize(state: _Latest, m: pd.DataFrame, schema):
    """상태 + m + 원본 스키마 → 원본과 같은 arrow 테이블."""
    import pyarrow as pa
    rows = _select(state, m)
    keys = state.keys[rows]
    names = schema.names
    ts = [f.name for f in schema if pa.types.is_timestamp(f.type)]
    if len(ts) != 1 or CODE_COL not in names:
        raise StoreError("원본 스키마가 예상과 다르다")
    date_col = ts[0]
    vcols = [n for n in names if n not in (CODE_COL, date_col)]
    vm = state.col_matrix(vcols, rows)
    arrays = []
    for f in schema:
        if f.name == CODE_COL:
            a = state.codes_pa(keys)
        elif f.name == date_col:
            a = pa.array(_Latest.dates_of(keys).view("datetime64[ns]"), type=pa.timestamp("ns"))
        else:
            a = pa.array(vm[:, vcols.index(f.name)], type=pa.float64(), from_pandas=True)
        arrays.append(a.cast(f.type))       # 안전 캐스트 — 소수가 정수 컬럼으로 잘리면 예외
    return pa.Table.from_arrays(arrays, schema=schema)


def _tables_equal(a, b) -> Tuple[bool, str]:
    """arrow 테이블이 같은가 — 스키마(메타데이터 포함) + 값. 실패 시 pandas 비교로 사유."""
    if not a.schema.equals(b.schema, check_metadata=True):
        return False, f"스키마 다름: {a.schema} vs {b.schema}"[:500]
    if a.num_rows != b.num_rows:
        return False, f"행수 다름: {a.num_rows} vs {b.num_rows}"
    if a.equals(b):
        return True, ""
    try:
        pd.testing.assert_frame_equal(a.to_pandas(), b.to_pandas(), check_exact=True)
        return True, ""                      # NaN 값(널이 아닌) 차이 — 프레임은 같다
    except AssertionError as e:
        logger.debug("[v85] 테이블 불일치: %s", str(e)[:200])
        return False, str(e)[:500]


class _Night:
    __slots__ = ("ymd", "state", "v", "m", "meta", "schema", "n_new", "n_rev", "rows")


def _build_night(state: _Latest, src: _Src, ymd: str, kind: str, sha: str, nbytes: int) -> _Night:
    """밤 하나: v·m 계산 + 상태 갱신 + 복원 검사(원본 테이블과 정확히 같아야 통과)."""
    keys = state.encode(src.code_idx, src.code_dict, src.dates)
    if len(keys) > 1:
        sk = np.sort(keys)
        if bool(np.any(sk[1:] == sk[:-1])):
            raise StoreError("파일 안에 중복 (종목, 날짜)")
    idx, found = state.lookup(keys)
    differ = ~found
    if found.any():
        old = state.col_matrix(src.cols, idx[found])
        neq = np.any(_bits(src.vals[found]) != _bits(old), axis=1)
        differ[np.flatnonzero(found)[neq]] = True
    sel = np.flatnonzero(differ)
    rev = found[sel]
    new_state = state.applied(keys[sel], src.cols, src.vals[sel])
    m = _membership(new_state, keys)
    rebuilt = _materialize(new_state, m, src.schema)
    ok, why = _tables_equal(rebuilt, src.table)
    if not ok:
        raise StoreError(f"복원 검사 실패: {why}")
    rev_codes = sorted(set(new_state.names_of(keys[sel][rev]).tolist()))
    n = _Night()
    n.ymd, n.state, n.schema = ymd, new_state, src.schema
    n.n_new, n.n_rev, n.rows = int(len(sel) - rev.sum()), int(rev.sum()), int(src.table.num_rows)
    n.v = _v_table(new_state, keys[sel], src.cols, src.vals[sel], rev)
    n.meta = {
        "store_version": STORE_VERSION, "kind": kind, "ymd": ymd,
        "rows": n.rows, "codes": int(len(src.code_dict)), "segments": int(len(m)),
        "n_new": n.n_new, "n_rev": n.n_rev,
        "rev_codes": rev_codes[:REV_CODES_CAP], "rev_codes_total": len(rev_codes),
        "date_min": (pd.Timestamp(int(src.dates.min())).strftime("%Y-%m-%d") if n.rows else None),
        "date_max": (pd.Timestamp(int(src.dates.max())).strftime("%Y-%m-%d") if n.rows else None),
        "source_sha256": sha, "source_bytes": int(nbytes),
        "value_cols": list(src.cols), "date_col": src.date_col,
    }
    n.m = m
    return n


def _fold_parts(store: str, kind: str, upto: Optional[str] = None,
                before: Optional[str] = None) -> _Latest:
    """커밋된 밤(m 있음)의 v 조각을 밤 순서로 접은 최신 값 상태."""
    st = _Latest()
    for ymd in nights("", kind, store_dir=store):
        if upto is not None and ymd > upto:
            break
        if before is not None and ymd >= before:
            break
        ci, cd, dates, cols, vals = _read_v(_part(store, kind, "v", ymd))
        st = st.applied(st.encode(ci, cd, dates), cols, vals)
    return st


def _iter_states(store: str, kind: str, wanted: Optional[Iterable[str]] = None
                 ) -> Iterator[Tuple[str, _Latest]]:
    """밤 순서로 상태를 한 번만 접으며 원하는 밤의 상태를 내준다."""
    ns = nights("", kind, store_dir=store)
    want = set(ns if wanted is None else wanted)
    last = max(want) if want else None
    st = _Latest()
    for ymd in ns:
        if last is not None and ymd > last:
            break
        ci, cd, dates, cols, vals = _read_v(_part(store, kind, "v", ymd))
        st = st.applied(st.encode(ci, cd, dates), cols, vals)
        if ymd in want:
            yield ymd, st


# ══════════════════════════════════════════════════════════════
#  공개 API
# ══════════════════════════════════════════════════════════════
def _night_meta(store: str, kind: str, ymd: str) -> Optional[dict]:
    p = _part(store, kind, "m", ymd)
    if not os.path.exists(p):
        return None
    try:
        return _read_m(p)[1]
    except Exception as e:
        logger.warning("[v85] m 메타 읽기 실패 %s: %s", p, e)
        return None


def _same_content(store: str, kind: str, ymd: str, path: str, state: Optional[_Latest] = None) -> bool:
    """저장된 밤 = 디스크의 날짜 파일인가 (sha 일치면 바로 True, 아니면 내용 비교)."""
    meta = _night_meta(store, kind, ymd)
    if meta is None:
        return False
    if meta.get("source_sha256") == _sha256(path):
        return True
    import pyarrow.parquet as pq
    st = state if state is not None else _fold_parts(store, kind, upto=ymd)
    m, _, schema = _read_m(_part(store, kind, "m", ymd))
    try:
        return _tables_equal(_materialize(st, m, schema), pq.read_table(path))[0]
    except Exception as e:
        logger.warning("[v85] %s %s 내용 비교 실패 — 바뀐 것으로 본다: %s", kind, ymd, e)
        return False


def _retire(store: str, kind: str, ymds: Sequence[str]) -> None:
    """밤 조각 제거 — m(마커)을 새 밤부터 먼저 지우고 v 를 지운다."""
    for ymd in sorted(ymds, reverse=True):
        p = _part(store, kind, "m", ymd)
        if os.path.exists(p):
            os.remove(p)
    for ymd in ymds:
        p = _part(store, kind, "v", ymd)
        if os.path.exists(p):
            os.remove(p)


def _ingest_kind(data_dir: str, store: str, kind: str, ymd: Optional[str]) -> dict:
    t0 = time.time()
    files = dated_files(data_dir, kind)
    stored = nights("", kind, store_dir=store)
    newest = stored[-1] if stored else None
    res = {"kind": kind, "ingested": [], "replaced": [], "unchanged": [], "skipped_older": [],
           "refused": [], "failed": [], "n_new": 0, "n_rev": 0, "nights": {}}
    _clean_tmp(store, kind)
    full = ymd == "all"
    if ymd in (None, "all"):
        scope = list(files)
    else:
        scope = [n for n in files if n <= ymd]
        if ymd not in files:
            res["missing_file"] = ymd
    stored_set = set(stored)
    first: Optional[str] = None                     # 다시 계산을 시작할 밤
    if newest is not None and newest in scope:
        if _same_content(store, kind, newest, files[newest]):
            res["unchanged"].append(newest)
        else:
            first = newest                          # 같은 밤 재수집 → 교체
    older = [n for n in scope if newest is not None and n < newest]
    if full and older:
        dirty = [n for n in older if n not in stored_set]
        for n, s in _iter_states(store, kind, [x for x in older if x in stored_set]):
            if not _same_content(store, kind, n, files[n], state=s):
                dirty.append(n)
        if dirty:
            cand = min(dirty)
            need = [n for n in stored if n >= cand and n not in files]
            if need:
                res["refused"].append({"ymd": cand, "reason": f"뒤 밤의 날짜 파일 없음: {need[:5]}"})
            else:
                first = cand if first is None else min(first, cand)
    elif older:
        res["skipped_older"] = [n for n in older if n not in stored_set]
    todo = {n for n in scope if newest is None or n > newest}
    if first is not None:
        todo |= {n for n in stored if n >= first} | {n for n in scope if n >= first}
    todo = sorted(todo)
    if not todo:
        res["seconds"] = round(time.time() - t0, 2)
        return res
    start = todo[0]
    replaced = [n for n in stored if n >= start]
    _retire(store, kind, replaced)
    state = _fold_parts(store, kind, before=start)
    for n in todo:
        path = files[n]
        try:
            src = _Src.read(path)
            night = _build_night(state, src, n, kind, _sha256(path), os.path.getsize(path))
            _atomic_write_part(night.v, _part(store, kind, "v", n))
            mt = _m_table(night.m, night.meta, night.schema)
            _atomic_write_part(mt, _part(store, kind, "m", n), mt.schema.metadata)  # 커밋 마커 — 마지막
        except Exception as e:
            vp = _part(store, kind, "v", n)
            if not os.path.exists(_part(store, kind, "m", n)) and os.path.exists(vp):
                os.remove(vp)
            logger.warning("[v85] %s %s 저장 실패 (그 밤은 저장소에 없다): %s", kind, n, e)
            res["failed"].append({"ymd": n, "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        state = night.state
        (res["replaced"] if n in replaced else res["ingested"]).append(n)
        res["n_new"] += night.n_new
        res["n_rev"] += night.n_rev
        res["nights"][n] = {"rows": night.rows, "n_new": night.n_new, "n_rev": night.n_rev}
        logger.info("[v85] %s %s 저장 — %d행 · 신규 %d · 개정 %d", kind, n, night.rows,
                    night.n_new, night.n_rev)
    res["seconds"] = round(time.time() - t0, 2)
    return res


def ingest(data_dir: str, ymd: Optional[str] = None, kind: str = "all",
           store_dir: Optional[str] = None) -> dict:
    """날짜 파일 → 저장소. 멱등.

    ymd=None    가장 최근 저장 밤 이후의 날짜 파일을 순서대로 적는다(빈 저장소면 전부 =
                부트스트랩). 가장 최근 저장 밤의 파일이 바뀌었으면(같은 밤 재수집) 교체한다.
                더 오래된 밤의 누락은 skipped_older 로 보고만 한다.
    ymd="all"   위 + 오래된 밤까지 전부 대조해, 다른 밤이 있으면 그 밤부터 뒤를 다시 계산한다
                (뒤 밤의 날짜 파일이 모두 있어야 한다 — 없으면 refused).
    ymd=YYYYMMDD  그 밤까지만 (그 밤 이후 파일은 보지 않는다).
    """
    if ymd not in (None, "all") and not re.fullmatch(r"\d{8}", str(ymd)):
        raise ValueError(f"ymd 는 None | 'all' | YYYYMMDD: {ymd!r}")
    store = store_path(data_dir, store_dir)
    out = {"store": store, "ymd": ymd}
    for k in _kinds(kind):
        out[k] = _ingest_kind(data_dir, store, k, ymd)
    return out


def snapshot(data_dir: str, ymd: str, kind: str = "main",
             store_dir: Optional[str] = None) -> pd.DataFrame:
    """저장소만으로 복원한 그 밤의 날짜 파일 = pd.read_parquet(원본) 과 정확히 같은 프레임."""
    import pyarrow.parquet as pq
    table = _snapshot_table(store_path(data_dir, store_dir), kind, ymd)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    return pd.read_parquet(buf)


def _snapshot_table(store: str, kind: str, ymd: str, state: Optional[_Latest] = None):
    p = _part(store, kind, "m", ymd)
    if not os.path.exists(p):
        raise StoreError(f"저장소에 없는 밤: {kind} {ymd}")
    m, _, schema = _read_m(p)
    st = state if state is not None else _fold_parts(store, kind, upto=ymd)
    return _materialize(st, m, schema)


def members(data_dir: str, ymd: str, kind: str = "main",
            store_dir: Optional[str] = None) -> pd.DataFrame:
    """그 밤 파일의 구성 (pos, code, dmin, dmax, n) — 시점 기준 유니버스."""
    m, _, _ = _read_m(_part(store_path(data_dir, store_dir), kind, "m", ymd))
    m = m.copy()
    m["dmin"] = pd.to_datetime(m["dmin"])
    m["dmax"] = pd.to_datetime(m["dmax"])
    return m


def _resolve_nights(store: str, kind: str, data_dir: str, ymds) -> List[str]:
    ns = nights("", kind, store_dir=store)
    if ymds in (None, "all"):
        return sorted(set(ns) | set(dated_files(data_dir, kind)))
    if ymds == "newest":
        return ns[-1:]
    if isinstance(ymds, str):
        return [ymds]
    return sorted(set(ymds))


def verify(data_dir: str, ymd="all", kind: str = "all", store_dir: Optional[str] = None,
           deep: bool = False) -> dict:
    """저장소에서 복원한 밤 ↔ data_dir 의 날짜 파일 정확 대조.

    ymd: "all" | "newest" | YYYYMMDD | [YYYYMMDD, …]
    deep=True 는 복원 테이블을 parquet 으로 써서 pd.read_parquet 프레임까지 비교한다
    (소비자가 실제로 읽는 경로).
    """
    store = store_path(data_dir, store_dir)
    out = {"store": store, "ok": True}
    for k in _kinds(kind):
        t0 = time.time()
        files = dated_files(data_dir, k)
        want = _resolve_nights(store, k, data_dir, ymd)
        stored = set(nights("", k, store_dir=store))
        r = {"checked": 0, "exact": 0, "mismatched": [], "missing_file": [], "missing_store": [],
             "rows": {}}
        for n in want:
            if n not in stored and n in files:
                r["missing_store"].append(n)
            elif n not in files and n in stored:
                r["missing_file"].append(n)
        check = [n for n in want if n in stored and n in files]
        try:
            for n, st in _iter_states(store, k, check):
                r["checked"] += 1
                ok, why = _verify_one(store, k, n, files[n], st, deep)
                if ok:
                    r["exact"] += 1
                else:
                    r["mismatched"].append({"ymd": n, "detail": why})
                    logger.warning("[v85] %s %s 복원 불일치: %s", k, n, why)
                meta = _night_meta(store, k, n) or {}
                r["rows"][n] = meta.get("rows")
        except Exception as e:
            logger.warning("[v85] %s 검증 중단: %s", k, e)
            r["error"] = f"{type(e).__name__}: {e}"[:300]
        r["seconds"] = round(time.time() - t0, 2)
        r["ok"] = not (r["mismatched"] or r["missing_store"] or r.get("error"))
        out[k] = r
        out["ok"] = out["ok"] and r["ok"]
    return out


def _verify_one(store: str, kind: str, ymd: str, path: str, state: _Latest,
                deep: bool) -> Tuple[bool, str]:
    import pyarrow.parquet as pq
    try:
        table = _snapshot_table(store, kind, ymd, state=state)
    except Exception as e:
        logger.warning("[v85] %s %s 복원 실패: %s", kind, ymd, e)
        return False, f"복원 실패 {type(e).__name__}: {e}"[:500]
    ok, why = _tables_equal(table, pq.read_table(path))
    if not ok or not deep:
        return ok, why
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(buf), pd.read_parquet(path), check_exact=True)
    except AssertionError as e:
        logger.debug("[v85] %s %s 프레임 불일치: %s", kind, ymd, str(e)[:200])
        return False, f"프레임 다름: {e}"[:500]
    return True, ""


def hydrate(data_dir: str, target_dir: Optional[str] = None, ymds=None, kind: str = "all",
            store_dir: Optional[str] = None, overwrite: bool = False) -> List[str]:
    """저장소 → 날짜 파일. 원본 작성기(pandas.to_parquet = pyarrow, snappy)와 같은 방식으로 쓴다.

    기존 파일은 덮지 않는다(overwrite=True 일 때만). 쓴 경로 목록을 돌려준다.
    v85-2a 에서는 어떤 워크플로에도 연결하지 않는다 — 전환 PR 에서 쓴다.
    """
    store = store_path(data_dir, store_dir)
    target = target_dir or data_dir
    written: List[str] = []
    for k in _kinds(kind):
        ns = nights("", k, store_dir=store)
        want = ns if ymds in (None, "all") else (ns[-1:] if ymds == "newest" else
                                                 sorted(set([ymds] if isinstance(ymds, str) else ymds)))
        missing = [n for n in want if n not in ns]
        if missing:
            raise StoreError(f"저장소에 없는 밤({k}): {missing[:5]}")
        todo = [n for n in want if overwrite or not os.path.exists(os.path.join(target, dated_name(k, n)))]
        for n, st in _iter_states(store, k, todo):
            p = os.path.join(target, dated_name(k, n))
            _atomic_write_table(_snapshot_table(store, k, n, state=st), p, "snappy")
            written.append(p)
    return written


def latest_union(data_dir: str, kind: str = "main", cols: Optional[Sequence[str]] = None,
                 store_dir: Optional[str] = None) -> pd.DataFrame:
    """모든 밤의 최신 값 합집합 (latest-wins) — ohlcv_union_hl 과 같은 의미.

    분할 전후 수정주가가 섞이는 구간(최신 파일에 없는 오래된 키)은 합집합과 똑같이 남는다.
    """
    st = _fold_parts(store_path(data_dir, store_dir), kind)
    cols = list(cols) if cols is not None else list(st.cols)
    vm = st.col_matrix(cols)
    out = pd.DataFrame({CODE_COL: st.names_of(st.keys),
                        "Date": _Latest.dates_of(st.keys).view("datetime64[ns]")})
    for j, c in enumerate(cols):
        out[c] = vm[:, j]
    return out


def stats(data_dir: str, store_dir: Optional[str] = None) -> dict:
    """저장소 크기 vs 날짜 파일 크기."""
    store = store_path(data_dir, store_dir)
    out: dict = {"store": store}
    tot_store = tot_dated = 0
    for k in KINDS:
        ns = nights("", k, store_dir=store)
        vb = sum(os.path.getsize(_part(store, k, "v", n)) for n in ns
                 if os.path.exists(_part(store, k, "v", n)))
        mb = sum(os.path.getsize(_part(store, k, "m", n)) for n in ns)
        files = dated_files(data_dir, k)
        db = sum(os.path.getsize(p) for p in files.values())
        rows = n_rev = 0
        for n in ns:
            meta = _night_meta(store, k, n) or {}
            rows += int(meta.get("n_new", 0)) + int(meta.get("n_rev", 0))
            n_rev += int(meta.get("n_rev", 0))
        out[k] = {"nights": len(ns), "first": ns[0] if ns else None, "last": ns[-1] if ns else None,
                  "v_bytes": vb, "m_bytes": mb, "versions": rows, "revisions": n_rev,
                  "dated_files": len(files), "dated_bytes": db,
                  "ratio": (round((vb + mb) / db, 4) if db else None)}
        tot_store += vb + mb
        tot_dated += db
    out["store_bytes"] = tot_store
    out["dated_bytes"] = tot_dated
    return out


# ══════════════════════════════════════════════════════════════
#  [v85] 휴장일 팬텀 캐시 방지 — collector 저장 지점에서 부른다
# ══════════════════════════════════════════════════════════════
#: [v85] 이 프로세스에서 휴장일로 판정한 trade_ymd. 같은 실행 안의 다음 저장도 막는다 —
#:   pipeline_calibrate 의 CARRY 재수집이 같은 trade_ymd 로 prepare_ohlcv_data 를 다시 부르는데,
#:   1단계 저장을 건너뛰었으니 캐시가 비어 CARRY 종목만 받는다. 그 표본(<30)은 'insufficient'
#:   → 저장 허용이라, 기억하지 않으면 CARRY 종목만 담긴 작은 팬텀 캐시가 생긴다.
_HOLIDAY_YMDS: set = set()


def cache_write_allowed(ohlcv_map, trade_ymd: str) -> Tuple[bool, dict]:
    """오늘자 날짜 캐시를 써도 되는가.

    v84 휴장일 가드(pipeline_finalize.finalize_outputs → market_calendar.is_trading_session)는
    collector 가 `ohlcv_cache_{ymd}` 를 **저장한 뒤에** 돈다. 그래서 휴장일에도 전일 봉을
    복제한 팬텀 캐시가 계속 커밋됐다(10/09). 같은 판정을 저장 전에 한 번 더 한다.

    'holiday' 판정일 때만 False(그리고 그 날짜를 이 프로세스 동안 기억한다). 표본 부족
    (insufficient)·판정 실패는 True — 이 가드가 거래일 캐시를 버리는 쪽으로 틀리면 안 된다
    (v84 규칙과 같다).
    """
    ymd = str(trade_ymd)
    if ymd in _HOLIDAY_YMDS:
        logger.warning("[v85] %s 은 이번 실행에서 이미 휴장일로 판정됨 — 캐시를 쓰지 않는다", ymd)
        return False, {"trade_ymd": ymd, "verdict": "holiday", "memo": True}
    try:
        from services import market_calendar as _MC
        ok, info = _MC.is_trading_session(ohlcv_map or {}, ymd)
    except Exception as e:
        logger.warning("[v85] 휴장일 판정 실패 — 캐시는 저장한다: %s", e)
        return True, {"trade_ymd": ymd, "verdict": "error", "error": str(e)[:200]}
    if not ok and info.get("verdict") == "holiday":
        _HOLIDAY_YMDS.add(ymd)
        logger.warning("[v85] %s 휴장일 판정(오늘 봉 %s, %s종목) — 팬텀 캐시를 쓰지 않는다",
                       ymd, info.get("share_today"), info.get("tickers"))
        return False, info
    return True, info


# ══════════════════════════════════════════════════════════════
#  그림자 검증 (야간 워크플로) + CLI
# ══════════════════════════════════════════════════════════════
def shadow(data_dir: str, ymd: Optional[str] = None, store_dir: Optional[str] = None,
           log_path: Optional[str] = None) -> dict:
    """ingest(증분) → 이번에 적은 밤 + 대상 밤 검증 → verify_log.jsonl 한 줄(종류별).

    대상 밤: ymd 가 주어지면 그 밤, 아니면 종류별 가장 최근 날짜 파일.
    대상 밤은 deep(pd.read_parquet 프레임까지), 부트스트랩·따라잡기로 함께 적힌 밤은
    arrow 수준으로 대조한다.
    """
    store = store_path(data_dir, store_dir)
    out = {"ok": True, "lines": []}
    for k in KINDS:
        t0 = time.time()
        line = {"ts": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%dT%H:%M:%S%z"),
                "kind": k, "ymd": None, "ok": False}
        try:
            files = dated_files(data_dir, k)
            target = ymd or (list(files)[-1] if files else None)
            line["ymd"] = target
            ing = _ingest_kind(data_dir, store, k, None)
            t1 = time.time()
            written = ing["ingested"] + ing["replaced"]
            line.update(ingested=len(ing["ingested"]), replaced=ing["replaced"],
                        failed=ing["failed"], skipped_older=ing["skipped_older"],
                        seconds_ingest=round(t1 - t0, 2))
            if target is None or target not in files:
                line.update(ok=False, mismatch=f"대상 날짜 파일 없음: {target}")
            else:
                tv = verify(data_dir, target, kind=k, store_dir=store_dir, deep=True)[k]
                others = [n for n in written if n != target]
                ov = (verify(data_dir, others, kind=k, store_dir=store_dir)[k] if others else None)
                meta = _night_meta(store, k, target) or {}
                line.update(rows=meta.get("rows"), n_new=meta.get("n_new"), n_rev=meta.get("n_rev"),
                            n_rev_codes=meta.get("rev_codes_total"))
                mism = tv["mismatched"] + (ov["mismatched"] if ov else [])
                miss = tv["missing_store"] + (ov["missing_store"] if ov else [])
                line["ok"] = bool(tv["ok"] and (ov is None or ov["ok"]) and not ing["failed"])
                if others:
                    line["also_verified"] = ov["checked"]
                    line["also_exact"] = ov["exact"]
                if mism or miss or ing["failed"] or tv.get("error"):
                    line["mismatch"] = {"mismatched": mism[:5], "missing_store": miss[:10],
                                        "failed": ing["failed"][:5], "error": tv.get("error")}
            line["seconds"] = round(time.time() - t0, 2)
        except Exception as e:
            logger.warning("[v85] %s 그림자 검증 실패: %s", k, e)
            line.update(ok=False, mismatch=f"{type(e).__name__}: {e}"[:300],
                        seconds=round(time.time() - t0, 2))
        out["lines"].append(line)
        out["ok"] = out["ok"] and bool(line["ok"])
    if log_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                for line in out["lines"]:
                    f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.warning("[v85] 검증 로그 기록 실패 %s: %s", log_path, e)
    out["stats"] = stats(data_dir, store_dir)
    return out


def _mb(b) -> str:
    return f"{(b or 0) / 1e6:.1f}MB"


def _summary_lines(sh: dict) -> List[str]:
    out = []
    for ln in sh["lines"]:
        k = ln["kind"]
        head = f"[v85 ohlcv_store] {k}: 대상 {ln.get('ymd')} "
        if ln.get("ok"):
            body = (f"✅ 정확 복원 {ln.get('rows')}행 · 신규 {ln.get('n_new')} · 개정 {ln.get('n_rev')}"
                    f" · 이번 실행 적재 {ln.get('ingested', 0)}밤")
            if ln.get("also_verified"):
                body += f" (함께 검증 {ln.get('also_exact')}/{ln.get('also_verified')})"
        else:
            body = f"❌ 불일치/실패 — {json.dumps(ln.get('mismatch'), ensure_ascii=False, default=str)[:400]}"
        out.append(head + body + f" · {ln.get('seconds')}s")
    st = sh.get("stats") or {}
    if st:
        out.append(f"[v85 ohlcv_store] 저장소 {_mb(st.get('store_bytes'))} · 날짜 파일 "
                   f"{_mb(st.get('dated_bytes'))} 대비 "
                   f"{(st.get('store_bytes', 0) / st['dated_bytes'] * 100) if st.get('dated_bytes') else 0:.1f}%"
                   f" (main {st['main']['nights']}밤 · lane {st['lane']['nights']}밤)")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m services.ohlcv_store",
                                 description="[v85] OHLCV 버전 저장소 (append-only)")
    ap.add_argument("cmd", choices=["ingest", "verify", "hydrate", "stats", "shadow"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--store-dir", default=None, help="기본: <data-dir>/ohlcv_store")
    ap.add_argument("--kind", default="all", choices=["all", "main", "lane"])
    ap.add_argument("--ymd", default=None, help="all | newest | YYYYMMDD[,YYYYMMDD…]")
    ap.add_argument("--target-dir", default=None, help="hydrate 출력 폴더 (기본 data-dir)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--deep", action="store_true", help="verify: pd.read_parquet 프레임까지 비교")
    ap.add_argument("--log", default=None, help="shadow: 검증 기록 jsonl 경로")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")
    a = ap.parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    ymd = a.ymd
    if ymd and "," in ymd:
        ymd = [x for x in ymd.split(",") if x]
    if a.cmd == "ingest":
        r = ingest(a.data_dir, ymd=ymd, kind=a.kind, store_dir=a.store_dir)
        bad = any(r[k]["failed"] or r[k]["refused"] for k in _kinds(a.kind))
        for k in _kinds(a.kind):
            x = r[k]
            print(f"[v85 ohlcv_store] ingest {k}: 적재 {len(x['ingested'])} · 교체 {len(x['replaced'])}"
                  f" · 신규 {x['n_new']} · 개정 {x['n_rev']} · 실패 {len(x['failed'])}"
                  f" · 거부 {len(x['refused'])} · 누락(과거) {len(x['skipped_older'])} · {x.get('seconds')}s")
    elif a.cmd == "verify":
        r = verify(a.data_dir, ymd=ymd or "all", kind=a.kind, store_dir=a.store_dir, deep=a.deep)
        bad = not r["ok"]
        for k in _kinds(a.kind):
            x = r[k]
            print(f"[v85 ohlcv_store] verify {k}: 정확 {x['exact']}/{x['checked']} · 불일치 "
                  f"{len(x['mismatched'])} · 저장소 누락 {len(x['missing_store'])} · 파일 없음 "
                  f"{len(x['missing_file'])} · {x['seconds']}s")
            for mm in x["mismatched"][:10]:
                print(f"  ✗ {mm['ymd']}: {mm['detail'][:300]}")
    elif a.cmd == "hydrate":
        r = hydrate(a.data_dir, a.target_dir, ymds=ymd, kind=a.kind, store_dir=a.store_dir,
                    overwrite=a.overwrite)
        bad = False
        print(f"[v85 ohlcv_store] hydrate: {len(r)}개 파일 작성")
    elif a.cmd == "stats":
        r = stats(a.data_dir, a.store_dir)
        bad = False
        for k in KINDS:
            x = r[k]
            print(f"[v85 ohlcv_store] {k}: {x['nights']}밤 ({x['first']}~{x['last']}) · v {_mb(x['v_bytes'])}"
                  f" + m {_mb(x['m_bytes'])} · 버전 {x['versions']:,} (개정 {x['revisions']:,}) · "
                  f"날짜 파일 {x['dated_files']}개 {_mb(x['dated_bytes'])}")
    else:
        r = shadow(a.data_dir, ymd=ymd if isinstance(ymd, str) else None,
                   store_dir=a.store_dir, log_path=a.log)
        bad = not r["ok"]
        for s in _summary_lines(r):
            print(s)
        if bad:
            print("::warning title=ohlcv_store 그림자 검증::저장소 복원이 날짜 파일과 다르다 — "
                  "verify_log.jsonl 확인 (날짜 파일은 그대로 커밋된다)")
    if a.json:
        print(json.dumps(r, ensure_ascii=False, default=str, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
