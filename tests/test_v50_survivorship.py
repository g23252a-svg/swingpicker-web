# -*- coding: utf-8 -*-
"""[v50] 알파 학습 패널 생존편향 제거.

버그
  _load_ohlcv_panel이 data/ohlcv_cache_*.parquet 중 **최신 1개만** 읽었다.
  각 캐시는 '그날의 유니버스'만 담으므로, 유니버스에서 밀려난 종목의 가격이
  통째로 사라지고 build_training_panel의 dropna(_fwd)에서 조용히 탈락한다.

실측 (최근 60배치)
  등장 종목 1,026개 중 455개가 최신 캐시에 없음 → 학습 행 14.4% 소실.
  탈락은 무작위가 아니다:
    탈락 종목 5일수익 -1.89% (승률 26.3%)
    생존 종목        -1.22% (승률 33.5%)   일별 페어드 t=+5.50
  즉 모델이 '나빠질 종목'을 학습에서 못 보고 있었다.

수정 효과 (프로덕션 야간 게이트, 동일 OOS 55일)
  학습 행  29,750 → 35,764   학습 종목  577 → 1,086
  IC t      +8.55 → +9.37    AUC   0.6088 → 0.6155
  OOS 행   18,022 → 20,432
  워크포워드 4폴드 별도 측정: IC +0.1236(t=7.45) → +0.1509(t=9.83),
  게이트 엣지 t +3.86 → +4.71, 상위10% 승률 38.2% → 39.3%.
  단 상위10% 실현수익은 -0.24%p(t=-0.77)로 유의한 변화 없음 —
  수익 개선을 주장하지 않는다. 표본 정확성 수정으로 채택한다.

비용: 합집합을 data/ohlcv_union.parquet에 캐싱하고 최신 캐시만 증분 병합.
      최초 구축 16.8초 → 이후 0.6초. 캐시 2.9MB.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alpha_engine as ae


def _cache(tmp: Path, name: str, codes, dates, base=10000.0):
    """ohlcv_cache_*.parquet 모사 — Date 인덱스 + 종목코드/시가/종가."""
    rows = []
    for c in codes:
        for i, d in enumerate(dates):
            rows.append({"Date": d, "종목코드": c,
                         "시가": base + i, "고가": base + i + 5,
                         "저가": base + i - 5, "종가": base + i + 1})
    df = pd.DataFrame(rows).set_index("Date")
    df.to_parquet(tmp / name)


@pytest.fixture
def rotating(tmp_path):
    """종목이 회전하는 2개 캐시 — 구 코드는 뒤쪽 캐시만 봐서 A001을 잃는다."""
    d1 = pd.date_range("2026-01-05", periods=30, freq="B")
    d2 = pd.date_range("2026-01-19", periods=30, freq="B")
    _cache(tmp_path, "ohlcv_cache_20260213.parquet", ["000001", "000002"], d1)
    _cache(tmp_path, "ohlcv_cache_20260227.parquet", ["000002", "000003"], d2)
    return tmp_path


# ── ① 합집합이 회전 종목을 살리는가 ──

def test_union_recovers_rotated_out_stock(rotating):
    close, fwd = ae._load_ohlcv_panel(str(rotating))
    assert close is not None
    cols = set(close.columns)
    assert "000001" in cols, "유니버스에서 밀려난 종목이 여전히 소실됨"
    assert {"000002", "000003"} <= cols
    assert len(cols) == 3


def test_union_covers_both_date_ranges(rotating):
    close, _ = ae._load_ohlcv_panel(str(rotating))
    assert close.index.min() == pd.Timestamp("2026-01-05")
    assert close.index.max() == pd.Timestamp("2026-02-27")


# ── ② 캐싱·증분 ──

def test_union_cache_is_written_and_reused(rotating):
    ae._load_ohlcv_panel(str(rotating))
    fp = rotating / ae._UNION_PATH
    assert fp.exists(), "합집합 캐시 미생성 — 매 배치 전체 재구축"
    before = fp.stat().st_mtime_ns
    close2, _ = ae._load_ohlcv_panel(str(rotating))     # 2차 호출
    assert "000001" in set(close2.columns), "재사용 경로에서 종목 소실"
    assert fp.stat().st_mtime_ns >= before


def test_incremental_merge_picks_up_new_cache(rotating):
    ae._load_ohlcv_panel(str(rotating))                # 합집합 캐시 생성
    d3 = pd.date_range("2026-03-02", periods=10, freq="B")
    _cache(rotating, "ohlcv_cache_20260316.parquet", ["000004"], d3)
    close, _ = ae._load_ohlcv_panel(str(rotating))
    assert "000004" in set(close.columns), "새 캐시가 증분 병합되지 않음"
    assert "000001" in set(close.columns), "증분 병합이 기존 종목을 날림"


def test_union_excludes_latest_alias_file(rotating):
    """ohlcv_cache_latest.parquet은 중복이므로 스캔에서 제외돼야 한다."""
    d = pd.date_range("2026-01-05", periods=5, freq="B")
    _cache(rotating, "ohlcv_cache_latest.parquet", ["999999"], d)
    close, _ = ae._load_ohlcv_panel(str(rotating))
    assert "999999" not in set(close.columns)


# ── ③ 안전성: 실패해도 배치를 깨지 않는다 ──

def test_falls_back_when_union_build_fails(rotating, monkeypatch):
    monkeypatch.setattr(ae, "_build_ohlcv_union",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    close, fwd = ae._load_ohlcv_panel(str(rotating))
    assert close is not None, "합집합 실패가 배치를 깨뜨림"
    # 폴백은 최신 캐시 1개 → 000001 없음이 정상(구 동작)
    assert "000002" in set(close.columns)


def test_corrupt_union_cache_is_rebuilt(rotating):
    (rotating / ae._UNION_PATH).write_bytes(b"not a parquet")
    close, _ = ae._load_ohlcv_panel(str(rotating))
    assert "000001" in set(close.columns), "손상된 캐시에서 복구 실패"


def test_no_cache_files_returns_none(tmp_path):
    close, fwd = ae._load_ohlcv_panel(str(tmp_path))
    assert close is None and fwd is None


def test_unreadable_cache_is_skipped_not_fatal(rotating):
    (rotating / "ohlcv_cache_20260301.parquet").write_bytes(b"garbage")
    close, _ = ae._load_ohlcv_panel(str(rotating))
    assert close is not None, "깨진 캐시 1개가 전체를 중단시킴"


# ── ④ 전방수익 계산이 보존되는가 (v31.8 0가격 방어 포함) ──

def test_forward_return_shape_and_zero_price_guard(rotating):
    close, fwd = ae._load_ohlcv_panel(str(rotating))
    assert fwd.shape == close.shape
    assert np.isfinite(fwd.values[np.isfinite(fwd.values)]).all()
    # 0가격이 섞여도 inf가 새지 않아야 한다
    d = pd.date_range("2026-04-01", periods=20, freq="B")
    rows = [{"Date": x, "종목코드": "000009", "시가": 0.0, "고가": 0.0,
             "저가": 0.0, "종가": 0.0} for x in d]
    pd.DataFrame(rows).set_index("Date").to_parquet(
        rotating / "ohlcv_cache_20260430.parquet")
    _, fwd2 = ae._load_ohlcv_panel(str(rotating))
    assert not np.isinf(fwd2.values[~np.isnan(fwd2.values)]).any()
