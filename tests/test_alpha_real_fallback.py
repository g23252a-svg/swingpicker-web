"""
tests/test_alpha_real_fallback.py
=================================
[v3.9.15e + 3] _calc_kospi_alpha 2단계 폴백 (real / simple / None).

검증 대상:
- trades_df 스키마 정합 (rec_date / net_ret) — v3.9.15d → e 핵심 fix
- daily KOSPI CSV 있을 때 real 알파 작동
- 표본 부족 (<10) 시 simple 폴백
- daily/bench 둘 다 없을 때 (None, None) 반환
- 거짓 알파 안 만드는지 (None > 거짓 값)
"""
import json
import os
import sys
from functools import lru_cache

import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────
# Fixtures — 가짜 data 디렉토리 환경
# ────────────────────────────────────────────────────────────────
@pytest.fixture
def fake_data_dir(tmp_path, monkeypatch):
    """services.benchmarks가 참조할 data 디렉토리 가짜로 세팅."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # services.benchmarks의 DATA_DIR을 가짜 경로로 교체
    monkeypatch.chdir(tmp_path)

    # 모듈 캐시 클리어 (lru_cache 회피 위해 매 테스트마다 fresh import)
    for mod in list(sys.modules.keys()):
        if "benchmarks" in mod or mod.startswith("services"):
            del sys.modules[mod]

    # [v3.9.15e + 6 hotfix] services.benchmarks.DATA_DIR을 가짜 경로로 교체.
    # services.benchmarks의 dirs_to_try는 [DATA_DIR, os.getcwd()+"/data", "data"]
    # 순서로 시도하는데, DATA_DIR은 모듈 import 시점에 절대경로로 박힘
    # (`_HERE = os.path.dirname(os.path.abspath(__file__))` 기반).
    # monkeypatch.chdir만 하면 2번 dirs_to_try만 바뀌고 1번 절대경로는 그대로 →
    # 실제 repo의 data/bench_cache_latest.json이 우선 hit되어 fixture 무시됨.
    # (Linux CI tmp 환경에선 1번이 빈 디렉토리라 우연히 통과, Windows 본인 repo에선 fail)
    import services.benchmarks as _bench
    monkeypatch.setattr(_bench, "DATA_DIR", str(data_dir))

    # [v3.9.15e + 4] tab_backtest는 모듈 캐시 유지 (재import 비용 큼).
    # 대신 services.benchmarks의 lru_cache만 클리어해서 stale 캐시 제거.
    # tab_backtest._calc_kospi_alpha는 함수 내부에서 services를 import하므로
    # services를 다시 가져올 때 새 mtime으로 캐시 갱신됨.
    return data_dir


@pytest.fixture
def clear_benchmark_caches():
    """lru_cache hit으로 stale 결과 받는 것 방지."""
    yield
    # 테스트 종료 시 캐시 클리어 (다음 테스트 격리)
    try:
        from services import benchmarks
        benchmarks._load_bench_cache_cached.cache_clear()
        benchmarks._load_kospi_daily_cached.cache_clear()
    except Exception:
        pass


@pytest.fixture
def trades_df_real_schema():
    """_run_backtest():446-454 실제 스키마와 동일한 trades_df."""
    return pd.DataFrame([
        {
            "rec_date": f"202605{i+1:02d}",
            "code": f"00000{i}",
            "name": f"종목{i}",
            "score": 70 + i,
            "raw_ret": 2.0 + i * 0.3,
            "net_ret": 1.5 + i * 0.3,
            "status": "HOLD_EXIT",
        }
        for i in range(15)
    ])


@pytest.fixture
def kospi_daily_csv(fake_data_dir):
    """ret_5d_%, ret_10d_%, ret_20d_% 포함한 daily KOSPI CSV."""
    dates = pd.date_range("2026-05-01", "2026-05-31", freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%Y%m%d"),
        "close": [2500 + i * 5 for i in range(len(dates))],
    })
    for n in [1, 3, 5, 10, 20, 60]:
        df[f"ret_{n}d_%"] = df["close"].pct_change(n).shift(-n) * 100
    path = fake_data_dir / "kospi_daily.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


@pytest.fixture
def bench_cache_json(fake_data_dir):
    """bench_cache_latest.json (60 키 없음 — 현재 상태 모사)."""
    path = fake_data_dir / "bench_cache_latest.json"
    path.write_text(json.dumps({
        "KOSPI": {"1": -0.5, "3": 1.0, "5": 2.5, "10": 5.0, "20": 8.0},
        "KOSDAQ": {"1": -1.0, "3": 0.5, "5": 1.5, "10": 3.0, "20": 6.0},
    }))
    return path


# ────────────────────────────────────────────────────────────────
# A. trades_df 스키마 정합 — v3.9.15e 핵심 fix
# ────────────────────────────────────────────────────────────────
class TestTradesDfSchema:
    """v3.9.15d의 critical bug — 컬럼명 mismatch로 real 경로가 영원히 0% 작동."""

    def test_trades_df_has_rec_date_not_date(self, trades_df_real_schema):
        """trades_df는 'date'가 아니라 'rec_date' 컬럼을 가져야 한다."""
        assert "rec_date" in trades_df_real_schema.columns
        assert "date" not in trades_df_real_schema.columns

    def test_trades_df_has_net_ret_not_net_pct(self, trades_df_real_schema):
        """trades_df는 'net_pct'가 아니라 'net_ret' 컬럼을 가져야 한다."""
        assert "net_ret" in trades_df_real_schema.columns
        assert "net_pct" not in trades_df_real_schema.columns

    def test_trades_df_schema_completeness(self, trades_df_real_schema):
        """_run_backtest():446-454 스키마 완전성 회귀 방지."""
        expected = {"rec_date", "code", "name", "score", "raw_ret", "net_ret", "status"}
        assert set(trades_df_real_schema.columns) == expected


# ────────────────────────────────────────────────────────────────
# B. real 알파 작동 — daily CSV 있고 표본 충분
# ────────────────────────────────────────────────────────────────
class TestRealAlpha:
    """daily KOSPI CSV 있을 때 real 경로 진입 검증."""

    def test_real_alpha_activates_with_csv_and_sufficient_samples(
        self, fake_data_dir, kospi_daily_csv, bench_cache_json, trades_df_real_schema
    ):
        """daily CSV + 매칭 가능 표본 ≥10건 → ('real', alpha) 반환."""
        from services.benchmarks import load_kospi_daily, get_kospi_return_for_date

        daily = load_kospi_daily()
        assert daily is not None, "daily CSV가 로드되어야 함"

        # 매칭 시뮬 (실제 _calc_kospi_alpha 로직 복제)
        kospi_rets, strat_rets = [], []
        for _, r in trades_df_real_schema.iterrows():
            date_str = str(r["rec_date"]).replace("-", "")
            kr = get_kospi_return_for_date(date_str, 5)
            sr = float(r["net_ret"])
            if kr is not None:
                kospi_rets.append(float(kr))
                strat_rets.append(sr)

        assert len(kospi_rets) >= 10, f"매칭 10건 이상이어야 함, 실제 {len(kospi_rets)}건"


# ────────────────────────────────────────────────────────────────
# C. simple 폴백 — daily 없거나 표본 부족
# ────────────────────────────────────────────────────────────────
class TestSimpleFallback:
    """real 경로 진입 불가 시 simple 알파로 떨어져야 한다."""

    def test_simple_fallback_when_csv_missing(self, fake_data_dir, bench_cache_json):
        """daily CSV 없음 → load_kospi_daily()=None → simple 경로 사용."""
        from services.benchmarks import load_kospi_daily, load_bench_cache, get_kospi_return

        assert load_kospi_daily() is None, "CSV 없으면 None"

        bench = load_bench_cache()
        assert get_kospi_return(bench, 5) == 2.5, "simple 경로의 bench 값 반환"

    def test_simple_alpha_uses_bench_cache(self, fake_data_dir, bench_cache_json):
        """simple 알파 = avg_win*win - avg_loss*loss - kospi_ret."""
        from services.benchmarks import load_bench_cache, get_kospi_return

        bench = load_bench_cache()
        kospi_5d = get_kospi_return(bench, 5)
        assert kospi_5d == 2.5

        # 가짜 백테스트 결과
        win_rate = 0.6
        avg_win = 3.0
        avg_loss = -2.0
        # strat_per_trade = 3.0 * 0.6 + (-2.0) * 0.4 = 1.8 - 0.8 = 1.0
        # alpha = 1.0 - 2.5 = -1.5
        strat_per_trade = avg_win * win_rate + avg_loss * (1.0 - win_rate)
        alpha = strat_per_trade - kospi_5d
        assert alpha == pytest.approx(-1.5, abs=0.01)


# ────────────────────────────────────────────────────────────────
# D. 데이터 완전 부재 → (None, None)
# ────────────────────────────────────────────────────────────────
class TestNoneFallback:
    """daily / bench 둘 다 없으면 (None, None) — UI 'no alpha' 표시."""

    def test_returns_none_when_no_data(self, fake_data_dir):
        """data 디렉토리 비어있으면 정직하게 None."""
        from services.benchmarks import (
            load_kospi_daily, load_bench_cache, get_kospi_return
        )
        assert load_kospi_daily() is None
        bench = load_bench_cache()
        assert get_kospi_return(bench, 5) is None


# ────────────────────────────────────────────────────────────────
# E. 거짓 알파 방지 — 41~60일 정직 처리
# ────────────────────────────────────────────────────────────────
class TestNoFakeAlpha:
    """없는 데이터를 다른 키로 끌어쓰지 않는다."""

    def test_50d_hold_with_no_60_key_returns_none(self, fake_data_dir, bench_cache_json):
        """bench에 60 키 없을 때 50일 보유 → None (20 키로 fallback 금지)."""
        from services.benchmarks import load_bench_cache, get_kospi_return
        bench = load_bench_cache()
        assert "60" not in bench["KOSPI"] and 60 not in bench["KOSPI"]
        assert get_kospi_return(bench, 50) is None
        assert get_kospi_return(bench, 50) != bench["KOSPI"].get(20)

    def test_1d_hold_with_no_ret_1d_col_returns_none(self, fake_data_dir):
        """daily CSV에 ret_1d_% 컬럼 없으면 1일 보유 → None."""
        # ret_5d_%만 있는 CSV
        df = pd.DataFrame({
            "date": ["20260501", "20260502", "20260503"],
            "close": [2500, 2510, 2520],
            "ret_5d_%": [1.0, 0.99, 0.97],
        })
        df.to_csv(fake_data_dir / "kospi_daily.csv", index=False)

        from services.benchmarks import get_kospi_return_for_date
        # 1일 보유는 ret_1d_% 필요한데 컬럼 없으므로 None
        assert get_kospi_return_for_date("20260501", 1) is None
        # 5일 보유는 정상 값
        assert get_kospi_return_for_date("20260501", 5) == pytest.approx(1.0)


# ────────────────────────────────────────────────────────────────
# F. End-to-end — 실제 _calc_kospi_alpha() 함수 호출
# ────────────────────────────────────────────────────────────────
# [v3.9.15e + 5] importorskip를 fixture로 이동.
# 이전 v3.9.15e + 4에서는 모듈 top-level에서 호출 → import 실패 시 파일
# 전체가 skip되어 services.benchmarks 단위 테스트까지 동시 skip되는 부작용.
# fixture로 이동 후엔 e2e 클래스만 skip되고, A~E 클래스는 정상 실행됨.
@pytest.fixture
def calc_kospi_alpha():
    """tab_backtest._calc_kospi_alpha를 lazy import.

    nicegui/plotly 등 의존성이 없는 환경에서는 이 fixture만 skip되고
    services.benchmarks 단위 테스트 (TestTradesDfSchema 등)는 정상 실행.
    """
    tab_backtest = pytest.importorskip(
        "components.tab_backtest",
        reason="tab_backtest 모듈 import 불가 (nicegui 미설치 등)",
        exc_type=ImportError,  # pytest 9.1 deprecation 대응 — ImportError 명시
    )
    return tab_backtest._calc_kospi_alpha


class TestCalcKospiAlphaE2E:
    """v3.9.15e + 4 — 실제 _calc_kospi_alpha 함수를 직접 호출하는 end-to-end.

    이전 테스트들은 services.benchmarks와 로직을 검증했지만, 본래 버그가
    있었던 tab_backtest._calc_kospi_alpha 자체는 직접 안 때림.
    이 클래스는 진짜 함수 호출 → 반환 튜플 (alpha, mode) 검증.

    회귀 보호 시나리오:
    - 누가 다시 "date"/"net_pct" 컬럼명으로 되돌리면 → real 매칭 0건 → simple
      → test_real_alpha_real_mode 실패로 즉시 감지
    - bench cache 없는데 가짜 매핑으로 폴백하면 → test_none_when_no_data 실패
    """

    def test_real_mode_when_csv_and_samples_sufficient(
        self,
        fake_data_dir,
        kospi_daily_csv,
        bench_cache_json,
        trades_df_real_schema,
        clear_benchmark_caches,
        calc_kospi_alpha,
    ):
        """daily CSV + 매칭 10건+ → alpha_mode == "real"."""
        result = {
            "trades_df": trades_df_real_schema,  # 15건, rec_date 컬럼
            "win_rate": 60.0,
            "avg_win": 3.0,
            "avg_loss": -2.0,
        }
        cfg = {"hold_days": 5}
        alpha, mode = calc_kospi_alpha(result, cfg)

        assert mode == "real", (
            f"daily CSV + 15건 표본인데 mode={mode}. "
            f"trades_df 컬럼명 정합 실패 가능 (date/net_pct로 회귀?)"
        )
        assert alpha is not None
        assert isinstance(alpha, float)

    def test_simple_mode_when_no_csv(
        self,
        fake_data_dir,
        bench_cache_json,
        trades_df_real_schema,
        clear_benchmark_caches,
        calc_kospi_alpha,
    ):
        """daily CSV 없음 + bench cache 있음 → alpha_mode == "simple"."""
        result = {
            "trades_df": trades_df_real_schema,
            "win_rate": 60.0,
            "avg_win": 3.0,
            "avg_loss": -2.0,
        }
        cfg = {"hold_days": 5}
        alpha, mode = calc_kospi_alpha(result, cfg)

        assert mode == "simple"
        assert alpha is not None

    def test_simple_mode_when_samples_short(
        self,
        fake_data_dir,
        kospi_daily_csv,
        bench_cache_json,
        clear_benchmark_caches,
        calc_kospi_alpha,
    ):
        """daily CSV 있지만 매칭 표본 <10건 → simple 폴백."""
        # trades_df 5건만 (10건 미만)
        trades_short = pd.DataFrame([{
            "rec_date": f"202605{i+1:02d}",
            "code": f"00000{i}",
            "name": f"종목{i}",
            "score": 70,
            "raw_ret": 2.0,
            "net_ret": 1.5,
            "status": "HOLD_EXIT",
        } for i in range(5)])

        result = {
            "trades_df": trades_short,
            "win_rate": 60.0,
            "avg_win": 3.0,
            "avg_loss": -2.0,
        }
        cfg = {"hold_days": 5}
        alpha, mode = calc_kospi_alpha(result, cfg)

        assert mode == "simple", (
            f"표본 5건 (<10건)인데 mode={mode}. "
            f"표본 부족 시 simple 폴백 실패."
        )

    def test_none_when_no_data(
        self, fake_data_dir, clear_benchmark_caches, calc_kospi_alpha
    ):
        """daily / bench 모두 없음 → (None, None)."""
        result = {
            "trades_df": pd.DataFrame(),
            "win_rate": 60.0,
            "avg_win": 3.0,
            "avg_loss": -2.0,
        }
        cfg = {"hold_days": 5}
        alpha, mode = calc_kospi_alpha(result, cfg)

        assert alpha is None
        assert mode is None

    def test_old_schema_falls_back_to_simple(
        self,
        fake_data_dir,
        kospi_daily_csv,
        bench_cache_json,
        clear_benchmark_caches,
        calc_kospi_alpha,
    ):
        """[critical 회귀 가드] trades_df에 옛 "date"/"net_pct" 컬럼만 있으면
        real 매칭 0건 → simple 폴백. real로 잘못 떨어지면 안 됨.

        이게 v3.9.15d의 실제 버그 모양 — daily CSV가 있어도 컬럼명 안 맞으면
        진짜 알파를 못 산출함. simple로 정직하게 떨어지는지 확인.
        """
        # 옛 스키마 (v3.9.15d 버그 재현)
        trades_old = pd.DataFrame([{
            "date": f"202605{i+1:02d}",     # ← 옛 컬럼명
            "code": f"00000{i}",
            "name": f"종목{i}",
            "score": 70,
            "raw_ret": 2.0,
            "net_pct": 1.5,                  # ← 옛 컬럼명
            "status": "HOLD_EXIT",
        } for i in range(15)])

        result = {
            "trades_df": trades_old,
            "win_rate": 60.0,
            "avg_win": 3.0,
            "avg_loss": -2.0,
        }
        cfg = {"hold_days": 5}
        alpha, mode = calc_kospi_alpha(result, cfg)

        # rec_date 없음 → real 경로 진입 못 함 → simple fallback
        assert mode == "simple", (
            f"옛 스키마(date/net_pct)인데 mode={mode}. "
            f"rec_date 컬럼 체크가 누락된 듯 (회귀!)."
        )
        # 알파 자체는 simple로 산출되므로 None 아님
        assert alpha is not None

    def test_60d_hold_with_no_bench_60_key(
        self,
        fake_data_dir,
        bench_cache_json,
        trades_df_real_schema,
        clear_benchmark_caches,
        calc_kospi_alpha,
    ):
        """60일 보유 + bench cache에 60 키 없음 + daily CSV 없음 → (None, None).

        가짜 20일 매핑으로 떨어지면 안 됨.
        """
        result = {
            "trades_df": trades_df_real_schema,
            "win_rate": 60.0,
            "avg_win": 3.0,
            "avg_loss": -2.0,
        }
        cfg = {"hold_days": 60}
        alpha, mode = calc_kospi_alpha(result, cfg)

        assert alpha is None, (
            f"bench 60 키 없는데 alpha={alpha} 산출됨. "
            f"가짜 20일 매핑 회귀 가능성!"
        )
        assert mode is None
