"""P2 개선과제 #9~#12 통합 테스트"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0

def test(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def run():
    global PASS, FAIL
    PASS, FAIL = 0, 0

    print("=" * 60)
    print("🧪 P2 통합 테스트 (#9~#12)")
    print("=" * 60)

    # ═══ #11: Config 중앙화 ═══
    print("\n📐 1. Config 중앙화 (#11)")
    from collector_config import CollectorConfig, DEFAULT_CONFIG

    test("DEFAULT_CONFIG 인스턴스", isinstance(DEFAULT_CONFIG, CollectorConfig))
    test("rsi_low=45", DEFAULT_CONFIG.rsi_low == 45.0)
    test("rsi_high=65", DEFAULT_CONFIG.rsi_high == 65.0)
    test("rsi_overheat=75", DEFAULT_CONFIG.rsi_overheat == 75.0)
    test("pass_ebs=4", DEFAULT_CONFIG.pass_ebs == 4)
    test("min_mcap_eok=1000", DEFAULT_CONFIG.min_mcap_eok == 1000)
    test("out_dir property", DEFAULT_CONFIG.out_dir.endswith("data"))
    test("rsi_range tuple", DEFAULT_CONFIG.rsi_range == (45.0, 65.0))

    # 커스텀 config
    custom = CollectorConfig(rsi_overheat=80.0, pass_ebs=6, cache_format="pickle")
    test("커스텀 config", custom.rsi_overheat == 80.0 and custom.pass_ebs == 6)
    test("커스텀 cache_format", custom.cache_format == "pickle")

    # 가중치 합 검증
    w_sum = (DEFAULT_CONFIG.w_rr + DEFAULT_CONFIG.w_t1 + DEFAULT_CONFIG.w_sl +
             DEFAULT_CONFIG.w_near + DEFAULT_CONFIG.w_mom + DEFAULT_CONFIG.w_liq +
             DEFAULT_CONFIG.w_tec)
    test("가중치 합 ≈ 1.0", abs(w_sum - 1.0) < 0.01, f"sum={w_sum}")

    # ═══ #12: DataSource 추상화 ═══
    print("\n📐 2. DataSource 추상화 (#12)")
    from data_source import KRXDataSource, OHLCVCache, get_data_source

    ds = KRXDataSource()
    test("KRXDataSource 생성", ds is not None)
    test("pykrx 감지", hasattr(ds, '_pykrx_ok'))
    test("fdr 감지", hasattr(ds, '_fdr_ok'))

    # 싱글턴
    ds1 = get_data_source()
    ds2 = get_data_source()
    test("싱글턴 동일 인스턴스", ds1 is ds2)

    # ═══ #10: Parquet 캐시 ═══
    print("\n📐 3. Parquet 캐시 (#10)")
    import tempfile, shutil
    import pandas as pd
    import numpy as np

    tmp = tempfile.mkdtemp()
    try:
        cache = OHLCVCache(tmp, fmt="parquet")
        test("OHLCVCache 생성", cache.fmt == "parquet")

        # 저장
        data = {
            "005930": pd.DataFrame({"Close": [70000, 71000, 72000], "Volume": [100, 200, 300]}),
            "000660": pd.DataFrame({"Close": [150000, 151000], "Volume": [50, 60]}),
        }
        cache.save("20260222", data)
        test("Parquet 저장", os.path.isdir(os.path.join(tmp, "ohlcv_cache_20260222")))

        # 파일 확인
        pq_files = os.listdir(os.path.join(tmp, "ohlcv_cache_20260222"))
        test("Parquet 파일 2개", len(pq_files) == 2, f"found {len(pq_files)}")
        test("005930 캐시파일 존재",
             any("005930" in f for f in pq_files),
             f"files: {pq_files}")

        # 로드
        loaded = cache.load("20260222")
        test("Parquet 로드 성공", len(loaded) == 2)
        test("005930 데이터 일치", loaded["005930"]["Close"].tolist() == [70000, 71000, 72000])
        test("000660 데이터 일치", len(loaded["000660"]) == 2)

        # exists
        test("exists=True", cache.exists("20260222"))
        test("exists=False", not cache.exists("99999999"))

        # pickle fallback 읽기 (명시적 허용)
        import pickle
        pkl_path = os.path.join(tmp, "ohlcv_cache_20260220.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"999999": pd.DataFrame({"x": [1]})}, f)
        cache_legacy = OHLCVCache(tmp, fmt="parquet", allow_legacy_pickle=True)
        loaded_pkl = cache_legacy.load("20260220")
        test("pickle fallback 로드 (allow=True)", len(loaded_pkl) == 1 and "999999" in loaded_pkl)

        # 빈 날짜
        empty = cache.load("20260101")
        test("빈 캐시 → {}", empty == {})

        # pickle 기본 차단
        cache_no_pkl = OHLCVCache(tmp, fmt="parquet", allow_legacy_pickle=False)
        blocked = cache_no_pkl.load("20260220")  # pkl 있지만 차단
        test("pickle 기본 차단 (allow=False)", blocked == {})

        cache_allow = OHLCVCache(tmp, fmt="parquet", allow_legacy_pickle=True)
        allowed = cache_allow.load("20260220")
        test("pickle 허용 (allow=True)", len(allowed) == 1)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ═══ #9: 모듈 분할 import 검증 ═══
    print("\n📐 4. 모듈 분할 (#9)")

    # macro_filter
    from macro_filter import check_macro_env, compute_market_breadth, label_market_temp
    test("macro_filter import", True)
    test("label_market_temp(70)=과열", "과열" in label_market_temp(70))
    test("label_market_temp(30)=침체", "침체" in label_market_temp(30))

    breadth = compute_market_breadth(pd.DataFrame({"ret_1d_%": [1, -1, 2, -0.5, 3]}))
    test("compute_market_breadth", breadth["ALL"] == 60.0, f"got {breadth['ALL']}")

    # news_engine
    from news_engine import fetch_naver_news_headlines, analyze_sentiment_llm, get_naver_theme_tags
    test("news_engine import", True)

    # telegram_sender
    from telegram_sender import send_telegram_auto
    test("telegram_sender import", True)

    # validation
    from validation import (list_snapshot_days, load_close_map, load_price_maps,
                            pick_recommend_file_per_day, run_reality_check)
    test("validation import", True)

    # ═══ 순환 import 없음 확인 ═══
    print("\n📐 5. 순환 import 방어")
    modules_to_check = [
        ("collector_config", ["collector", "data_source", "macro_filter"]),
        ("data_source", ["collector", "macro_filter", "news_engine"]),
        ("macro_filter", ["collector", "news_engine", "telegram_sender"]),
    ]
    for mod_name, forbidden in modules_to_check:
        mod_file = os.path.join(os.path.dirname(__file__), f"{mod_name}.py")
        src = open(mod_file).read()
        for f in forbidden:
            has_import = (f"import {f} " in src or f"import {f}\n" in src or
                     f"from {f} " in src or f"from {f}." in src)
            test(f"{mod_name} → {f} import 없음", not has_import,
                 f"found '{f}' import in {mod_name}")

    # ═══ Config 타입 안전성 ═══
    print("\n📐 6. Config 타입 안전성")
    import dataclasses
    fields = {f.name: f.type for f in dataclasses.fields(CollectorConfig)}
    test("fields > 30개", len(fields) > 30, f"got {len(fields)}")
    test("rsi_overheat: float", fields["rsi_overheat"] == "float" or fields["rsi_overheat"] is float)

    # ═══ 7. 매직넘버 잔존 자동검출 ═══
    print("\n📐 7. 매직넘버 잔존 자동검출 (#11 완결)")
    import re as _re

    # collector.py에서 Config 밖에 하드코딩된 "위험 패턴" 탐지
    collector_src = open(os.path.join(os.path.dirname(__file__), "collector.py")).read()

    # 이미 Config로 이동한 상수가 다시 하드코딩되면 안 됨
    hardcoded_patterns = [
        (r'\bRSI_LOW\s*=\s*\d', "RSI_LOW 재정의"),
        (r'\bRSI_HIGH\s*=\s*\d', "RSI_HIGH 재정의"),
        (r'\bPASS_EBS\s*=\s*\d', "PASS_EBS 재정의"),
        (r'\bMIN_MCAP_EOK\s*=\s*\d', "MIN_MCAP_EOK 재정의"),
        (r'\bBB_PERIOD\s*=\s*\d', "BB_PERIOD 재정의"),
    ]
    for pattern, desc in hardcoded_patterns:
        # _CFG.xxx 형태는 허용, 직접 숫자 할당만 감지
        matches = _re.findall(pattern, collector_src)
        # _CFG로 시작하는 건 제외
        real_hardcoded = [m for m in matches if "_CFG" not in m]
        test(f"하드코딩 없음: {desc}", len(real_hardcoded) == 0,
             f"found {len(real_hardcoded)} occurrences")

    # scoring_engine.py에서도 Config 상수 재정의 없는지
    se_src = open(os.path.join(os.path.dirname(__file__), "scoring_engine.py")).read()
    se_patterns = [
        (r'\bRSI_LOW\s*=\s*\d', "RSI_LOW in scoring_engine"),
        (r'\bMIN_MCAP\s*=\s*\d', "MIN_MCAP in scoring_engine"),
    ]
    for pattern, desc in se_patterns:
        matches = _re.findall(pattern, se_src)
        test(f"scoring_engine 하드코딩 없음: {desc}", len(matches) == 0)

    # ── 결과 ──
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"🏁 결과: {PASS}/{total} 통과 ({FAIL} 실패)")
    if FAIL > 0:
        print("⚠️ 실패 항목이 있습니다!")
        sys.exit(1)
    else:
        print("🏆 ALL PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run()
