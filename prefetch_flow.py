# -*- coding: utf-8 -*-
"""
prefetch_flow.py — 수급 데이터 선수집 + 캐시 저장
══════════════════════════════════════════════════
[v1.0] collector.py 본체와 분리된 수급 전용 수집기

GitHub Actions prefetch_flow job에서 단독 실행.
성공 시 data/flow_cache_latest.json 에 저장.
본 collector는 해당 파일을 읽기만 함.

실행:
    python prefetch_flow.py            # 오늘/직전 거래일 자동
    python prefetch_flow.py 20250307   # 날짜 지정
"""

import sys
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── 출력 디렉토리 ──────────────────────────────────────────────────
OUT_DIR = os.environ.get("LDY_OUT_DIR", "data")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_LATEST = os.path.join(OUT_DIR, "flow_cache_latest.json")
CACHE_STALE_HOURS = 20  # 이 시간 이전 캐시는 STALE 처리


def _last_weekday(d: datetime) -> str:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def fetch_flow(ymd: str, max_retries: int = 3) -> Tuple[Dict, Dict, Dict, str]:
    """
    pykrx로 외인/기관/개인 순매수 수집.
    Returns: (frg, inst, ant, status)
      status: "OK" | "PARTIAL" | "EMPTY" | "FETCH_FAIL" | "NO_PYKRX"
    """
    try:
        from pykrx import stock
    except ImportError:
        log.warning("⚠️ pykrx 미설치")
        return {}, {}, {}, "NO_PYKRX"

    frg, inst, ant = {}, {}, {}
    errors = []

    def _fetch_one(investor_type: str, max_retries: int = max_retries) -> Optional[Dict]:
        for attempt in range(max_retries):
            try:
                df = stock.get_market_net_purchases_of_equities_by_ticker(
                    ymd, ymd, "ALL", investor_type
                )
                if df is None or df.empty:
                    return {}
                if "순매수거래대금" not in df.columns:
                    log.warning(f"⚠️ [{investor_type}] 컬럼 없음: {list(df.columns)}")
                    return {}
                codes = df.index.astype(str).str.zfill(6)
                return dict(zip(codes, df["순매수거래대금"].astype(int)))
            except Exception as e:
                log.warning(f"⚠️ [{investor_type}] 시도 {attempt+1}/{max_retries} 실패: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s
        return None  # 전체 실패

    log.info(f"💰 수급 선수집 시작 (기준일: {ymd})")

    r_frg = _fetch_one("외국인")
    if r_frg is None:
        errors.append("외국인")
    else:
        frg = r_frg
    time.sleep(0.5)

    r_inst = _fetch_one("기관합계")
    if r_inst is None:
        errors.append("기관합계")
    else:
        inst = r_inst
    time.sleep(0.5)

    r_ant = _fetch_one("개인")
    if r_ant is None:
        errors.append("개인")
    else:
        ant = r_ant

    log.info(f"   외인: {len(frg)}건  기관: {len(inst)}건  개인: {len(ant)}건")

    if errors:
        log.warning(f"⚠️ 실패 항목: {errors}")

    major_ok = len(frg) > 0 or len(inst) > 0
    any_ok = major_ok or len(ant) > 0

    if major_ok:
        status = "OK"
    elif any_ok:
        status = "PARTIAL"
    elif not errors:
        status = "EMPTY"
    else:
        status = "FETCH_FAIL"

    return frg, inst, ant, status


def save_cache(frg: Dict, inst: Dict, ant: Dict, ymd: str, status: str) -> str:
    """수급 캐시를 JSON으로 저장"""
    payload = {
        "ymd": ymd,
        "status": status,
        "fetched_at": datetime.now().isoformat(),
        "frg": frg,
        "inst": inst,
        "ant": ant,
        "counts": {"frg": len(frg), "inst": len(inst), "ant": len(ant)},
    }
    # dated 파일
    dated_path = os.path.join(OUT_DIR, f"flow_{ymd}.json")
    with open(dated_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    # latest 파일 (collector가 읽는 파일)
    with open(CACHE_LATEST, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    log.info(f"💾 수급 캐시 저장: {CACHE_LATEST} (status={status})")
    return CACHE_LATEST


def load_cache(ymd: str) -> Optional[Dict]:
    """
    저장된 수급 캐시 로드.
    stale 여부도 함께 반환.
    Returns: dict with keys frg/inst/ant/status/stale, or None
    """
    if not os.path.exists(CACHE_LATEST):
        return None
    try:
        with open(CACHE_LATEST, encoding="utf-8") as f:
            c = json.load(f)
    except Exception as e:
        log.warning(f"⚠️ 수급 캐시 로드 실패: {e}")
        return None

    # stale 판정
    fetched_at = c.get("fetched_at", "")
    stale = True
    if fetched_at:
        try:
            age_hours = (datetime.now() - datetime.fromisoformat(fetched_at)).total_seconds() / 3600
            stale = age_hours > CACHE_STALE_HOURS
        except Exception:
            pass

    # ymd 불일치 → stale
    if c.get("ymd") != ymd:
        stale = True

    c["stale"] = stale
    return c


if __name__ == "__main__":
    ymd = sys.argv[1] if len(sys.argv) > 1 else _last_weekday(datetime.now())
    frg, inst, ant, status = fetch_flow(ymd)
    save_cache(frg, inst, ant, ymd, status)

    print(f"\n결과: {status}")
    print(f"외인 {len(frg)}건 / 기관 {len(inst)}건 / 개인 {len(ant)}건")

    if status == "FETCH_FAIL":
        sys.exit(1)   # Actions에서 실패 표시
    sys.exit(0)
