# -*- coding: utf-8 -*-
"""
investor_flow.py — 투자자별 매매동향(수급) 데이터 수집 (v20.6.5)
═══════════════════════════════════════════════════════════════
[v20.6.5] collector.py에서 분리 — pipeline_score.py 순환 의존 해소

분리 대상:
  - fetch_investor_net_buying(): KIS 캐시 → pykrx 폴백

하위 호환:
  collector.py에서 재수출
  → 기존 `from collector import fetch_investor_net_buying` 동작 유지
"""

import json
import time
import logging
import pathlib
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# pykrx 가용성
try:
    from pykrx import stock
    PYKRX_OK = True
except ImportError:
    stock = None
    PYKRX_OK = False


# [v86] 마지막 호출의 출처 메타 — run_health가 FLOW를 정직하게 보고하게 한다.
#   source: kis_cache(flow_{ymd}.json) | kis_cache_latest(같은 날짜의 latest) | pykrx | none
#   cache_ymd_mismatch: flow_cache_latest.json이 있었지만 날짜가 달라 쓰지 않았음
_LAST_FETCH_META: Dict[str, object] = {}


def last_fetch_meta() -> Dict[str, object]:
    """[v86] 직전 fetch_investor_net_buying 호출의 출처 메타(복사본)."""
    return dict(_LAST_FETCH_META)


def _cache_ymd(raw: dict) -> str:
    return str(raw.get("ymd") or "").replace("-", "")[:8]


def fetch_investor_net_buying(ymd: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    해당 일자의 외국인, 기관, 개인 순매수금액(원)을 가져옵니다.
    Returns: (외인_맵, 기관_맵, 개인_맵)

    우선순위:
      1. KIS API 캐시 (data/flow_{ymd}.json)
      2. pykrx 실시간 조회
    """
    # ── KIS 캐시 우선 로드 ──
    # [v86] flow_cache_latest.json은 그 안의 ymd가 요청일과 같을 때만 쓴다.
    #   11/02 배치는 flow_20261102.json이 없자 다른 날짜의 latest를 조용히 읽고도
    #   run_health FLOW=OK였다. 날짜가 다르면 경고를 남기고 pykrx 폴백으로 간다.
    ymd = str(ymd)
    meta: Dict[str, object] = {"ymd": ymd, "source": "none", "cache_ymd": None,
                               "cache_ymd_mismatch": False}
    _LAST_FETCH_META.clear()
    _LAST_FETCH_META.update(meta)
    _cp = pathlib.Path(f"data/flow_{ymd}.json")
    _src = "kis_cache"
    if not _cp.exists():
        _cp = pathlib.Path("data/flow_cache_latest.json")
        _src = "kis_cache_latest"
    _raw = None
    if _cp.exists():
        try:
            _raw = json.loads(_cp.read_text(encoding="utf-8"))
        except Exception as _e:
            logger.warning(f"flow 캐시 로드 실패: {_e}")
            _raw = None
    if isinstance(_raw, dict):
        _cy = _cache_ymd(_raw)
        meta["cache_ymd"] = _cy or None
        _usable = True
        if _cy and _cy != ymd:
            # 날짜가 박힌 캐시가 다른 날짜를 담고 있으면 쓰지 않는다
            _usable = False
            logger.warning(f"⚠️ [v86] 수급 캐시 날짜 불일치 — {_cp.name} ymd={_cy} ≠ 요청 {ymd} "
                           f"→ 캐시 미사용, pykrx 폴백")
        elif not _cy and _src == "kis_cache_latest":
            _usable = False
            logger.warning(f"⚠️ [v86] flow_cache_latest.json에 ymd 없음 — 요청 {ymd}와 "
                           f"같은 날짜인지 확인 불가 → 캐시 미사용, pykrx 폴백")
        if not _usable:
            meta["cache_ymd_mismatch"] = True
            _LAST_FETCH_META.update(meta)
        else:
            try:
                _mf = {k: int(v) for k, v in _raw.get("frg", {}).items()}
                _mi = {k: int(v) for k, v in _raw.get("inst", {}).items()}
                _ma = {k: int(v) for k, v in _raw.get("ant", {}).items()}
                logger.info(f"📊 [수급] 캐시 로드({_cp.name}): 외인 {len(_mf)}건 기관 {len(_mi)}건")
                meta["source"] = _src
                _LAST_FETCH_META.update(meta)
                return _mf, _mi, _ma
            except Exception as _e:
                logger.warning(f"flow 캐시 해석 실패: {_e}")

    # ── pykrx 폴백 ──
    if not PYKRX_OK or stock is None:
        return {}, {}, {}

    frg, inst, ant = {}, {}, {}

    try:
        logger.info(f"💰 수급 데이터(투자자별 매매동향) 수집 중... ({ymd})")

        # 1. 외국인
        df_f = stock.get_market_net_purchases_of_equities_by_ticker(ymd, ymd, "ALL", "외국인")
        if df_f is not None and '순매수거래대금' in df_f.columns:
            codes_f = df_f.index.astype(str).str.zfill(6)
            frg.update(dict(zip(codes_f, df_f['순매수거래대금'].astype(int))))
        time.sleep(0.3)

        # 2. 기관
        df_i = stock.get_market_net_purchases_of_equities_by_ticker(ymd, ymd, "ALL", "기관합계")
        if df_i is not None and '순매수거래대금' in df_i.columns:
            codes_i = df_i.index.astype(str).str.zfill(6)
            inst.update(dict(zip(codes_i, df_i['순매수거래대금'].astype(int))))
        time.sleep(0.3)

        # 3. 개인
        df_a = stock.get_market_net_purchases_of_equities_by_ticker(ymd, ymd, "ALL", "개인")
        if df_a is not None and '순매수거래대금' in df_a.columns:
            codes_a = df_a.index.astype(str).str.zfill(6)
            ant.update(dict(zip(codes_a, df_a['순매수거래대금'].astype(int))))

    except Exception as e:
        logger.warning(f"⚠️ 수급 데이터 수집 실패: {e}")

    if frg or inst or ant:
        _LAST_FETCH_META["source"] = "pykrx"
    return frg, inst, ant
