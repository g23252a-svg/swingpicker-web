# -*- coding: utf-8 -*-
"""
sector_utils.py — 업종 맵핑 + 대분류 엔진
──────────────────────────────────────────
collector.py 에서 추출한 업종 관련 함수 전체.

사용법
------
    from sector_utils import build_sector_map, classify_big_sector

    # build_sector_map 은 내부에서 KIND/FDR/fallback/override 를 합침
    sector_map = build_sector_map(out_dir="data")
    
    # classify_big_sector 는 종목명+세부업종 → 대분류 문자열
    big = classify_big_sector("삼성전자", "전기전자")
"""
import io
import os
import logging
from typing import Dict

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

UTF8 = "utf-8-sig"

# 기본 out_dir (호출자가 지정하지 않으면 사용)
_DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ═══════════════════════════════════════════════════
#  1. 하드코딩 Fallback
# ═══════════════════════════════════════════════════

def get_fallback_sector_map() -> Dict[str, str]:
    return {
        "005930": "전기전자", "000660": "전기전자", "373220": "전기전자", "207940": "의약품",
        "005380": "운수장비", "005935": "전기전자", "068270": "의약품", "000270": "운수장비",
        "105560": "금융업", "005490": "철강금속", "035420": "서비스업", "035720": "서비스업",
        "006400": "전기전자", "051910": "화학", "012330": "화학", "028260": "유통업",
        "055550": "금융업", "086790": "금융업", "032830": "금융업", "003550": "화학",
        "015760": "전기가스업", "034020": "기계", "010120": "전기전자", "323410": "서비스업",
        "259960": "서비스업", "011200": "운수창고", "000810": "금융업", "018260": "서비스업",
        "010130": "철강금속", "009150": "전기전자", "033780": "금융업", "017670": "통신업",
        "329180": "운수장비", "096770": "화학", "003490": "운수창고", "030200": "통신업",
        "316140": "금융업", "000100": "의약품", "251270": "서비스업", "024110": "금융업",
        "036570": "서비스업", "086280": "운수창고", "090430": "화학", "010950": "화학",
        "009540": "운수장비", "267260": "전기전자", "042700": "전기전자", "010620": "화학",
        "138040": "금융업", "034730": "서비스업", "241560": "화학", "000150": "기계",
        "298040": "전기전자", "108490": "기계", "466100": "기계", "437730": "운수장비",
        "098460": "기계", "277810": "기계", "352820": "서비스업", "253450": "서비스업",
    }


# ═══════════════════════════════════════════════════
#  2. KIND (KRX) 업종
# ═══════════════════════════════════════════════════

def get_sector_map_krx(out_dir: str = None) -> Dict[str, str]:
    """KIND(상장법인 목록) 기준 업종 맵"""
    out_dir = out_dir or _DEFAULT_OUT_DIR
    _ensure_dir(out_dir)
    cache_path = os.path.join(out_dir, "sector_map_krx.csv")

    # 캐시 시도
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype=str)
            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
            df["업종"] = df["업종"].fillna("기타")
            logger.info(f"KIND 업종 캐시 로드: {len(df)} rows")
            return dict(zip(df["종목코드"], df["업종"]))
        except Exception as e:
            logger.warning(f"KIND 캐시 로드 실패, 재다운로드: {e}")

    # 웹 다운로드
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
    try:
        data = {
            "method": "download", "orderMode": "1", "orderStat": "D",
            "searchType": "13", "fiscalYearEnd": "all", "location": "all",
        }
        resp = requests.post(url, data=data,
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()

        dfs = pd.read_html(io.BytesIO(resp.content), header=0)
        if not dfs:
            return {}

        df = dfs[0]
        df.columns = [str(c).strip() for c in df.columns]

        if "종목코드" not in df.columns or "업종" not in df.columns:
            logger.warning(f"KIND CSV 컬럼 이상: {df.columns.tolist()}")
            return {}

        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        df["업종"] = df["업종"].replace("", np.nan).fillna("기타")

        df_out = df[["종목코드", "업종"]].copy()
        df_out.to_csv(cache_path, index=False, encoding=UTF8)
        logger.info(f"KIND 업종 다운로드 완료: {len(df_out)} rows")
        return dict(zip(df_out["종목코드"], df_out["업종"]))

    except Exception as e:
        logger.error(f"KIND 업종 다운로드 실패: {e}")
        return {}


# ═══════════════════════════════════════════════════
#  3. FDR 업종
# ═══════════════════════════════════════════════════

def get_sector_map_fdr(out_dir: str = None) -> Dict[str, str]:
    """FinanceDataReader 기반 보조 업종 맵"""
    out_dir = out_dir or _DEFAULT_OUT_DIR
    _ensure_dir(out_dir)
    cache_path = os.path.join(out_dir, "sector_map_fdr_v2.csv")

    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype=str)
            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
            df["업종"] = df["업종"].fillna("기타")
            logger.info(f"FDR 업종 캐시(v2) 로드: {len(df)} rows")
            return dict(zip(df["종목코드"], df["업종"]))
        except Exception as e:
            logger.warning(f"FDR 캐시(v2) 로드 실패: {e}")

    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KRX")

        code_col = None
        for c in ("Symbol", "Code", "ISU_CD"):
            if c in df.columns:
                code_col = c
                break
        if code_col is None:
            logger.warning(f"FDR 코드 컬럼 없음: {df.columns.tolist()}")
            return {}

        df[code_col] = df[code_col].astype(str).str.zfill(6)

        sector_col = None
        for c in ("업종", "Sector", "Wics", "Industry"):
            if c in df.columns:
                sector_col = c
                break
        if sector_col is None:
            logger.warning("FDR에 업종 컬럼 없음")
            return {}

        df_out = df[[code_col, sector_col]].rename(
            columns={code_col: "종목코드", sector_col: "업종"}
        )

        bad_vals = {"기술성장 기업부", "우량기업부", "중견기업부", "기타 기업부"}
        df_out["업종"] = (
            df_out["업종"]
            .replace("", np.nan)
            .fillna("기타")
            .apply(lambda x: "기타" if str(x).strip() in bad_vals else x)
        )

        df_out.to_csv(cache_path, index=False, encoding=UTF8)
        logger.info(f"FDR 업종(v2) 생성: {len(df_out)} rows")
        return dict(zip(df_out["종목코드"], df_out["업종"]))

    except Exception as e:
        logger.error(f"FDR 업종 생성 실패: {e}")
        return {}


# ═══════════════════════════════════════════════════
#  4. Override (사용자 수동)
# ═══════════════════════════════════════════════════

def load_sector_override(out_dir: str = None) -> Dict[str, str]:
    out_dir = out_dir or _DEFAULT_OUT_DIR
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "sector_override.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
        if "종목코드" not in df.columns or "업종" not in df.columns:
            return {}
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        df["업종"] = df["업종"].fillna("기타")
        return dict(zip(df["종목코드"], df["업종"]))
    except Exception as e:
        logger.warning(f"업종 Override 로드 실패: {e}")
        return {}


# ═══════════════════════════════════════════════════
#  5. 통합 빌더
# ═══════════════════════════════════════════════════

def build_sector_map(out_dir: str = None) -> Dict[str, str]:
    """
    KIND → FDR → fallback → override 우선순위로 업종 맵 합산.
    """
    kind_map = get_sector_map_krx(out_dir)
    fdr_map = get_sector_map_fdr(out_dir)
    fallback = get_fallback_sector_map()
    override = load_sector_override(out_dir)

    sector_map: Dict[str, str] = {}

    # 1순위: KIND
    sector_map.update(kind_map)

    # 2순위: FDR (KIND 가 없거나 '기타' 일 때만)
    for code, sec in fdr_map.items():
        cur = sector_map.get(code)
        if cur is None or str(cur).strip() in ("", "기타"):
            sector_map[code] = sec

    # 3순위: fallback
    for code, sec in fallback.items():
        sector_map.setdefault(code, sec)

    # 4순위: 수동 override (최상위)
    sector_map.update(override)

    logger.info(f"최종 업종 맵: {len(sector_map)}개")
    return sector_map


# ═══════════════════════════════════════════════════
#  6. 대분류 분류기
# ═══════════════════════════════════════════════════

def classify_big_sector(name: str, detailed: str) -> str:
    """KRX 세부업종 + 종목명 → 대분류 업종 문자열"""
    t = (detailed or "").strip()

    # KRX 구형 업종명 fallback
    _old_map = {
        "전기전자": "IT/전기전자",
        "의약품": "바이오·의약품",
        "운수장비": "자동차·모빌리티",
        "철강금속": "철강·금속",
        "화학": "화학·소재",
        "금융업": "금융",
        "서비스업": "서비스 기타",
        "유통업": "유통·소비재",
        "통신업": "IT/전기전자",
        "전기가스업": "인프라·에너지",
        "운수창고": "운송·물류",
    }
    for k, v in _old_map.items():
        if k in t:
            return v

    # 2차전지
    if any(k in t for k in ["2차전지", "이차전지", "이차 전지", "전지"]):
        return "2차전지"
    if any(k in name for k in ["에코프로", "엘앤에프", "퓨처엠", "에너지솔루션", "SDI", "에스디아이"]):
        return "2차전지"

    # 반도체
    if "반도체" in t:
        return "반도체"
    if any(k in name for k in ["하이닉스", "DB하이텍", "한미반도체", "티씨케이", "덕산네오룩스"]):
        return "반도체"

    # 인터넷/플랫폼·게임
    if any(k in t for k in ["포털", "인터넷"]) or any(
            k in name for k in ["네이버", "NAVER", "카카오", "크래프톤", "넷마블", "엔씨소프트"]):
        return "인터넷/플랫폼·게임"

    # IT/전기전자
    if any(k in t for k in [
        "전자부품", "전자 제품", "전기장비", "컴퓨터",
        "통신 및 방송 장비", "자료처리", "소프트웨어", "정보 서비스",
    ]):
        return "IT/전기전자"

    # 자동차·모빌리티
    if any(k in t for k in ["자동차", "운수장비", "차량부품"]) or any(
            k in name for k in ["현대차", "기아", "만도", "현대모비스", "HL클라테크", "롯데렌탈"]):
        return "자동차·모빌리티"

    # 조선·기계·설비
    if any(k in t for k in ["조선", "기계", "선박", "보트 건조업", "산업용 장비", "펌프", "밸브", "터빈"]):
        return "조선·기계·설비"

    # 철강·금속
    if any(k in t for k in ["철강", "1차 금속", "비철금속", "금속가공"]):
        return "철강·금속"

    # 화학·소재
    if any(k in t for k in ["화학", "플라스틱 제품", "고무제품", "합성수지", "섬유제품"]):
        return "화학·소재"

    # 바이오·의약품
    if any(k in t for k in ["의약품", "제약", "생명공학", "의료기기"]):
        return "바이오·의약품"
    if any(k in name for k in ["셀트리온", "삼성바이오로직스", "HLB"]):
        return "바이오·의약품"

    # 금융
    if any(k in t for k in ["은행", "증권", "보험", "기타 금융업", "금융 지원 서비스"]):
        return "금융"

    # 건설·부동산
    if any(k in t for k in ["건설", "주택", "부동산", "토목"]):
        return "건설·부동산"

    # 유통·소비재
    if any(k in t for k in ["도소매", "소매업", "유통업", "전자상거래",
                             "음·식료품", "음료", "식품", "의복", "패션", "화장품"]):
        return "유통·소비재"

    # 운송·물류
    if any(k in t for k in ["운수", "물류", "항공운송", "해상운송", "창고업", "택배"]):
        return "운송·물류"

    # 인프라·에너지
    if any(k in t for k in ["전기가스", "수도", "발전", "송전", "에너지 공급"]):
        return "인프라·에너지"
    if "전동기, 발전기 및 전기 변환 · 공급 · 제어 장치 제조업" in t:
        return "인프라·에너지"

    # 미디어·콘텐츠
    if any(k in t for k in ["방송업", "영화", "비디오물", "출판", "광고업"]):
        return "미디어·콘텐츠"

    # 서비스 기타
    if any(k in t for k in ["서비스업", "사업 지원 서비스", "기타 개인 서비스"]):
        return "서비스 기타"

    return "기타"
