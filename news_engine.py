# -*- coding: utf-8 -*-
"""
news_engine.py — 뉴스 크롤링 + LLM 감성분석 + AI 코멘트
═══════════════════════════════════════════════════
[v14] #9 collector.py 분할 — 뉴스/LLM 관련 함수
"""
import os
import re
import logging
from typing import List, Tuple, Optional

import time
import random

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  Retry 유틸 (429/RESOURCE_EXHAUSTED 방어)
# ═══════════════════════════════════════════════════

def _extract_retry_after(exc: Exception) -> Optional[float]:
    """에러 객체에서 Retry-After 힌트 추출 (있으면 우선 사용)"""
    # google.api_core.exceptions 스타일
    if hasattr(exc, 'retry_after'):
        return float(exc.retry_after)
    # HTTP response 헤더 스타일
    if hasattr(exc, 'response') and hasattr(exc.response, 'headers'):
        ra = exc.response.headers.get('Retry-After')
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
    return None


def _is_retryable(exc: Exception) -> bool:
    """재시도 가능한 에러인지 판정 (status code 우선, 문자열 fallback)"""
    # status_code 속성 (google, requests 등)
    status = getattr(exc, 'code', None) or getattr(exc, 'status_code', None)
    if status is not None:
        try:
            code = int(status)
            return code in (429, 503, 529)
        except (ValueError, TypeError):
            pass
    # grpc status
    if hasattr(exc, 'grpc_status_code'):
        grpc_code = str(exc.grpc_status_code)
        if 'RESOURCE_EXHAUSTED' in grpc_code or 'UNAVAILABLE' in grpc_code:
            return True
    # 문자열 fallback (최후 수단)
    msg = str(exc)
    return any(kw in msg for kw in ('429', 'RESOURCE_EXHAUSTED', 'quota', 'rate limit'))


def _llm_call_with_retry(
    call_fn,
    max_retries: int = 3,
    base_delay: float = 2.0,
    cap: float = 30.0,
    total_timeout: float = 60.0,
):
    """
    LLM API 호출 + exponential backoff + jitter + Retry-After.
    - 429/RESOURCE_EXHAUSTED → 재시도
    - 400/500 등 비재시도 에러 → 즉시 raise
    - total_timeout: 전체 대기시간 상한 (초). 초과 시 마지막 에러 raise.
    """
    t_start = time.monotonic()
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except Exception as e:
            last_exc = e
            if not _is_retryable(e):
                raise  # 비재시도 에러 → 즉시 전파

            if attempt >= max_retries:
                logger.warning(f"LLM 재시도 한도 초과 ({max_retries}회): {e}")
                raise

            # 총 대기시간 상한 체크
            elapsed = time.monotonic() - t_start
            remaining = total_timeout - elapsed
            if remaining <= 0:
                logger.warning(f"LLM 총 대기시간 상한 초과 ({total_timeout}s): {e}")
                raise

            # Retry-After 우선, 없으면 exponential backoff + jitter
            retry_after = _extract_retry_after(e)
            if retry_after and retry_after > 0:
                wait = min(retry_after, cap)
            else:
                wait = min(cap, base_delay * (2 ** attempt))
                wait *= (0.5 + random.random())  # jitter: 0.5x ~ 1.5x

            # remaining으로 한번 더 클램프
            wait = min(wait, remaining)

            logger.info(f"LLM 429/재시도 {attempt+1}/{max_retries}: "
                       f"wait={wait:.1f}s, elapsed={elapsed:.1f}s/{total_timeout}s")
            time.sleep(wait)

# LLM import (optional) — 신규/구 SDK 자동 감지
_USE_NEW_GENAI = False
try:
    from google import genai as _genai_client
    _USE_NEW_GENAI = True
    LLM_AVAILABLE = True
except ImportError:
    _genai_client = None
    try:
        import google.generativeai as genai
        LLM_AVAILABLE = True
    except ImportError:
        LLM_AVAILABLE = False


def fetch_naver_news_headlines(code: str, days: int = 2) -> List[str]:
    """네이버 금융 뉴스 헤드라인 크롤링"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    import requests
    headlines = []
    try:
        url = f"https://finance.naver.com/item/news_news.naver?code={code}&page=1"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        rows = soup.select("table.type5 tr")
        for row in rows[:20]:
            a_tag = row.select_one("td.title a")
            if a_tag:
                title = a_tag.get_text(strip=True)
                if title:
                    headlines.append(title)
    except Exception as e:
        logger.debug(f"뉴스 크롤링 실패 {code}: {e}")

    return headlines[:10]


def analyze_sentiment_llm(
    stock_name: str,
    headlines: List[str],
) -> Tuple[float, str]:
    """LLM 기반 뉴스 감성 분석. Returns (score -1~1, summary)"""
    if not LLM_AVAILABLE or not headlines:
        return (0.0, "")

    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return (0.0, "")

        prompt = f"""다음은 '{stock_name}'에 대한 최신 뉴스 헤드라인입니다.
전체적인 투자 심리를 -1.0(매우 부정) ~ +1.0(매우 긍정) 사이 점수로 평가하고,
한 줄 요약을 해주세요.

헤드라인:
{chr(10).join(f'- {h}' for h in headlines[:8])}

응답 형식 (반드시 이 형식만):
SCORE: [숫자]
SUMMARY: [한줄요약]"""

        # 신규/구 SDK 분기 호출
        if _USE_NEW_GENAI:
            client = _genai_client.Client(api_key=api_key)
            response = _llm_call_with_retry(
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash", contents=prompt
                )
            )
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = _llm_call_with_retry(lambda: model.generate_content(prompt))

        text = response.text.strip()

        score = 0.0
        summary = ""
        for line in text.split("\n"):
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip())
                    score = max(-1.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()

        return (score, summary)
    except Exception as e:
        logger.debug(f"LLM 감성분석 실패: {e}")
        return (0.0, "")


def generate_ai_comment(
    stock_name: str,
    row: dict,
    headlines: Optional[List[str]] = None,
) -> str:
    """LLM 기반 AI 종합 코멘트"""
    if not LLM_AVAILABLE:
        return ""

    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return ""

        # 핵심 지표 요약
        rsi = row.get("RSI14", "N/A")
        timing = row.get("TIMING_SCORE", "N/A")
        struct = row.get("STRUCT_SCORE", "N/A")
        route = row.get("ROUTE", "N/A")
        ret5 = row.get("ret_5d_%", "N/A")

        news_text = ""
        if headlines:
            news_text = "\n최근 뉴스: " + " / ".join(headlines[:5])

        prompt = f"""'{stock_name}' 종목에 대해 간단한 투자 코멘트를 작성하세요.

기술 지표: RSI={rsi}, TIMING={timing}, STRUCT={struct}, 상태={route}, 5일수익률={ret5}%
{news_text}

2~3문장으로 핵심만 간결하게. 매수/매도 추천이 아닌 객관적 분석만."""

        if _USE_NEW_GENAI:
            client = _genai_client.Client(api_key=api_key)
            response = _llm_call_with_retry(
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash", contents=prompt
                )
            )
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = _llm_call_with_retry(lambda: model.generate_content(prompt))
        return response.text.strip()[:300]
    except Exception as e:
        logger.debug(f"AI 코멘트 생성 실패: {e}")
        return ""


def get_naver_theme_tags(code: str) -> str:
    """네이버 금융에서 섹터/테마 해시태그 추출"""
    try:
        from bs4 import BeautifulSoup
        import requests
    except ImportError:
        return ""

    try:
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        tags = []
        # 업종 정보
        sector_el = soup.select_one("div.section.trade_compare a")
        if sector_el:
            tags.append(f"#{sector_el.get_text(strip=True)}")

        # 테마 정보
        for a in soup.select("table.tb_type1_b td a"):
            text = a.get_text(strip=True)
            if text and len(text) < 20:
                tags.append(f"#{text}")

        return " ".join(tags[:5])
    except Exception:
        return ""
