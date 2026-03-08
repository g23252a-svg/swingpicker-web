# -*- coding: utf-8 -*-
"""
news_engine.py — 뉴스 크롤링 + LLM 감성분석 + AI 코멘트
═══════════════════════════════════════════════════
[v15] retry 유틸 → llm_retry_utils.py로 분리
"""
import os
import re
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Retry 유틸 — llm_retry_utils.py에서 re-export (하위 호환 유지)
from llm_retry_utils import (
    _extract_retry_after,
    _is_retryable,
    _llm_call_with_retry,
)

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

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.
{{"score": 0.0, "summary": "한줄요약"}}"""

        # 신규/구 SDK 분기 호출
        if _USE_NEW_GENAI:
            from google.genai import types as _genai_types
            client = _genai_client.Client(api_key=api_key)
            response = _llm_call_with_retry(
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt,
                    config=_genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=1024,
                    ),
                )
            )
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = _llm_call_with_retry(lambda: model.generate_content(prompt))

        text = response.text.strip()

        score = 0.0
        summary = ""

        # 1차: JSON 파싱
        try:
            import json as _json
            _cleaned = re.sub(r'```json\s*|```\s*', '', text).strip()
            data = _json.loads(_cleaned)
            score = max(-1.0, min(1.0, float(data.get("score", 0))))
            summary = str(data.get("summary", ""))
            return (score, summary)
        except (ValueError, KeyError):
            pass

        # 2차: regex 폴백 (잘린 JSON 대응)
        import re as _re
        _sm = _re.search(r'"score"\s*:\s*(-?\d+\.?\d*)', text)
        if _sm:
            try:
                score = max(-1.0, min(1.0, float(_sm.group(1))))
            except ValueError:
                pass
        _rm = _re.search(r'"summary"\s*:\s*"([^"]*)', text)
        if _rm:
            summary = _rm.group(1)

        # 3차: SCORE:/SUMMARY: 레거시 포맷 폴백
        if score == 0.0:
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
                    model="gemini-2.5-flash", contents=prompt
                )
            )
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
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
