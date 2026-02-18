# -*- coding: utf-8 -*-
"""
async_crawler.py — 비동기 네이버 금융 뉴스 크롤러
──────────────────────────────────────────────────
v2.0 개선사항:
  1. 키워드 필터 대폭 확대 (호재→호실적, 악재 키워드 추가)
  2. 다중 페이지 수집 (days 기간 내 뉴스를 최대 max_pages 까지)
  3. Exponential backoff 재시도 (0.5s → 1.0s → 2.0s)
  4. timeout 을 aiohttp.ClientTimeout 으로 명시 제어
"""
import asyncio
import aiohttp
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from typing import List, Dict

logger = logging.getLogger("NewsFetcher")

# ───────────────────── 키워드 사전 ─────────────────────
# 호재/악재 모두 수집하되 분류는 LLM 분석(collector)에 위임
KEYWORDS_POSITIVE = [
    "특징주", "공시", "수주", "계약", "공급계약",
    "증설", "M&A", "인수", "합병", "흑자전환", "흑자",
    "실적", "호실적", "신고가", "상한가", "자사주",
    "무상증자", "배당", "기술수출", "FDA", "임상",
    "테마", "급등", "신사업", "MOU",
]
KEYWORDS_NEGATIVE = [
    "적자", "하한가", "감사의견", "상장폐지", "관리종목",
    "횡령", "배임", "유상증자", "CB발행", "전환사채",
    "공매도", "급락", "하락", "손실", "리콜",
]
ALL_KEYWORDS = KEYWORDS_POSITIVE + KEYWORDS_NEGATIVE


class AsyncNewsFetcher:
    """네이버 금융 종목 뉴스 비동기 수집기"""

    def __init__(self, max_concurrent: int = 10, max_pages: int = 3,
                 max_retries: int = 3, timeout_sec: float = 10.0):
        """
        Parameters
        ----------
        max_concurrent : int
            동시 요청 수 제한 (세마포어)
        max_pages : int
            종목당 최대 크롤링 페이지 수 (1페이지 ≈ 뉴스 20건)
        max_retries : int
            요청 실패 시 최대 재시도 횟수
        timeout_sec : float
            HTTP 요청 타임아웃 (초)
        """
        self.sem = asyncio.Semaphore(max_concurrent)
        self.max_pages = max(1, max_pages)
        self.max_retries = max(1, max_retries)
        self.timeout = aiohttp.ClientTimeout(total=timeout_sec)
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://finance.naver.com/",
        }
        self.kst = timezone(timedelta(hours=9))

    # ───────────────────── HTML 파싱 ─────────────────────

    def _parse_html(self, text: str, days: int) -> tuple:
        """
        HTML 파싱 → (headlines, has_more)

        Returns
        -------
        headlines : list[str]
            키워드 매칭된 뉴스 제목 리스트
        has_more : bool
            cutoff_date 이후 기사가 페이지 끝까지 있으면 True (다음 페이지 필요)
        """
        headlines = []
        has_more = True
        soup = BeautifulSoup(text, "lxml")

        now_kst = datetime.now(self.kst)
        cutoff_date = now_kst - timedelta(days=days)

        rows = soup.select("table.type5 tr")
        if not rows:
            return [], False

        found_any_date = False

        for row in rows:
            title_node = row.select_one("td.title > a")
            date_node = row.select_one("td.date")

            if not (title_node and date_node):
                continue

            subject = title_node.text.strip()
            date_str = date_node.text.strip()

            try:
                fmt = "%Y.%m.%d %H:%M" if len(date_str) > 10 else "%Y.%m.%d"
                a_date = datetime.strptime(date_str, fmt).replace(tzinfo=self.kst)
            except Exception:
                continue

            found_any_date = True

            # cutoff 이전 기사가 나오면 → 더 이상 다음 페이지 불필요
            if a_date < cutoff_date:
                has_more = False
                break

            # 키워드 필터
            if any(k in subject for k in ALL_KEYWORDS):
                headlines.append(subject)

        # 기사 날짜 자체가 하나도 없으면 다음 페이지 불필요
        if not found_any_date:
            has_more = False

        return headlines, has_more

    # ───────────────────── 단일 종목 수집 ─────────────────────

    async def fetch_news(self, session: aiohttp.ClientSession,
                         code: str, days: int = 2) -> Dict[str, List[str]]:
        """종목 코드 하나에 대해 다중 페이지 뉴스 수집"""
        all_headlines = []

        async with self.sem:
            for page in range(1, self.max_pages + 1):
                url = (
                    f"https://finance.naver.com/item/news_news.naver"
                    f"?code={code}&page={page}"
                )
                html_text = await self._fetch_with_backoff(session, url)
                if html_text is None:
                    break

                headlines, has_more = await asyncio.to_thread(
                    self._parse_html, html_text, days
                )
                all_headlines.extend(headlines)

                if not has_more:
                    break

                # 페이지 간 예의 간격
                await asyncio.sleep(0.2)

        # 순서 보존 중복 제거 + 최대 10건
        unique = list(dict.fromkeys(all_headlines))[:10]
        return {code: unique}

    # ───────────────────── Exponential Backoff ─────────────────────

    async def _fetch_with_backoff(self, session: aiohttp.ClientSession,
                                  url: str) -> str | None:
        """
        GET 요청 + exponential backoff 재시도
        성공 시 HTML 텍스트, 실패 시 None 반환
        """
        for attempt in range(self.max_retries):
            try:
                async with session.get(
                    url, headers=self.headers, timeout=self.timeout
                ) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        return content.decode('euc-kr', 'replace')

                    # 429 Too Many Requests → 더 오래 대기
                    if resp.status == 429:
                        wait = 2.0 ** (attempt + 1)
                        logger.warning(f"⚠️ 429 Too Many Requests → {wait}s 대기")
                        await asyncio.sleep(wait)
                        continue

                    logger.warning(
                        f"⚠️ {url} → HTTP {resp.status} (시도 {attempt + 1})"
                    )

            except asyncio.TimeoutError:
                logger.warning(
                    f"⏱️ 타임아웃 {url} (시도 {attempt + 1})"
                )
            except Exception as e:
                logger.error(
                    f"❌ 요청 실패 {url} (시도 {attempt + 1}): {e}"
                )

            # Exponential backoff: 0.5s → 1.0s → 2.0s
            await asyncio.sleep(0.5 * (2 ** attempt))

        return None

    # ───────────────────── 일괄 수집 ─────────────────────

    async def fetch_all(self, codes: List[str],
                        days: int = 2) -> Dict[str, List[str]]:
        """
        종목 코드 리스트 전체 뉴스 수집

        Parameters
        ----------
        codes : list[str]
            종목 코드 리스트 (6자리)
        days : int
            최근 며칠치 뉴스를 수집할지
        """
        results = {}
        connector = aiohttp.TCPConnector(limit_per_host=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.fetch_news(session, code, days) for code in codes]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for res in completed:
                if isinstance(res, dict):
                    results.update(res)
                elif isinstance(res, Exception):
                    logger.error(f"❌ gather 예외: {res}")

        return results
