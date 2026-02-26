# -*- coding: utf-8 -*-
"""
async_crawler.py — 비동기 네이버 금융 뉴스 크롤러
──────────────────────────────────────────────────
v3.0 개선사항 (코드 리뷰 반영):
  1. [#1] 세마포어 Docstring 명확화 — "동시 처리 종목 수 제한" (Polite Scraping)
  2. [#2] cutoff_date 자정 기준 정규화 — 캘린더 날짜 기반 필터
  3. [#3] ClientSession에 headers/timeout 기본값 주입 — 하위 호출 간소화
  4. [#4] 429 Backoff 재시도 카운트 분리 + aiohttp.ClientError 명시 처리
v2.0:
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

# ───────────────────── 429 전용 상수 ─────────────────────
_429_MAX_RETRIES = 5        # 429는 별도 카운터로 최대 5회까지 허용
_429_BASE_WAIT   = 2.0      # 429 대기 시작 초 (지수 증가)


class AsyncNewsFetcher:
    """네이버 금융 종목 뉴스 비동기 수집기"""

    def __init__(self, max_concurrent: int = 10, max_pages: int = 3,
                 max_retries: int = 3, timeout_sec: float = 10.0):
        """
        Parameters
        ----------
        max_concurrent : int
            동시 처리 종목 수 제한 (세마포어).
            종목 단위로 락을 걸어 한 종목의 다중 페이지 요청이
            순차 실행되도록 합니다 (Polite Scraping 전략).
            → 네이버 서버에 대한 과도한 동시 요청을 방지하기 위함.
        max_pages : int
            종목당 최대 크롤링 페이지 수 (1페이지 ≈ 뉴스 20건)
        max_retries : int
            네트워크 에러/타임아웃 시 최대 재시도 횟수.
            (429 Too Many Requests는 별도 카운터로 관리)
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

    def _parse_html(self, text: str, cutoff_date: datetime) -> tuple:
        """
        HTML 파싱 → (headlines, has_more)

        Parameters
        ----------
        text : str
            HTML 원문
        cutoff_date : datetime
            이 시각 이전 기사는 무시 (자정 기준 정규화된 값)

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

        # [#2] 자정 기준 정규화: "최근 N일" = 오늘 포함 N일간
        # days=2 → 어제 00:00:00 부터 (오늘+어제 포함)
        now_kst = datetime.now(self.kst)
        cutoff_date = now_kst.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=days - 1)

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
                    self._parse_html, html_text, cutoff_date
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
        GET 요청 + exponential backoff 재시도.

        재시도 정책:
          - 네트워크 에러 / 타임아웃 / 5xx: max_retries 만큼 재시도
          - 429 Too Many Requests: 별도 카운터 (_429_MAX_RETRIES)로 관리,
            일반 재시도 횟수를 소모하지 않음
        """
        retries_left = self.max_retries
        rate_limit_retries = _429_MAX_RETRIES

        while retries_left > 0:
            try:
                # [#3] headers/timeout은 세션 기본값 사용 → url만 전달
                async with session.get(url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        return content.decode('euc-kr', 'replace')

                    # [#4] 429는 재시도 카운트를 깎지 않고 별도 대기
                    if resp.status == 429:
                        if rate_limit_retries <= 0:
                            logger.error(
                                f"🚫 429 한도 초과 {url} — "
                                f"{_429_MAX_RETRIES}회 대기 후에도 차단 지속"
                            )
                            return None

                        # Retry-After 헤더 우선, 없으면 지수 백오프
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = float(retry_after)
                            except ValueError:
                                wait = _429_BASE_WAIT ** (
                                    _429_MAX_RETRIES - rate_limit_retries + 1
                                )
                        else:
                            wait = _429_BASE_WAIT ** (
                                _429_MAX_RETRIES - rate_limit_retries + 1
                            )
                        wait = min(wait, 60.0)  # 최대 60초 캡

                        logger.warning(
                            f"⚠️ 429 Too Many Requests → {wait:.1f}s 대기 "
                            f"(남은 429 재시도: {rate_limit_retries})"
                        )
                        await asyncio.sleep(wait)
                        rate_limit_retries -= 1
                        continue  # retries_left 안 깎음

                    # 4xx (429 제외) — 재시도 무의미
                    if 400 <= resp.status < 500:
                        logger.warning(
                            f"⚠️ {url} → HTTP {resp.status} (클라이언트 에러, 재시도 안 함)"
                        )
                        return None

                    # 5xx — 서버 에러, 재시도
                    logger.warning(
                        f"⚠️ {url} → HTTP {resp.status} "
                        f"(남은 재시도: {retries_left - 1})"
                    )

            except asyncio.TimeoutError:
                logger.warning(
                    f"⏱️ 타임아웃 {url} (남은 재시도: {retries_left - 1})"
                )
            except aiohttp.ClientError as e:
                # [#4] 네트워크 에러 명시 처리 (DNS 실패, 연결 거부 등)
                logger.warning(
                    f"🔌 네트워크 에러 {url} (남은 재시도: {retries_left - 1}): {e}"
                )
            except Exception as e:
                logger.error(
                    f"❌ 예상치 못한 에러 {url} (남은 재시도: {retries_left - 1}): {e}"
                )

            retries_left -= 1

            if retries_left > 0:
                # Exponential backoff: 0.5s → 1.0s → 2.0s → ...
                backoff = 0.5 * (2 ** (self.max_retries - retries_left - 1))
                await asyncio.sleep(backoff)

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
            최근 며칠치 뉴스를 수집할지 (캘린더 날짜 기준, 오늘 포함)
        """
        results = {}
        # [v3.1] limit_per_host를 세마포어(max_concurrent)와 동기화.
        # 모든 요청이 finance.naver.com 단일 호스트이므로,
        # limit_per_host < sem이면 TCP 커넥터가 숨은 병목이 됩니다.
        # 세마포어가 이미 Polite Scraping 역할을 하므로 커넥터는 맞춰줍니다.
        max_conn = self.sem._value  # 세마포어 초기값
        connector = aiohttp.TCPConnector(limit_per_host=max_conn)

        # [#3] 세션 생성 시 headers/timeout 기본값 주입
        async with aiohttp.ClientSession(
            connector=connector,
            headers=self.headers,
            timeout=self.timeout,
        ) as session:
            tasks = [self.fetch_news(session, code, days) for code in codes]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for res in completed:
                if isinstance(res, dict):
                    results.update(res)
                elif isinstance(res, Exception):
                    logger.error(f"❌ gather 예외: {res}")

        return results
