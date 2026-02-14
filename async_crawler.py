import asyncio
import aiohttp
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from typing import List, Dict

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NewsFetcher")

class AsyncNewsFetcher:
    def __init__(self, max_concurrent=10):
        self.sem = asyncio.Semaphore(max_concurrent)
        # [채찍 C 반영] Accept-Language 추가로 차단 회피력 극대화
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://finance.naver.com/"
        }
        # 한국 표준시(KST) 정의
        self.kst = timezone(timedelta(hours=9))

    def _parse_html(self, text: str, days: int):
        """
        [채찍 A, B 반영] 
        - KST 기준 날짜 처리로 데이터 오차 제거
        - 순서 보존 중복 제거로 최신성 유지
        """
        headlines = []
        soup = BeautifulSoup(text, "lxml")
        
        # [Fix B] 서버 시간대와 무관하게 KST 기준으로 데드라인 설정
        now_kst = datetime.now(self.kst)
        cutoff_date = now_kst - timedelta(days=days)
        
        rows = soup.select("table.type5 tr")
        for row in rows:
            title_node = row.select_one("td.title > a")
            date_node = row.select_one("td.date")
            
            if title_node and date_node:
                subject = title_node.text.strip()
                date_str = date_node.text.strip()
                
                try:
                    # 네이버 날짜 파싱 (KST 기준 가정)
                    fmt = "%Y.%m.%d %H:%M" if len(date_str) > 10 else "%Y.%m.%d"
                    # 파싱 후 KST 타임존 주입
                    a_date = datetime.strptime(date_str, fmt).replace(tzinfo=self.kst)
                    
                    if a_date >= cutoff_date:
                        if any(k in subject for k in ["특징주", "공시", "체결", "계약", "호재", "수주", "증설", "M&A"]):
                            headlines.append(subject)
                except Exception:
                    continue
        
        # [Fix A] dict.fromkeys를 사용하여 '최신 순서'를 유지하면서 중복만 제거
        return list(dict.fromkeys(headlines))[:5]

    async def fetch_news(self, session, code: str, days: int = 2) -> Dict[str, List[str]]:
        url = f"https://finance.naver.com/item/news_news.naver?code={code}&page=1"
        
        async with self.sem:
            for attempt in range(2):
                try:
                    async with session.get(url, headers=self.headers, timeout=7) as response:
                        if response.status == 200:
                            content = await response.read()
                            text = content.decode('euc-kr', 'replace')
                            
                            # BS4 파싱을 별도 스레드에서 실행 (CPU 병목 방지)
                            headlines = await asyncio.to_thread(self._parse_html, text, days)
                            return {code: headlines}
                        
                        logger.warning(f"⚠️ {code} 응답 코드: {response.status} (시도 {attempt+1})")
                except Exception as e:
                    if attempt == 1: logger.error(f"❌ {code} 뉴스 수집 최종 실패: {e}")
                    await asyncio.sleep(0.5)
            
            return {code: []}

    async def fetch_all(self, codes: List[str]) -> Dict[str, List[str]]:
        results = {}
        # TCP 커넥션 재사용 최적화
        connector = aiohttp.TCPConnector(limit_per_host=5) 
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.fetch_news(session, code) for code in codes]
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in completed:
                if isinstance(res, dict):
                    results.update(res)
        return results
