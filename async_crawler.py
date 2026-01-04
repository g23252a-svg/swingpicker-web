# async_crawler.py
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict

class AsyncNewsFetcher:
    def __init__(self, max_concurrent=5):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.headers = {"User-Agent": "Mozilla/5.0"}

    async def fetch_news(self, session, code: str, days: int = 2) -> Dict[str, List[str]]:
        url = f"https://finance.naver.com/item/news_news.naver?code={code}&page=1"
        headlines = []
        
        async with self.sem:
            try:
                async with session.get(url, headers=self.headers, timeout=5) as response:
                    if response.status != 200:
                        return {code: []}
                    
                    content = await response.read()
                    text = content.decode('euc-kr', 'replace')
                    soup = BeautifulSoup(text, "html.parser")
                    
                    titles = soup.select("td.title > a")
                    dates = soup.select("td.date")
                    
                    cutoff_date = datetime.now() - timedelta(days=days)
                    
                    for t, d in zip(titles, dates):
                        article_date_str = d.text.strip()
                        try:
                            if len(article_date_str) > 10:
                                a_date = datetime.strptime(article_date_str, "%Y.%m.%d %H:%M")
                            else:
                                a_date = datetime.strptime(article_date_str, "%Y.%m.%d")
                            
                            if a_date >= cutoff_date:
                                subject = t.text.strip()
                                # 필터링 (필요시 키워드 추가)
                                if any(k in subject for k in ["특징주", "공시", "체결", "계약", "호재", "수주"]):
                                    headlines.append(subject)
                        except:
                            continue
            except:
                pass
        return {code: list(set(headlines))[:5]}

    async def fetch_all(self, codes: List[str]) -> Dict[str, List[str]]:
        results = {}
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_news(session, code) for code in codes]
            completed = await asyncio.gather(*tasks)
            for res in completed:
                results.update(res)
        return results
