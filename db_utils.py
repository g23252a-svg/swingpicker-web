# -----------------------------------------------------------
# [Upgrade 1] DuckDB Manager Class
# -----------------------------------------------------------
import duckdb
import pandas as pd
from datetime import datetime

class LDYDBManager:
    def __init__(self, db_path="ldy_trader.db"):
        self.db_path = db_path
        self.conn = duckdb.connect(self.db_path)
        self._init_tables()

    def _init_tables(self):
        """테이블 스키마 초기화 (없을 경우 생성)"""
        # 추천 종목 테이블 (날짜 컬럼 포함)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_recommend (
                trade_date DATE,
                code VARCHAR,
                name VARCHAR,
                sector VARCHAR,
                close_price DOUBLE,
                ldy_score DOUBLE,
                rank_score DOUBLE,
                route VARCHAR,
                ai_comment VARCHAR,
                PRIMARY KEY (trade_date, code)
            )
        """)
        # 가격 스냅샷 테이블
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                snap_date DATE,
                code VARCHAR,
                name VARCHAR,
                market VARCHAR,
                close DOUBLE,
                PRIMARY KEY (snap_date, code)
            )
        """)

    def save_recommendations(self, df: pd.DataFrame, ymd: str):
        """추천 종목 DataFrame을 DB에 저장 (중복 시 무시 혹은 덮어쓰기 로직 가능)"""
        if df is None or df.empty:
            return
        
        # 날짜 포맷팅 (YYYYMMDD -> YYYY-MM-DD)
        date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
        
        # 필요한 컬럼만 선택 및 매핑 (예시)
        save_df = df.copy()
        save_df['trade_date'] = date_str
        
        # 컬럼명 매핑 (한글 -> 영문 DB 컬럼명)
        col_map = {
            '종목코드': 'code', '종목명': 'name', '업종': 'sector',
            '종가': 'close_price', 'LDY_SCORE': 'ldy_score',
            'RANK_SCORE': 'rank_score', 'ROUTE': 'route',
            'AI_COMMENT': 'ai_comment'
        }
        save_df = save_df.rename(columns=col_map)
        
        # DB에 없는 컬럼 제거 및 순서 정렬을 위한 쿼리 생성
        try:
            # DuckDB의 강력한 기능: DF를 바로 쿼리에서 참조 가능
            self.conn.execute(f"""
                INSERT OR IGNORE INTO daily_recommend 
                SELECT 
                    CAST('{date_str}' AS DATE) as trade_date,
                    code, name, sector, close_price, ldy_score, rank_score, route, ai_comment
                FROM save_df
            """)
            print(f"✅ [DuckDB] 추천 종목 저장 완료 ({len(save_df)}건)")
        except Exception as e:
            print(f"⚠️ [DuckDB] 저장 실패: {e}")

    def save_snapshot(self, df: pd.DataFrame, ymd: str):
        """전 종목 시세 스냅샷 저장"""
        if df is None or df.empty:
            return
            
        date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
        save_df = df.copy()
        
        col_map = {'종목코드': 'code', '종목명': 'name', '시장': 'market', '종가': 'close'}
        save_df = save_df.rename(columns=col_map)
        
        try:
            self.conn.execute(f"""
                INSERT OR IGNORE INTO price_snapshots
                SELECT 
                    CAST('{date_str}' AS DATE) as snap_date,
                    code, name, market, close
                FROM save_df
            """)
            print(f"✅ [DuckDB] 가격 스냅샷 저장 완료 ({len(save_df)}건)")
        except Exception as e:
            print(f"⚠️ [DuckDB] 스냅샷 저장 실패: {e}")

    def close(self):
        self.conn.close()
