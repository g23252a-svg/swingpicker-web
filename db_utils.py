# -----------------------------------------------------------
# [Upgrade 2] DuckDB Manager Class (User & Prime Logic Added)
# -----------------------------------------------------------
import duckdb
import pandas as pd
from datetime import datetime, timedelta  # ✅ timedelta 추가 (날짜 계산용)

class LDYDBManager:
    def __init__(self, db_path="ldy_trader.db"):
        self.db_path = db_path
        # DuckDB는 동시성 제어가 엄격하므로, read_only=False로 확실히 명시
        self.conn = duckdb.connect(self.db_path, read_only=False)
        self._init_tables()

    def _init_tables(self):
        """테이블 스키마 초기화"""
        
        # 1. 추천 종목 테이블
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
        
        # 2. 가격 스냅샷 테이블
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

        # 3. ✅ [NEW] 사용자 테이블 (프라임 만료일 포함)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR PRIMARY KEY,
                password VARCHAR NOT NULL,
                join_date TIMESTAMP,
                prime_expire_date TIMESTAMP,
                is_admin BOOLEAN DEFAULT FALSE
            )
        """)

    # -------------------------------------------------------
    # 🆕 사용자 관리 & 프라임 로직 추가됨
    # -------------------------------------------------------
    
    def add_user(self, user_id, password):
        """
        신규 회원가입: 가입 즉시 1주일(7일) 무료 프라임 체험 부여
        """
        try:
            # ID 중복 체크
            check = self.conn.execute("SELECT id FROM users WHERE id = ?", [user_id]).fetchone()
            if check:
                return False, "이미 존재하는 아이디입니다."

            now = datetime.now()
            # 🔥 [핵심] 가입일 + 7일 = 무료 체험 만료일
            expire_date = now + timedelta(days=7)

            self.conn.execute("""
                INSERT INTO users (id, password, join_date, prime_expire_date, is_admin)
                VALUES (?, ?, ?, ?, ?)
            """, [user_id, password, now, expire_date, False])
            
            return True, f"회원가입 완료! 🎁 1주일 프라임 무료 체험이 시작됩니다. (~{expire_date.strftime('%Y-%m-%d')})"
            
        except Exception as e:
            return False, f"가입 오류 발생: {e}"

    def login(self, user_id, password):
        """로그인 확인"""
        row = self.conn.execute("""
            SELECT id FROM users WHERE id = ? AND password = ?
        """, [user_id, password]).fetchone()
        
        return True if row else False

    def check_prime_status(self, user_id):
        """
        프라임 멤버십 상태 확인
        Returns: (is_prime: bool, message: str)
        """
        row = self.conn.execute("SELECT prime_expire_date FROM users WHERE id = ?", [user_id]).fetchone()
        
        if row and row[0]:
            expire_dt = row[0]  # DuckDB는 datetime 객체로 반환함
            remaining = expire_dt - datetime.now()
            
            if remaining.total_seconds() > 0:
                days = remaining.days
                return True, f"프라임 이용 중 (남은 기간: {days}일)"
            else:
                return False, "프라임 체험이 만료되었습니다. 구독이 필요합니다."
        
        return False, "멤버십 정보가 없습니다."

    # -------------------------------------------------------
    # (기존 데이터 저장 함수들 유지)
    # -------------------------------------------------------

    def save_recommendations(self, df: pd.DataFrame, ymd: str):
        if df is None or df.empty: return
        
        date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
        save_df = df.copy()
        save_df['trade_date'] = date_str
        
        col_map = {
            '종목코드': 'code', '종목명': 'name', '업종': 'sector',
            '종가': 'close_price', 'LDY_SCORE': 'ldy_score',
            'RANK_SCORE': 'rank_score', 'ROUTE': 'route',
            'AI_COMMENT': 'ai_comment'
        }
        # 존재하는 컬럼만 rename
        existing_cols = {k: v for k, v in col_map.items() if k in save_df.columns}
        save_df = save_df.rename(columns=existing_cols)
        
        try:
            # DuckDB register 기능을 이용해 DataFrame을 SQL에서 바로 사용
            self.conn.register('temp_rec_df', save_df)
            self.conn.execute(f"""
                INSERT OR IGNORE INTO daily_recommend 
                (trade_date, code, name, sector, close_price, ldy_score, rank_score, route, ai_comment)
                SELECT 
                    CAST(trade_date AS DATE), code, name, sector, close_price, ldy_score, rank_score, route, ai_comment
                FROM temp_rec_df
            """)
            self.conn.unregister('temp_rec_df')
            print(f"✅ [DuckDB] 추천 종목 저장 완료 ({len(save_df)}건)")
        except Exception as e:
            print(f"⚠️ [DuckDB] 저장 실패: {e}")

    def save_snapshot(self, df: pd.DataFrame, ymd: str):
        if df is None or df.empty: return
            
        date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
        save_df = df.copy()
        
        col_map = {'종목코드': 'code', '종목명': 'name', '시장': 'market', '종가': 'close'}
        existing_cols = {k: v for k, v in col_map.items() if k in save_df.columns}
        save_df = save_df.rename(columns=existing_cols)
        
        try:
            self.conn.register('temp_snap_df', save_df)
            self.conn.execute(f"""
                INSERT OR IGNORE INTO price_snapshots (snap_date, code, name, market, close)
                SELECT 
                    CAST('{date_str}' AS DATE), code, name, market, close
                FROM temp_snap_df
            """)
            self.conn.unregister('temp_snap_df')
            print(f"✅ [DuckDB] 가격 스냅샷 저장 완료 ({len(save_df)}건)")
        except Exception as e:
            print(f"⚠️ [DuckDB] 스냅샷 저장 실패: {e}")

    def close(self):
        self.conn.close()
