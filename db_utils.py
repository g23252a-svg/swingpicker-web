import duckdb
import pandas as pd
import requests
import json
import os
import streamlit as st
from datetime import datetime, timedelta

# Gist 설정 (secrets.toml에서 가져오기)
GIST_ID = st.secrets.get("LDY_GIST_ID") or os.environ.get("LDY_GIST_ID")
GIST_TOKEN = st.secrets.get("LDY_GIST_TOKEN") or os.environ.get("LDY_GIST_TOKEN")
USER_DB_FILE = "users_db_v2.json"  # Gist에 저장될 파일명

class LDYDBManager:
    def __init__(self):
        # 1. 파일이 아닌 '메모리' DB 사용 (휘발성이지만 빠름)
        self.conn = duckdb.connect(":memory:")
        self._init_tables()
        self._load_from_gist() # 2. 켜질 때 Gist에서 데이터 복원

    def _init_tables(self):
        """사용자 테이블 생성"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR PRIMARY KEY,
                password VARCHAR,
                join_date TIMESTAMP,
                prime_expire_date TIMESTAMP,
                is_admin BOOLEAN DEFAULT FALSE
            )
        """)

    # -------------------------------------------------------
    # ☁️ Gist 연동 (데이터 영구 보존용)
    # -------------------------------------------------------
    def _load_from_gist(self):
        """Gist에서 JSON 데이터를 가져와 DuckDB에 넣음"""
        if not GIST_ID or not GIST_TOKEN:
            print("⚠️ Gist 설정이 없습니다. 데이터가 저장되지 않습니다.")
            return

        try:
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            resp = requests.get(url, headers=headers, timeout=5)
            
            if resp.status_code == 200:
                files = resp.json().get("files", {})
                if USER_DB_FILE in files:
                    content = files[USER_DB_FILE]["content"]
                    users_data = json.loads(content)
                    
                    # DuckDB에 데이터 삽입
                    for u in users_data:
                        self.conn.execute("""
                            INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?)
                        """, [u['id'], u['password'], u['join_date'], u['prime_expire_date'], u['is_admin']])
                    print(f"✅ Gist에서 {len(users_data)}명 유저 복원 완료")
        except Exception as e:
            print(f"⚠️ Gist 로드 실패: {e}")

    def _sync_to_gist(self):
        """DuckDB 데이터를 JSON으로 변환하여 Gist에 업로드"""
        if not GIST_ID or not GIST_TOKEN: return

        try:
            # DuckDB 데이터를 딕셔너리 리스트로 변환
            df = self.conn.execute("SELECT * FROM users").fetchdf()
            # 날짜 객체를 문자열로 변환
            df['join_date'] = df['join_date'].astype(str)
            df['prime_expire_date'] = df['prime_expire_date'].astype(str)
            
            users_json = df.to_dict(orient='records')
            json_str = json.dumps(users_json, ensure_ascii=False, indent=2)

            # Gist 업데이트 API 호출
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            payload = {"files": {USER_DB_FILE: {"content": json_str}}}
            
            requests.patch(url, headers=headers, json=payload, timeout=5)
            print("☁️ Gist 동기화(저장) 완료")
        except Exception as e:
            print(f"⚠️ Gist 저장 실패: {e}")

    # -------------------------------------------------------
    # 👤 사용자 기능 (가입/로그인)
    # -------------------------------------------------------
    def add_user(self, user_id, password):
        """회원가입 + 1주일 무료 + Gist 저장"""
        try:
            check = self.conn.execute("SELECT id FROM users WHERE id = ?", [user_id]).fetchone()
            if check: return False, "이미 존재하는 아이디입니다."

            now = datetime.now()
            expire_date = now + timedelta(days=7)

            # DB Insert
            self.conn.execute("""
                INSERT INTO users VALUES (?, ?, ?, ?, ?)
            """, [user_id, password, now, expire_date, False])
            
            # 🔥 중요: 변경사항 생길 때마다 Gist에 업로드
            self._sync_to_gist()
            
            return True, f"가입 완료! 1주일 무료 체험이 시작됩니다. (~{expire_date.strftime('%Y-%m-%d')})"
        except Exception as e:
            return False, f"오류: {e}"

    def login(self, user_id, password):
        """로그인"""
        row = self.conn.execute("SELECT id FROM users WHERE id = ? AND password = ?", [user_id, password]).fetchone()
        return True if row else False

    def check_prime_status(self, user_id):
        """프라임 만료 확인"""
        row = self.conn.execute("SELECT prime_expire_date FROM users WHERE id = ?", [user_id]).fetchone()
        if row and row[0]:
            expire_dt = row[0]
            remaining = expire_dt - datetime.now()
            
            if remaining.total_seconds() > 0:
                return True, f"프라임 이용 중 (남은 시간: {remaining.days}일)"
            else:
                return False, "프라임 체험이 만료되었습니다."
        return False, "멤버십 정보 없음"
