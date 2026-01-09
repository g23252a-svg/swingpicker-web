# db_utils.py (Admin & Inquiry Support Version)

import duckdb
import pandas as pd
import requests
import json
import os
import streamlit as st
from datetime import datetime, timedelta

# Gist 설정
GIST_ID = st.secrets.get("LDY_GIST_ID") or os.environ.get("LDY_GIST_ID")
GIST_TOKEN = st.secrets.get("LDY_GIST_TOKEN") or os.environ.get("LDY_GIST_TOKEN")
USER_DB_FILE = "users_db_v3.json"
INQUIRY_DB_FILE = "inquiries_db_v3.json" # 문의 내역 저장용

class LDYDBManager:
    def __init__(self):
        self.conn = duckdb.connect(":memory:")
        self._init_tables()
        self._load_users_from_gist()
        self._load_inquiries_from_gist()

    def _init_tables(self):
        # 1. 사용자 테이블
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR PRIMARY KEY,
                password VARCHAR,
                salt VARCHAR,
                nickname VARCHAR,
                role VARCHAR DEFAULT 'free',
                join_date TIMESTAMP,
                last_login TIMESTAMP,
                is_banned BOOLEAN DEFAULT FALSE,
                security_q_idx INTEGER DEFAULT 0,
                security_a_hash VARCHAR,
                session_token VARCHAR,
                prime_expire_date TIMESTAMP
            )
        """)
        # 2. 문의 게시판 테이블
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS inquiries (
                id VARCHAR,  -- email
                nickname VARCHAR,
                title VARCHAR,
                content VARCHAR,
                created_at VARCHAR
            )
        """)

    # --- Gist Sync Logic ---
    def _load_users_from_gist(self):
        self._load_gist_to_table(USER_DB_FILE, "users", 12)

    def _load_inquiries_from_gist(self):
        self._load_gist_to_table(INQUIRY_DB_FILE, "inquiries", 5)

    def _load_gist_to_table(self, filename, tablename, col_count):
        if not GIST_ID or not GIST_TOKEN: return
        try:
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                files = resp.json().get("files", {})
                if filename in files:
                    content = files[filename]["content"]
                    data = json.loads(content)
                    if not data: return
                    
                    # 'users'와 'inquiries' 구조에 따라 처리
                    placeholders = ", ".join(["?"] * col_count)
                    for item in data:
                        # dict values를 순서대로 리스트화 (스키마 순서 중요)
                        if tablename == 'users':
                            vals = [
                                item.get('id'), item.get('password'), item.get('salt'), item.get('nickname'),
                                item.get('role'), item.get('join_date'), item.get('last_login'), item.get('is_banned'),
                                item.get('security_q_idx'), item.get('security_a_hash'), item.get('session_token'),
                                item.get('prime_expire_date')
                            ]
                        else: # inquiries
                            vals = [
                                item.get('id'), item.get('nickname'), item.get('title'),
                                item.get('content'), item.get('created_at')
                            ]
                        
                        self.conn.execute(f"INSERT INTO {tablename} VALUES ({placeholders})", vals)
                    print(f"✅ {tablename} 복원 완료 ({len(data)}건)")
        except Exception as e:
            print(f"⚠️ {tablename} 로드 실패: {e}")

    def _sync_table_to_gist(self, tablename, filename):
        if not GIST_ID or not GIST_TOKEN: return
        try:
            df = self.conn.execute(f"SELECT * FROM {tablename}").fetchdf()
            # 날짜 등 객체 타입을 문자열로 변환
            for col in df.columns:
                if df[col].dtype == 'object' or 'date' in str(df[col].dtype):
                    df[col] = df[col].astype(str).replace('NaT', None).replace('nan', None)
            
            json_str = df.to_json(orient='records', force_ascii=False, indent=2)
            
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            payload = {"files": {filename: {"content": json_str}}}
            requests.patch(url, headers=headers, json=payload, timeout=5)
        except Exception as e:
            print(f"⚠️ {tablename} 저장 실패: {e}")

    # --- User Methods ---
    def register_user(self, email, pw_hash, salt, nickname, q_idx, a_hash):
        try:
            check = self.conn.execute("SELECT id FROM users WHERE id = ?", [email]).fetchone()
            if check: return False, "이미 존재하는 이메일입니다."
            now = datetime.now()
            expire = now + timedelta(days=7) # 7일 무료
            self.conn.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                              [email, pw_hash, salt, nickname, 'prime', now, now, False, q_idx, a_hash, "token", expire])
            self._sync_table_to_gist("users", USER_DB_FILE)
            return True, "가입 완료! 🎁 7일간 프라임 혜택 적용"
        except Exception as e: return False, f"DB Error: {e}"

    def get_user_by_id(self, email):
        cols = [x[0] for x in self.conn.execute("DESCRIBE users").fetchall()]
        row = self.conn.execute("SELECT * FROM users WHERE id = ?", [email]).fetchone()
        return dict(zip(cols, row)) if row else None

    def update_login_timestamp(self, email):
        self.conn.execute("UPDATE users SET last_login = ? WHERE id = ?", [datetime.now(), email])
        self._sync_table_to_gist("users", USER_DB_FILE)

    def update_password(self, email, pw, salt):
        self.conn.execute("UPDATE users SET password = ?, salt = ? WHERE id = ?", [pw, salt, email])
        self._sync_table_to_gist("users", USER_DB_FILE)

    # --- Admin Support Methods (New) ---
    def get_all_users(self):
        """관리자 페이지용 전체 유저 리스트"""
        cols = [x[0] for x in self.conn.execute("DESCRIBE users").fetchall()]
        rows = self.conn.execute("SELECT * FROM users").fetchall()
        # dashboard.py가 기대하는 'login_id' 키 등을 맞춰주기 위해 딕셔너리 변환
        result = []
        for r in rows:
            d = dict(zip(cols, r))
            d['login_id'] = d['id'] # 호환성
            result.append(d)
        return result

    def update_user_role(self, email, new_role):
        self.conn.execute("UPDATE users SET role = ? WHERE id = ?", [new_role, email])
        self._sync_table_to_gist("users", USER_DB_FILE)
        return True

    def toggle_user_ban(self, email):
        curr = self.conn.execute("SELECT is_banned FROM users WHERE id = ?", [email]).fetchone()
        if not curr: return False, "유저 없음"
        new_stat = not curr[0]
        self.conn.execute("UPDATE users SET is_banned = ? WHERE id = ?", [new_stat, email])
        self._sync_table_to_gist("users", USER_DB_FILE)
        return True, f"{'차단' if new_stat else '해제'} 완료"

    def update_user_subscription(self, email, role, expire_date_str):
        """구독 정보(역할, 만료일) 업데이트"""
        try:
            exp_dt = datetime.strptime(expire_date_str, "%Y-%m-%d")
            self.conn.execute("UPDATE users SET role = ?, prime_expire_date = ? WHERE id = ?", 
                              [role, exp_dt, email])
            self._sync_table_to_gist("users", USER_DB_FILE)
        except:
            pass # 날짜 포맷 에러 등 무시

    # --- Inquiry Methods (New) ---
    def get_all_inquiries(self):
        cols = [x[0] for x in self.conn.execute("DESCRIBE inquiries").fetchall()]
        rows = self.conn.execute("SELECT * FROM inquiries").fetchall()
        result = []
        for r in rows:
            d = dict(zip(cols, r))
            d['email'] = d['id'] # 호환성
            result.append(d)
        return result

    def save_inquiries(self, items):
        """리스트 전체를 덮어쓰기 (기존 로직 유지)"""
        self.conn.execute("DELETE FROM inquiries")
        if items:
            # executemany가 안될 수 있으니 loop
            for i in items:
                self.conn.execute("INSERT INTO inquiries VALUES (?, ?, ?, ?, ?)", 
                                  [i.get('email', i.get('id')), i.get('nickname'), i.get('title'), i.get('content'), i.get('created_at')])
        self._sync_table_to_gist("inquiries", INQUIRY_DB_FILE)
        return True
