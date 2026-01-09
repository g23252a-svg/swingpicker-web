# db_utils.py (Admin Event Feature Added)

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

USER_DB_FILE = "users_db.json"
INQUIRY_DB_FILE = "inquiries_db.json" 

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
                id VARCHAR,
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
                    
                    # 1. Users 테이블
                    if tablename == 'users':
                        if isinstance(data, dict) and "users" in data:
                            user_dict = data["users"]
                            for u in user_dict.values():
                                join_dt_str = u.get('created_at', u.get('join_date'))
                                join_dt = None
                                try:
                                    if join_dt_str:
                                        clean_str = str(join_dt_str)[:19].replace("T", " ")
                                        join_dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
                                except: pass

                                # 만료일 없으면 가입일 + 7일로 자동 설정 (복구 로직)
                                expire_val = u.get('prime_expire_date')
                                role = u.get('role', 'free')
                                
                                if not expire_val and role in ['prime', 'pro'] and join_dt:
                                    expire_val = join_dt + timedelta(days=7)
                                
                                vals = [
                                    u.get('login_id', u.get('id')),
                                    u.get('password_hash', u.get('password')),
                                    u.get('salt'),
                                    u.get('nickname'),
                                    role,
                                    join_dt_str,
                                    u.get('last_login'),
                                    u.get('is_banned', False),
                                    u.get('security_q_idx', 0),
                                    u.get('security_a_hash'),
                                    u.get('session_token'),
                                    expire_val
                                ]
                                self.conn.execute("INSERT OR IGNORE INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", vals)
                            print(f"✅ {tablename} 복원 완료 (Dict Format)")
                            return

                        elif isinstance(data, list):
                            for item in data:
                                join_dt_str = item.get('join_date')
                                expire_val = item.get('prime_expire_date')
                                role = item.get('role', 'free')
                                
                                if not expire_val and role in ['prime', 'pro'] and join_dt_str:
                                    try:
                                        clean_str = str(join_dt_str)[:19].replace("T", " ")
                                        join_dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
                                        expire_val = join_dt + timedelta(days=7)
                                    except: pass

                                vals = [
                                    item.get('id'), item.get('password'), item.get('salt'), item.get('nickname'),
                                    role, join_dt_str, item.get('last_login'), item.get('is_banned'),
                                    item.get('security_q_idx'), item.get('security_a_hash'), item.get('session_token'),
                                    expire_val
                                ]
                                self.conn.execute("INSERT OR IGNORE INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", vals)
                            print(f"✅ {tablename} 로드 완료 (List Format)")
                            return

                    # 2. Inquiries 테이블
                    elif tablename == 'inquiries':
                        if isinstance(data, list):
                            for item in data:
                                vals = [
                                    item.get('id'), item.get('nickname'), item.get('title'),
                                    item.get('content'), item.get('created_at')
                                ]
                                self.conn.execute("INSERT INTO inquiries VALUES (?,?,?,?,?)", vals)
                            print(f"✅ {tablename} 로드 완료")

        except Exception as e:
            print(f"⚠️ {tablename} 로드 실패: {e}")

    def _sync_table_to_gist(self, tablename, filename):
        if not GIST_ID or not GIST_TOKEN: return
        try:
            df = self.conn.execute(f"SELECT * FROM {tablename}").fetchdf()
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
        try:
            cols = [x[0] for x in self.conn.execute("DESCRIBE users").fetchall()]
            row = self.conn.execute("SELECT * FROM users WHERE id = ?", [email]).fetchone()
            return dict(zip(cols, row)) if row else None
        except: return None

    def update_login_timestamp(self, email):
        self.conn.execute("UPDATE users SET last_login = ? WHERE id = ?", [datetime.now(), email])
        self._sync_table_to_gist("users", USER_DB_FILE)

    def update_password(self, email, pw, salt):
        self.conn.execute("UPDATE users SET password = ?, salt = ? WHERE id = ?", [pw, salt, email])
        self._sync_table_to_gist("users", USER_DB_FILE)

    # --- Admin Support Methods ---
    def get_all_users(self):
        try:
            cols = [x[0] for x in self.conn.execute("DESCRIBE users").fetchall()]
            rows = self.conn.execute("SELECT * FROM users").fetchall()
            result = []
            for r in rows:
                d = dict(zip(cols, r))
                d['login_id'] = d['id']
                result.append(d)
            return result
        except: return []

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
        try:
            exp_dt = datetime.strptime(expire_date_str, "%Y-%m-%d")
            self.conn.execute("UPDATE users SET role = ?, prime_expire_date = ? WHERE id = ?", 
                              [role, exp_dt, email])
            self._sync_table_to_gist("users", USER_DB_FILE)
        except:
            pass 
            
    # [NEW] 이벤트: 모든 회원에게 N일 체험권 지급
    def grant_all_users_trial(self, days=7):
        try:
            new_expire = datetime.now() + timedelta(days=days)
            # Admin 제외하고 모두 업데이트
            self.conn.execute("UPDATE users SET role = 'prime', prime_expire_date = ? WHERE role != 'admin'", [new_expire])
            self._sync_table_to_gist("users", USER_DB_FILE)
            return True, f"모든 회원(관리자 제외)에게 {days}일 Prime 권한이 부여되었습니다."
        except Exception as e:
            return False, f"DB Error: {e}"

    # --- Inquiry Methods ---
    def get_all_inquiries(self):
        try:
            cols = [x[0] for x in self.conn.execute("DESCRIBE inquiries").fetchall()]
            rows = self.conn.execute("SELECT * FROM inquiries").fetchall()
            result = []
            for r in rows:
                d = dict(zip(cols, r))
                d['email'] = d['id']
                result.append(d)
            return result
        except: return []

    def save_inquiries(self, items):
        self.conn.execute("DELETE FROM inquiries")
        if items:
            for i in items:
                self.conn.execute("INSERT INTO inquiries VALUES (?, ?, ?, ?, ?)", 
                                  [i.get('email', i.get('id')), i.get('nickname'), i.get('title'), i.get('content'), i.get('created_at')])
        self._sync_table_to_gist("inquiries", INQUIRY_DB_FILE)
        return True
