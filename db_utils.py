# db_utils.py (DuckDB + Gist Sync Full Version)

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
USER_DB_FILE = "users_db_v3.json"  # 파일명 변경 (구조 변경됨)

class LDYDBManager:
    def __init__(self):
        self.conn = duckdb.connect(":memory:")
        self._init_tables()
        self._load_from_gist()

    def _init_tables(self):
        """기존 JSON 구조를 모두 수용하는 테이블 생성"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR PRIMARY KEY,       -- login_id (이메일)
                password VARCHAR,             -- password_hash
                salt VARCHAR,                 -- salt
                nickname VARCHAR,             -- 닉네임
                role VARCHAR DEFAULT 'free',  -- 권한 (free/prime/pro/admin)
                join_date TIMESTAMP,          -- created_at
                last_login TIMESTAMP,         -- last_login
                is_banned BOOLEAN DEFAULT FALSE,
                
                -- 보안 질문 관련
                security_q_idx INTEGER DEFAULT 0,
                security_a_hash VARCHAR,
                
                -- 세션 관리
                session_token VARCHAR,
                
                -- 프라임 만료일 (New)
                prime_expire_date TIMESTAMP
            )
        """)

    # -------------------------------------------------------
    # ☁️ Gist 동기화 로직 (JSON <-> DuckDB)
    # -------------------------------------------------------
    def _load_from_gist(self):
        """Gist -> DuckDB 복원"""
        if not GIST_ID or not GIST_TOKEN: return
        try:
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                files = resp.json().get("files", {})
                if USER_DB_FILE in files:
                    content = files[USER_DB_FILE]["content"]
                    data = json.loads(content)
                    
                    # Insert data
                    for u in data:
                        # None 값 처리
                        u = {k: (v if v is not None else None) for k, v in u.items()}
                        self.conn.execute("""
                            INSERT OR IGNORE INTO users VALUES 
                            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            u.get('id'), u.get('password'), u.get('salt'), u.get('nickname'),
                            u.get('role'), u.get('join_date'), u.get('last_login'), u.get('is_banned'),
                            u.get('security_q_idx'), u.get('security_a_hash'), u.get('session_token'),
                            u.get('prime_expire_date')
                        ])
                    print(f"✅ Gist 데이터 복원 완료 ({len(data)}명)")
        except Exception as e:
            print(f"⚠️ Gist 로드 실패: {e}")

    def _sync_to_gist(self):
        """DuckDB -> Gist 백업"""
        if not GIST_ID or not GIST_TOKEN: return
        try:
            df = self.conn.execute("SELECT * FROM users").fetchdf()
            # 날짜 컬럼 문자열 변환
            for col in ['join_date', 'last_login', 'prime_expire_date']:
                df[col] = df[col].astype(str).replace('NaT', None)
                
            json_str = df.to_json(orient='records', force_ascii=False, indent=2)
            
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            payload = {"files": {USER_DB_FILE: {"content": json_str}}}
            requests.patch(url, headers=headers, json=payload, timeout=5)
        except Exception as e:
            print(f"⚠️ Gist 저장 실패: {e}")

    # -------------------------------------------------------
    # 👤 사용자 기능 (Full Features)
    # -------------------------------------------------------
    def register_user(self, email, pw_hash, salt, nickname, q_idx, a_hash, role='free'):
        """회원가입 (모든 정보 포함 + 1주 무료)"""
        try:
            check = self.conn.execute("SELECT id FROM users WHERE id = ?", [email]).fetchone()
            if check: return False, "이미 존재하는 이메일입니다."

            now = datetime.now()
            # 🔥 프라임 1주일 자동 적용 (만료일 설정)
            expire_date = now + timedelta(days=7) if role in ['free', 'prime'] else None
            final_role = 'prime' if role == 'free' else role  # 무료가입 -> 프라임 체험

            self.conn.execute("""
                INSERT INTO users (
                    id, password, salt, nickname, role, join_date, last_login, 
                    is_banned, security_q_idx, security_a_hash, session_token, prime_expire_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                email, pw_hash, salt, nickname, final_role, now, now, 
                False, q_idx, a_hash, "init_token", expire_date
            ])
            
            self._sync_to_gist()
            return True, f"가입 완료! 🎁 7일간 프라임 혜택이 적용됩니다."
        except Exception as e:
            return False, f"DB 오류: {e}"

    def get_user_by_id(self, email):
        """사용자 정보 전체 조회 (딕셔너리 반환)"""
        columns = [desc[0] for desc in self.conn.execute("DESCRIBE users").fetchall()]
        row = self.conn.execute("SELECT * FROM users WHERE id = ?", [email]).fetchone()
        if row:
            return dict(zip(columns, row))
        return None

    def update_login_timestamp(self, email):
        """로그인 시간 업데이트"""
        self.conn.execute("UPDATE users SET last_login = ? WHERE id = ?", [datetime.now(), email])
        self._sync_to_gist()

    def update_password(self, email, new_pw_hash, new_salt):
        """비밀번호 변경 (복구용)"""
        self.conn.execute("""
            UPDATE users SET password = ?, salt = ? WHERE id = ?
        """, [new_pw_hash, new_salt, email])
        self._sync_to_gist()
