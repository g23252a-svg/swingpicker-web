# db_utils.py (Admin Event Feature Added)

import duckdb
import pandas as pd
import requests
import json
import os
try:
    import streamlit as st
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
from datetime import datetime, timedelta

# Gist 설정 — os.environ 우선, st.secrets는 안전하게 폴백
def _safe_secret(key: str) -> str:
    """환경변수 → st.secrets 순으로 조회. secrets.toml 없어도 크래시 안 남."""
    val = os.environ.get(key)
    if val: return val
    if _HAS_ST:
        try:
            return st.secrets.get(key) or ""
        except Exception:
            return ""
    return ""

GIST_ID = _safe_secret("LDY_GIST_ID") or None
GIST_TOKEN = _safe_secret("LDY_GIST_TOKEN") or None

# [진단 로그] Gist 연동 상태 확인
if GIST_ID and GIST_TOKEN:
    print(f"✅ [Gist] 연동 준비 완료 (ID: {GIST_ID[:8]}...)")
else:
    _missing = []
    if not GIST_ID: _missing.append("LDY_GIST_ID")
    if not GIST_TOKEN: _missing.append("LDY_GIST_TOKEN")
    print(f"⚠️ [Gist] 연동 불가 — 누락된 키: {', '.join(_missing)}")

USER_DB_FILE = "users_db.json"
INQUIRY_DB_FILE = "inquiries_db.json" 

class LDYDBManager:
    def __init__(self):
        # 파일 DB 사용 (없으면 생성됨)
        self.conn = duckdb.connect("ldy_trader.db") 
        self._init_tables()
        self._load_users_from_gist()
        self._load_inquiries_from_gist()

    def _init_tables(self):
        # 1. 사용자 테이블 (로그인 실패 및 잠금 컬럼 추가)
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
                prime_expire_date TIMESTAMP,
                login_fail_count INTEGER DEFAULT 0,  -- 👈 추가: 실패 횟수
                lock_until TIMESTAMP                 -- 👈 추가: 잠금 해제 시간
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
        if not GIST_ID or not GIST_TOKEN:
            print(f"⚠️ [Gist] {tablename} 로드 스킵 — Gist 인증 키 없음")
            return
        try:
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            resp = requests.get(url, headers=headers, timeout=5)

            if resp.status_code != 200:
                print(f"❌ [Gist] API 응답 실패: {resp.status_code} — {resp.text[:200]}")
                return

            files = resp.json().get("files", {})
            if filename not in files:
                print(f"⚠️ [Gist] '{filename}' 파일이 Gist에 없음. 존재하는 파일: {list(files.keys())}")
                return

            content = files[filename]["content"]
            data = json.loads(content)
            if not data:
                print(f"⚠️ [Gist] '{filename}' 데이터가 비어있음")
                return

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
                        self.conn.execute("""
                            INSERT OR IGNORE INTO users 
                            (id, password, salt, nickname, role, join_date, last_login, 
                             is_banned, security_q_idx, security_a_hash, session_token, prime_expire_date) 
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """, vals)
                    print(f"✅ {tablename} 복원 완료 (Dict Format, {len(user_dict)}명)")
                    return

                elif isinstance(data, list):
                    loaded = 0
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
                        self.conn.execute("""
                            INSERT OR IGNORE INTO users 
                            (id, password, salt, nickname, role, join_date, last_login, 
                             is_banned, security_q_idx, security_a_hash, session_token, prime_expire_date) 
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """, vals)
                        loaded += 1
                    print(f"✅ {tablename} 로드 완료 (List Format, {loaded}명)")
                    return

                else:
                    print(f"⚠️ [Gist] {filename} 데이터 형식 미지원: {type(data).__name__}")

            # 2. Inquiries 테이블
            elif tablename == 'inquiries':
                if isinstance(data, list):
                    for item in data:
                        vals = [
                            item.get('id'), item.get('nickname'), item.get('title'),
                            item.get('content'), item.get('created_at')
                        ]
                        self.conn.execute("INSERT INTO inquiries VALUES (?,?,?,?,?)", vals)
                    print(f"✅ {tablename} 로드 완료 ({len(data)}건)")

        except Exception as e:
            print(f"❌ [Gist] {tablename} 로드 실패: {e}")

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
            
            # [핵심 수정] 7일 무료 자동 지급 로직 제거! 
            # 이제 가입하면 무조건 'free' 등급이고 만료일은 없습니다.
            role = 'free'       # 기존 'prime' -> 'free'로 변경
            expire = None       # 기존 'now + 7일' -> None으로 변경
            
            self.conn.execute("""
                INSERT INTO users 
                (id, password, salt, nickname, role, join_date, last_login, 
                 is_banned, security_q_idx, security_a_hash, session_token, prime_expire_date) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [email, pw_hash, salt, nickname, role, now, now, False, q_idx, a_hash, "token", expire])
            
            self._sync_table_to_gist("users", USER_DB_FILE)
            
            # 메시지도 변경
            return True, "가입 완료! (체험권이 필요하시면 '문의 게시판'에 신청해주세요)"
            
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

    def update_user_password(self, email, pw_hash, salt):
        """[v11.0 패치] 비밀번호 변경 시 Salt Rotation 필수 적용"""
        try:
            self.conn.execute("UPDATE users SET password = ?, salt = ?, login_fail_count = 0, lock_until = NULL WHERE id = ?", 
                             [pw_hash, salt, email])
            self._sync_table_to_gist("users", USER_DB_FILE)
            return True
        except Exception as e:
            print(f"❌ 비밀번호 업데이트 실패: {e}")
            return False

    # --- [v11.0 보안 패치: Rate Limit 로직] ---
    def record_login_failure(self, email):
        """실패 횟수 누적 및 5회 초과 시 10분 잠금"""
        try:
            curr = self.conn.execute("SELECT login_fail_count FROM users WHERE id = ?", [email]).fetchone()
            if not curr: return
            
            new_count = curr[0] + 1
            lock_time = None
            if new_count >= 5:
                lock_time = datetime.now() + timedelta(minutes=10)
                print(f"🔒 {email} 계정 10분 잠금 발동")
                
            self.conn.execute("UPDATE users SET login_fail_count = ?, lock_until = ? WHERE id = ?", 
                             [new_count, lock_time, email])
            self._sync_table_to_gist("users", USER_DB_FILE)
        except Exception as e:
            print(f"⚠️ 실패 기록 오류: {e}")

    def get_login_failures(self, email):
        """현재 실패 횟수와 잠금 시간 조회"""
        try:
            res = self.conn.execute("SELECT login_fail_count, lock_until FROM users WHERE id = ?", [email]).fetchone()
            if res:
                # 0: count, 1: lock_until (aware datetime 처리를 위해 timezone 주입 권장)
                return res[0], res[1].replace(tzinfo=timezone.utc) if res[1] else None
            return 0, None
        except:
            return 0, None

    def reset_login_failures(self, email):
        """로그인 성공 시 실패 기록 초기화"""
        self.conn.execute("UPDATE users SET login_fail_count = 0, lock_until = NULL WHERE id = ?", [email])
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

    def save_recommendations(self, df, trade_ymd=None):
        """
        [v11.2 Fix] 일일 추천 종목 결과 저장
        - 'str' object has no attribute 'astype' 오류 해결
        - 컬럼 존재 여부 체크 강화 및 안전한 데이터 타입 변환
        """
        if df is None or df.empty:
            return
            
        try:
            # 1. 테이블 스키마 체크 및 자동 보정
            try:
                table_info = self.conn.execute("PRAGMA table_info(daily_recommend)").fetchall()
                if len(table_info) > 0 and len(table_info) != 7:
                    print(f"⚠️ 스키마 불일치 감지. 테이블을 재생성합니다.")
                    self.conn.execute("DROP TABLE daily_recommend")
            except:
                pass

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_recommend (
                    trade_date VARCHAR,
                    code VARCHAR,
                    name VARCHAR,
                    close_price DOUBLE,
                    display_score DOUBLE,  -- 👈 ldy_score에서 명칭 변경 또는 추가
                    final_score DOUBLE,    -- 👈 rank_score에서 명칭 변경 또는 추가
                    ai_comment VARCHAR
                )
            """)

            # 2. 데이터 전처리 (안전한 방식)
            save_df = df.copy()
            
            # [날짜 처리]
            if trade_ymd:
                s_ymd = str(trade_ymd)
                formatted_date = f"{s_ymd[:4]}-{s_ymd[4:6]}-{s_ymd[6:]}" if (len(s_ymd) == 8 and s_ymd.isdigit()) else s_ymd
                save_df['trade_date'] = formatted_date
            elif '기준일' in save_df.columns:
                save_df['trade_date'] = save_df['기준일'].astype(str)
            else:
                save_df['trade_date'] = datetime.now().strftime("%Y-%m-%d")
                
            # [필수 컬럼 변환]
            save_df['code'] = save_df['종목코드'].astype(str).str.zfill(6)
            save_df['name'] = save_df['종목명']
            save_df['close_price'] = pd.to_numeric(save_df['종가'], errors='coerce').fillna(0)
            
            # [점수 및 코멘트 처리 - 에러 방어]
            save_df['display_score'] = pd.to_numeric(save_df.get('DISPLAY_SCORE', save_df.get('LDY_SCORE', 0)), errors='coerce').fillna(0)
            save_df['final_score'] = pd.to_numeric(save_df.get('FINAL_SCORE', 0), errors='coerce').fillna(0)
            
            # ✅ 문제의 구간 수정: 컬럼이 있으면 변환, 없으면 빈 문자열 할당
            if 'AI_COMMENT' in save_df.columns:
                save_df['ai_comment'] = save_df['AI_COMMENT'].astype(str).fillna("")
            else:
                save_df['ai_comment'] = ""

            # 3. DB 삽입 (7개 컬럼 명시)
            target_cols = ['trade_date', 'code', 'name', 'close_price', 'display_score', 'final_score', 'ai_comment']
            target_df = save_df[target_cols]

            if target_df.empty:
                return

            date_val = target_df['trade_date'].iloc[0]
            self.conn.execute(f"DELETE FROM daily_recommend WHERE trade_date = '{date_val}'")
            self.conn.execute("INSERT INTO daily_recommend SELECT * FROM target_df")
            
            print(f"✅ DB Saved: {len(target_df)} rows for {date_val}")
            
        except Exception as e:
            print(f"❌ DB Save Failed: {e}")

    def save_inquiries(self, items):
        self.conn.execute("DELETE FROM inquiries")
        if items:
            for i in items:
                self.conn.execute("INSERT INTO inquiries VALUES (?, ?, ?, ?, ?)", 
                                  [i.get('email', i.get('id')), i.get('nickname'), i.get('title'), i.get('content'), i.get('created_at')])
        self._sync_table_to_gist("inquiries", INQUIRY_DB_FILE)
        return True


    # 👇👇 [여기] 클래스 맨 마지막에 close 메서드 추가 👇👇
    def close(self):
        """DB 연결 종료"""
        try:
            self.conn.close()
        except Exception as e:
            print(f"⚠️ DB Close Error: {e}")
