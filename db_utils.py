# db_utils.py — DuckDB 싱글톤 + Gist 동기화 (v5.0)
# ═══════════════════════════════════════════════════
# [v5.0] 3건 치명 버그 수정:
#   #1 execute_safe 전면 적용 (conn.execute 직접 호출 전면 금지)
#   #2 _sync_table_to_gist → Fire & Forget 백그라운드 (UI freeze 제거)
#   #3 f-string SQL 인젝션 → 파라미터 바인딩
# ═══════════════════════════════════════════════════

import duckdb
import json
import logging
import os
import threading
import time as _time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

_logger = logging.getLogger("db_utils")

# Gist 설정 — Railway 환경변수 전용
def _safe_secret(key: str) -> str:
    return os.environ.get(key, "")

GIST_ID = _safe_secret("LDY_GIST_ID") or None
GIST_TOKEN = _safe_secret("LDY_GIST_TOKEN") or None

if GIST_ID and GIST_TOKEN:
    _logger.info(f" [Gist] 연동 준비 완료 (ID: {GIST_ID[:8]}...)")
else:
    _missing = []
    if not GIST_ID: _missing.append("LDY_GIST_ID")
    if not GIST_TOKEN: _missing.append("LDY_GIST_TOKEN")
    _logger.warning(f" [Gist] 연동 불가 — 누락된 키: {', '.join(_missing)}")

USER_DB_FILE = "users_db.json"
INQUIRY_DB_FILE = "inquiries_db.json"


class LDYDBManager:
    _DB_PATH = "ldy_trader.db"

    def __init__(self):
        self.conn = duckdb.connect(self._DB_PATH)
        self._conn_lock = threading.Lock()
        self._gist_loaded = False
        self._init_tables()

    def ensure_gist_loaded(self):
        """Gist 데이터를 lazy 로드 (최초 1회만 실행)"""
        if self._gist_loaded:
            return
        try:
            self._load_users_from_gist()
            self._load_inquiries_from_gist()
            self._gist_loaded = True
        except Exception as e:
            _logger.warning(f"Gist 초기 로드 실패 (DB는 정상): {e}")

    def _reconnect(self):
        """Stale 커넥션 재연결"""
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = duckdb.connect(self._DB_PATH)

    def execute_safe(self, query, params=None):
        """Thread-safe 쿼리 실행 + 커넥션 헬스체크

        [v5.0 #1] 클래스 내부의 모든 DB 호출은 반드시 이 메서드를 경유.
        """
        with self._conn_lock:
            try:
                return self.conn.execute(query, params) if params else self.conn.execute(query)
            except (duckdb.ConnectionException, duckdb.IOException) as e:
                _logger.warning(f"DuckDB 재연결: {e}")
                self._reconnect()
                return self.conn.execute(query, params) if params else self.conn.execute(query)

    # ═══════════════════════════════════════════
    #  테이블 초기화
    # ═══════════════════════════════════════════

    def _init_tables(self):
        self.execute_safe("""
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
                login_fail_count INTEGER DEFAULT 0,
                lock_until TIMESTAMP
            )
        """)
        self.execute_safe("""
            CREATE TABLE IF NOT EXISTS inquiries (
                id VARCHAR,
                nickname VARCHAR,
                title VARCHAR,
                content VARCHAR,
                created_at VARCHAR
            )
        """)

        # [v19.5] 추천/스냅샷 테이블 미리 생성 + 조회 성능용 인덱스
        self.execute_safe("""
            CREATE TABLE IF NOT EXISTS daily_recommend (
                trade_date VARCHAR, code VARCHAR, name VARCHAR,
                close_price DOUBLE, display_score DOUBLE,
                final_score DOUBLE, ai_comment VARCHAR
            )
        """)
        self.execute_safe("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                trade_date VARCHAR, code VARCHAR, name VARCHAR,
                market VARCHAR, close_price DOUBLE, open_price DOUBLE,
                low_price DOUBLE, high_price DOUBLE
            )
        """)
        self.execute_safe("CREATE INDEX IF NOT EXISTS idx_rec_date ON daily_recommend (trade_date)")
        self.execute_safe("CREATE INDEX IF NOT EXISTS idx_snap_date ON price_snapshots (trade_date)")

    # ═══════════════════════════════════════════
    #  Gist 다운로드 (Read — 락 밖 I/O)
    # ═══════════════════════════════════════════

    def _download_gist_data(self) -> dict:
        """Gist 데이터를 다운로드만 수행 (락 없이, I/O만)"""
        if not GIST_ID or not GIST_TOKEN:
            return {}
        try:
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return {}
            files = resp.json().get("files", {})
            result = {}
            for fname in [USER_DB_FILE, INQUIRY_DB_FILE]:
                if fname in files:
                    content = files[fname].get("content", "")
                    if content:
                        result[fname] = json.loads(content)
            return result
        except Exception as e:
            _logger.warning(f" [Gist] 다운로드 실패: {e}")
            return {}

    def _apply_gist_data(self, downloaded: dict):
        """다운로드된 Gist 데이터를 DB에 적용"""
        if USER_DB_FILE in downloaded:
            self._insert_gist_users(downloaded[USER_DB_FILE])
        if INQUIRY_DB_FILE in downloaded:
            self._insert_gist_inquiries(downloaded[INQUIRY_DB_FILE])

    def _insert_gist_users(self, data):
        if not data:
            return
        try:
            if isinstance(data, dict) and "users" in data:
                user_dict = data["users"]
                for u in user_dict.values():
                    self._upsert_user_row(u)
                _logger.info(f" users 적용 완료 (Dict, {len(user_dict)}명)")
            elif isinstance(data, list):
                for item in data:
                    self._upsert_user_row(item)
                _logger.info(f" users 적용 완료 (List, {len(data)}명)")
        except Exception as e:
            _logger.warning(f" users INSERT 실패: {e}")

    def _upsert_user_row(self, u: dict):
        """단일 유저 Safe UPSERT — Gist 관리 필드만 갱신, 운영 데이터 보존"""
        join_dt_str = u.get('created_at', u.get('join_date'))
        join_dt = None
        try:
            if join_dt_str:
                clean_str = str(join_dt_str)[:19].replace("T", " ")
                join_dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        expire_val = u.get('prime_expire_date')
        role = u.get('role', 'free')
        if not expire_val and role in ['prime', 'pro'] and join_dt:
            expire_val = join_dt + timedelta(days=7)
        vals = [
            u.get('login_id', u.get('id')),
            u.get('password_hash', u.get('password')),
            u.get('salt'), u.get('nickname'), role, join_dt_str,
            u.get('last_login'), u.get('is_banned', False),
            u.get('security_q_idx', 0), u.get('security_a_hash'),
            u.get('session_token'), expire_val,
        ]
        self.execute_safe("""
            INSERT INTO users 
            (id, password, salt, nickname, role, join_date, last_login, 
             is_banned, security_q_idx, security_a_hash, session_token, prime_expire_date) 
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                password = EXCLUDED.password,
                salt = EXCLUDED.salt,
                nickname = EXCLUDED.nickname,
                role = EXCLUDED.role,
                join_date = COALESCE(users.join_date, EXCLUDED.join_date),
                is_banned = EXCLUDED.is_banned,
                security_q_idx = EXCLUDED.security_q_idx,
                security_a_hash = EXCLUDED.security_a_hash,
                prime_expire_date = EXCLUDED.prime_expire_date
        """, vals)

    def _insert_gist_inquiries(self, data):
        if not data or not isinstance(data, list):
            return
        try:
            for item in data:
                vals = [
                    item.get('id'), item.get('nickname'), item.get('title'),
                    item.get('content'), item.get('created_at'),
                ]
                self.execute_safe("INSERT INTO inquiries VALUES (?,?,?,?,?)", vals)
            _logger.info(f" inquiries 적용 완료 ({len(data)}건)")
        except Exception as e:
            _logger.warning(f" inquiries INSERT 실패: {e}")

    def _load_users_from_gist(self):
        self._load_gist_to_table(USER_DB_FILE, "users", 12)

    def _load_inquiries_from_gist(self):
        self._load_gist_to_table(INQUIRY_DB_FILE, "inquiries", 5)

    def _load_gist_to_table(self, filename, tablename, col_count):
        if not GIST_ID or not GIST_TOKEN:
            _logger.warning(f" [Gist] {tablename} 로드 스킵 — Gist 인증 키 없음")
            return
        try:
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            resp = requests.get(url, headers=headers, timeout=5)

            if resp.status_code != 200:
                _logger.error(f" [Gist] API 응답 실패: {resp.status_code} — {resp.text[:200]}")
                return

            files = resp.json().get("files", {})
            if filename not in files:
                _logger.warning(f" [Gist] '{filename}' 파일이 Gist에 없음. 존재하는 파일: {list(files.keys())}")
                return

            content = files[filename]["content"]
            data = json.loads(content)
            if not data:
                _logger.warning(f" [Gist] '{filename}' 데이터가 비어있음")
                return

            if tablename == 'users':
                if isinstance(data, dict) and "users" in data:
                    user_dict = data["users"]
                    for u in user_dict.values():
                        self._upsert_user_row(u)
                    _logger.info(f" {tablename} 복원 완료 (Dict Format, {len(user_dict)}명)")
                elif isinstance(data, list):
                    for item in data:
                        self._upsert_user_row(item)
                    _logger.info(f" {tablename} 로드 완료 (List Format, {len(data)}명)")
                else:
                    _logger.warning(f" [Gist] {filename} 데이터 형식 미지원: {type(data).__name__}")

            elif tablename == 'inquiries':
                if isinstance(data, list):
                    for item in data:
                        vals = [
                            item.get('id'), item.get('nickname'), item.get('title'),
                            item.get('content'), item.get('created_at')
                        ]
                        self.execute_safe("INSERT INTO inquiries VALUES (?,?,?,?,?)", vals)
                    _logger.info(f" {tablename} 로드 완료 ({len(data)}건)")

        except Exception as e:
            _logger.error(f" [Gist] {tablename} 로드 실패: {e}")

    # ═══════════════════════════════════════════
    #  Gist 업로드 — Fire & Forget (UI freeze 제거)
    # ═══════════════════════════════════════════

    def _sync_table_to_gist(self, tablename, filename):
        """[v5.0 #2] Gist 업로드를 백그라운드 스레드로 분리

        Before: requests.patch(5초 타임아웃) → 전체 UI 멈춤
        After:  DB 읽기(락 내) → 스레드로 HTTP 전송 (Fire & Forget)
        """
        if not GIST_ID or not GIST_TOKEN:
            return

        # ① DB 읽기는 execute_safe (짧은 락)
        try:
            result = self.execute_safe(f"SELECT * FROM {tablename}")
            df = result.fetchdf()
            for col in df.columns:
                if df[col].dtype == 'object' or 'date' in str(df[col].dtype):
                    df[col] = df[col].astype(str).replace('NaT', None).replace('nan', None)
            json_str = df.to_json(orient='records', force_ascii=False, indent=2)
        except Exception as e:
            _logger.warning(f" {tablename} Gist 동기화 읽기 실패: {e}")
            return

        # ② HTTP 전송은 백그라운드 스레드 (Fire & Forget)
        def _upload():
            try:
                url = f"https://api.github.com/gists/{GIST_ID}"
                headers = {"Authorization": f"token {GIST_TOKEN}"}
                payload = {"files": {filename: {"content": json_str}}}
                resp = requests.patch(url, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    _logger.debug(f" Gist 업로드 완료: {tablename}")
                else:
                    _logger.warning(f" Gist 업로드 실패 ({resp.status_code}): {tablename}")
            except Exception as e:
                _logger.warning(f" Gist 업로드 에러: {e}")

        threading.Thread(target=_upload, daemon=True, name=f"gist-upload-{tablename}").start()

    # ═══════════════════════════════════════════
    #  User Methods — 전부 execute_safe 경유
    # ═══════════════════════════════════════════

    def register_user(self, email, pw_hash, salt, nickname, q_idx, a_hash):
        try:
            check = self.execute_safe("SELECT id FROM users WHERE id = ?", [email]).fetchone()
            if check:
                return False, "이미 존재하는 이메일입니다."

            now = datetime.now()
            role = 'free'
            expire = None

            self.execute_safe("""
                INSERT INTO users 
                (id, password, salt, nickname, role, join_date, last_login, 
                 is_banned, security_q_idx, security_a_hash, session_token, prime_expire_date) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [email, pw_hash, salt, nickname, role, now, now, False, q_idx, a_hash, "token", expire])

            self._sync_table_to_gist("users", USER_DB_FILE)
            return True, "가입 완료! (체험권이 필요하시면 '문의 게시판'에 신청해주세요)"

        except Exception as e:
            return False, f"DB Error: {e}"

    def get_user_by_id(self, email):
        try:
            cols = [x[0] for x in self.execute_safe("DESCRIBE users").fetchall()]
            row = self.execute_safe("SELECT * FROM users WHERE id = ?", [email]).fetchone()
            return dict(zip(cols, row)) if row else None
        except Exception:
            return None

    def update_login_timestamp(self, email):
        self.execute_safe("UPDATE users SET last_login = ? WHERE id = ?", [datetime.now(), email])
        self._sync_table_to_gist("users", USER_DB_FILE)

    def update_user_password(self, email, pw_hash, salt):
        try:
            self.execute_safe(
                "UPDATE users SET password = ?, salt = ?, login_fail_count = 0, lock_until = NULL WHERE id = ?",
                [pw_hash, salt, email]
            )
            self._sync_table_to_gist("users", USER_DB_FILE)
            return True
        except Exception as e:
            _logger.error(f" 비밀번호 업데이트 실패: {e}")
            return False

    # --- Rate Limit ---
    def record_login_failure(self, email):
        try:
            curr = self.execute_safe("SELECT login_fail_count FROM users WHERE id = ?", [email]).fetchone()
            if not curr:
                return

            new_count = curr[0] + 1
            lock_time = None
            if new_count >= 5:
                lock_time = datetime.now() + timedelta(minutes=10)
                _logger.warning(f"🔒 {email} 계정 10분 잠금 발동")

            self.execute_safe(
                "UPDATE users SET login_fail_count = ?, lock_until = ? WHERE id = ?",
                [new_count, lock_time, email]
            )
            self._sync_table_to_gist("users", USER_DB_FILE)
        except Exception as e:
            _logger.warning(f" 실패 기록 오류: {e}")

    def get_login_failures(self, email):
        try:
            res = self.execute_safe("SELECT login_fail_count, lock_until FROM users WHERE id = ?", [email]).fetchone()
            if res:
                return res[0], res[1].replace(tzinfo=timezone.utc) if res[1] else None
            return 0, None
        except Exception:
            return 0, None

    def reset_login_failures(self, email):
        self.execute_safe("UPDATE users SET login_fail_count = 0, lock_until = NULL WHERE id = ?", [email])
        self._sync_table_to_gist("users", USER_DB_FILE)

    # --- Admin Methods ---
    def get_all_users(self):
        try:
            cols = [x[0] for x in self.execute_safe("DESCRIBE users").fetchall()]
            rows = self.execute_safe("SELECT * FROM users").fetchall()
            result = []
            for r in rows:
                d = dict(zip(cols, r))
                d['login_id'] = d['id']
                result.append(d)
            return result
        except Exception:
            return []

    def update_user_role(self, email, new_role):
        self.execute_safe("UPDATE users SET role = ? WHERE id = ?", [new_role, email])
        self._sync_table_to_gist("users", USER_DB_FILE)
        return True

    def toggle_user_ban(self, email):
        curr = self.execute_safe("SELECT is_banned FROM users WHERE id = ?", [email]).fetchone()
        if not curr:
            return False, "유저 없음"
        new_stat = not curr[0]
        self.execute_safe("UPDATE users SET is_banned = ? WHERE id = ?", [new_stat, email])
        self._sync_table_to_gist("users", USER_DB_FILE)
        return True, f"{'차단' if new_stat else '해제'} 완료"

    def update_user_subscription(self, email, role, expire_date_str):
        try:
            exp_dt = datetime.strptime(expire_date_str, "%Y-%m-%d")
            self.execute_safe("UPDATE users SET role = ?, prime_expire_date = ? WHERE id = ?",
                              [role, exp_dt, email])
            self._sync_table_to_gist("users", USER_DB_FILE)
        except Exception:
            pass

    def grant_all_users_trial(self, days=7):
        try:
            new_expire = datetime.now() + timedelta(days=days)
            self.execute_safe(
                "UPDATE users SET role = 'prime', prime_expire_date = ? WHERE role != 'admin'",
                [new_expire]
            )
            self._sync_table_to_gist("users", USER_DB_FILE)
            return True, f"모든 회원(관리자 제외)에게 {days}일 Prime 권한이 부여되었습니다."
        except Exception as e:
            return False, f"DB Error: {e}"

    # --- Inquiry Methods ---
    def get_all_inquiries(self):
        try:
            cols = [x[0] for x in self.execute_safe("DESCRIBE inquiries").fetchall()]
            rows = self.execute_safe("SELECT * FROM inquiries").fetchall()
            result = []
            for r in rows:
                d = dict(zip(cols, r))
                d['email'] = d['id']
                result.append(d)
            return result
        except Exception:
            return []

    def save_recommendations(self, df, trade_ymd=None):
        """일일 추천 종목 결과 저장 — [v5.0 #1,#3] execute_safe + SQL 인젝션 제거"""
        if df is None or df.empty:
            return

        try:
            # 1. 테이블 스키마 체크
            try:
                table_info = self.execute_safe("PRAGMA table_info(daily_recommend)").fetchall()
                if len(table_info) > 0 and len(table_info) != 7:
                    _logger.warning(f" 스키마 불일치 감지. 테이블을 재생성합니다.")
                    self.execute_safe("DROP TABLE daily_recommend")
            except Exception:
                pass

            self.execute_safe("""
                CREATE TABLE IF NOT EXISTS daily_recommend (
                    trade_date VARCHAR,
                    code VARCHAR,
                    name VARCHAR,
                    close_price DOUBLE,
                    display_score DOUBLE,
                    final_score DOUBLE,
                    ai_comment VARCHAR
                )
            """)

            # 2. 데이터 전처리
            save_df = df.copy()

            if trade_ymd:
                s_ymd = str(trade_ymd)
                formatted_date = f"{s_ymd[:4]}-{s_ymd[4:6]}-{s_ymd[6:]}" if (len(s_ymd) == 8 and s_ymd.isdigit()) else s_ymd
                save_df['trade_date'] = formatted_date
            elif '기준일' in save_df.columns:
                save_df['trade_date'] = save_df['기준일'].astype(str)
            else:
                save_df['trade_date'] = datetime.now().strftime("%Y-%m-%d")

            save_df['code'] = save_df['종목코드'].astype(str).str.zfill(6)
            save_df['name'] = save_df['종목명']
            save_df['close_price'] = pd.to_numeric(save_df['종가'], errors='coerce').fillna(0)
            save_df['display_score'] = pd.to_numeric(save_df.get('DISPLAY_SCORE', save_df.get('LDY_SCORE', 0)), errors='coerce').fillna(0)
            save_df['final_score'] = pd.to_numeric(save_df.get('FINAL_SCORE', 0), errors='coerce').fillna(0)

            if 'AI_COMMENT' in save_df.columns:
                save_df['ai_comment'] = save_df['AI_COMMENT'].astype(str).fillna("")
            else:
                save_df['ai_comment'] = ""

            # 3. DB 삽입
            target_cols = ['trade_date', 'code', 'name', 'close_price', 'display_score', 'final_score', 'ai_comment']
            target_df = save_df[target_cols]

            if target_df.empty:
                return

            date_val = target_df['trade_date'].iloc[0]
            # [v5.0 #3] f-string SQL 인젝션 → 파라미터 바인딩
            self.execute_safe("DELETE FROM daily_recommend WHERE trade_date = ?", [date_val])

            # [v19.5 Fix 1] DuckDB는 Python 변수를 직접 참조 불가
            # → con.register()로 가상 테이블 등록 후 INSERT, 완료 후 해제
            self.conn.register("_tmp_target_df", target_df)
            try:
                self.execute_safe("INSERT INTO daily_recommend SELECT * FROM _tmp_target_df")
            finally:
                self.conn.unregister("_tmp_target_df")

            _logger.info(f" DB Saved: {len(target_df)} rows for {date_val}")

        except Exception as e:
            _logger.error(f" DB Save Failed: {e}")

    def save_snapshot(self, df, trade_ymd):
        """[v19.5] 가격 스냅샷 DB 저장 — register 패턴 사용"""
        if df is None or df.empty:
            return
        try:
            self.execute_safe("""
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    trade_date VARCHAR,
                    code VARCHAR,
                    name VARCHAR,
                    market VARCHAR,
                    close_price DOUBLE,
                    open_price DOUBLE,
                    low_price DOUBLE,
                    high_price DOUBLE
                )
            """)

            snap = df.copy()
            s_ymd = str(trade_ymd)
            formatted = f"{s_ymd[:4]}-{s_ymd[4:6]}-{s_ymd[6:]}" if len(s_ymd) == 8 and s_ymd.isdigit() else s_ymd
            snap["trade_date"] = formatted
            snap["code"] = snap["종목코드"].astype(str).str.zfill(6)
            snap["name"] = snap.get("종목명", "")
            snap["market"] = snap.get("시장", "")
            snap["close_price"] = pd.to_numeric(snap.get("종가", 0), errors="coerce").fillna(0)
            snap["open_price"] = pd.to_numeric(snap.get("시가", 0), errors="coerce").fillna(0)
            snap["low_price"] = pd.to_numeric(snap.get("저가", 0), errors="coerce").fillna(0)
            snap["high_price"] = pd.to_numeric(snap.get("고가", 0), errors="coerce").fillna(0)

            target_cols = ["trade_date", "code", "name", "market",
                           "close_price", "open_price", "low_price", "high_price"]
            snap_db = snap[target_cols]

            self.conn.register("_tmp_snap", snap_db)
            try:
                self.execute_safe("DELETE FROM price_snapshots WHERE trade_date = ?", [formatted])
                self.execute_safe("INSERT INTO price_snapshots SELECT * FROM _tmp_snap")
            finally:
                self.conn.unregister("_tmp_snap")

            _logger.info(f" Snapshot Saved: {len(snap_db)} rows for {formatted}")
        except Exception as e:
            _logger.error(f" Snapshot Save Failed: {e}")

    def save_inquiries(self, items):
        self.execute_safe("DELETE FROM inquiries")
        if items:
            for i in items:
                self.execute_safe(
                    "INSERT INTO inquiries VALUES (?, ?, ?, ?, ?)",
                    [i.get('email', i.get('id')), i.get('nickname'), i.get('title'),
                     i.get('content'), i.get('created_at')]
                )
        self._sync_table_to_gist("inquiries", INQUIRY_DB_FILE)
        return True

    def close(self):
        """DB 연결 종료"""
        try:
            self.conn.close()
        except Exception as e:
            _logger.warning(f" DB Close Error: {e}")


# ═══════════════════════════════════════════════════
#  Thread-Safe 싱글톤 + TTL 기반 갱신
# ═══════════════════════════════════════════════════

_db_instance = None
_db_lock = threading.Lock()
_db_init_time = 0.0
_DB_TTL_SECONDS = 600

_bg_refresh_running = False


def get_db(force_refresh: bool = False) -> LDYDBManager:
    """Thread-safe 싱글톤 DB 인스턴스."""
    global _db_instance, _db_init_time

    now = _time.monotonic()

    if _db_instance is not None and not force_refresh:
        if (now - _db_init_time) > _DB_TTL_SECONDS:
            _schedule_background_refresh()
        return _db_instance

    with _db_lock:
        if _db_instance is None or force_refresh:
            if _db_instance is not None:
                try:
                    _db_instance.close()
                except Exception:
                    pass
            _db_instance = LDYDBManager()
            _db_init_time = _time.monotonic()

    return _db_instance


def _schedule_background_refresh():
    """백그라운드 스레드로 Gist 갱신"""
    global _bg_refresh_running, _db_init_time
    if _bg_refresh_running:
        return

    def _do_refresh():
        global _bg_refresh_running, _db_init_time
        try:
            _bg_refresh_running = True
            if _db_instance is None:
                return
            downloaded = _db_instance._download_gist_data()
            if downloaded:
                with _db_lock:
                    _db_instance._apply_gist_data(downloaded)
                    _db_init_time = _time.monotonic()
            else:
                _db_init_time = _time.monotonic()
        except Exception as e:
            _logger.warning(f" 백그라운드 Gist 갱신 실패 (기존 데이터 유지): {e}")
            _db_init_time = _time.monotonic()
        finally:
            _bg_refresh_running = False

    threading.Thread(target=_do_refresh, daemon=True, name="gist-refresh").start()


def start_gist_background_loader(interval_sec: int = 600):
    """NiceGUI app.on_startup에서 호출"""
    def _loop():
        db = get_db()
        if db:
            db.ensure_gist_loaded()
        while True:
            _time.sleep(interval_sec)
            _schedule_background_refresh()

    threading.Thread(target=_loop, daemon=True, name="gist-bg-loader").start()


def _reset_db_singleton():
    """테스트용: 싱글톤 초기화"""
    global _db_instance, _db_init_time
    with _db_lock:
        if _db_instance is not None:
            try:
                _db_instance.close()
            except Exception:
                pass
        _db_instance = None
        _db_init_time = 0.0
