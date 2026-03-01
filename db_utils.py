# db_utils.py — SQLite WAL + 배치 Gist 동기화 (v6.0)
# ═══════════════════════════════════════════════════
# [v6.0] DuckDB → SQLite WAL 마이그레이션
#   #1 OLTP 워크로드(유저 세션/로그인) → SQLite WAL 모드
#   #2 Gist 동기화 폭탄 제거 → 디바운스 배치 (60초 쿨다운)
#   #3 DuckDB는 OLAP 전용 (daily_recommend, price_snapshots)으로 분리 유지
# ═══════════════════════════════════════════════════

import sqlite3
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

# ───────────────────────────────────────────
#  Gist 설정
# ───────────────────────────────────────────
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

# ═══════════════════════════════════════════
#  [v6.0 핵심] Gist 디바운스 동기화
#  - 트랜잭션마다 즉시 쏘지 않고, 변경 플래그만 세움
#  - 백그라운드 루프가 60초 간격으로 dirty 테이블만 업로드
# ═══════════════════════════════════════════

_GIST_SYNC_INTERVAL = 60  # 초

class _GistSyncManager:
    """변경 감지 → 배치 업로드 (Rate Limit 폭탄 방지 + 실패 시 자동 재시도)"""

    def __init__(self):
        self._dirty: set[str] = set()  # {"users", "inquiries"}
        self._lock = threading.Lock()
        self._running = False
        self._consecutive_fails: dict[str, int] = {}  # 테이블별 연속 실패 횟수
        self._MAX_RETRIES = 5  # 연속 실패 상한 (초과 시 경고 후 포기)

    def mark_dirty(self, table_name: str):
        with self._lock:
            self._dirty.add(table_name)

    def start(self, db_manager: "LDYDBManager"):
        if self._running:
            return
        self._running = True

        def _loop():
            while True:
                _time.sleep(_GIST_SYNC_INTERVAL)
                with self._lock:
                    tables = list(self._dirty)
                    self._dirty.clear()

                for tbl in tables:
                    filename = USER_DB_FILE if tbl == "users" else INQUIRY_DB_FILE
                    success = db_manager._do_gist_upload(tbl, filename)

                    if success:
                        # 성공 → 연속 실패 카운터 초기화
                        self._consecutive_fails.pop(tbl, None)
                    else:
                        # 실패 → dirty 플래그 복원 (다음 주기에 재시도)
                        fail_count = self._consecutive_fails.get(tbl, 0) + 1
                        self._consecutive_fails[tbl] = fail_count

                        if fail_count <= self._MAX_RETRIES:
                            with self._lock:
                                self._dirty.add(tbl)
                            _logger.warning(
                                f" [Gist] {tbl} 업로드 실패 ({fail_count}/{self._MAX_RETRIES}) "
                                f"→ 다음 주기에 재시도"
                            )
                        else:
                            _logger.error(
                                f" [Gist] {tbl} 업로드 {self._MAX_RETRIES}회 연속 실패 "
                                f"→ 재시도 중단 (수동 확인 필요)"
                            )
                            self._consecutive_fails.pop(tbl, None)

        t = threading.Thread(target=_loop, daemon=True, name="gist-batch-sync")
        t.start()
        _logger.info(f" [Gist] 배치 동기화 시작 (간격: {_GIST_SYNC_INTERVAL}초, 최대 재시도: {self._MAX_RETRIES}회)")


_gist_sync = _GistSyncManager()


class LDYDBManager:
    """
    [v6.0] 이중 DB 아키텍처:
      - SQLite (WAL) : users, inquiries — OLTP (잦은 읽기/쓰기)
      - DuckDB        : daily_recommend, price_snapshots — OLAP (배치 분석)
    """
    _SQLITE_PATH = "ldy_users.db"
    _DUCKDB_PATH = "ldy_trader.db"

    def __init__(self):
        # ── SQLite (OLTP) ──
        self._sqlite = sqlite3.connect(
            self._SQLITE_PATH,
            check_same_thread=False,
            timeout=30,
        )
        self._sqlite.execute("PRAGMA journal_mode=WAL")
        self._sqlite.execute("PRAGMA busy_timeout=5000")
        self._sqlite_lock = threading.Lock()

        # ── DuckDB (OLAP) ──
        self._duck = duckdb.connect(self._DUCKDB_PATH)
        self._duck_lock = threading.Lock()

        self._gist_loaded = False
        self._init_tables()

    # ═══════════════════════════════════════════
    #  Thread-Safe 실행 메서드
    # ═══════════════════════════════════════════

    def _exec_sqlite(self, query: str, params=None, fetch=False):
        """SQLite Thread-safe 실행 (WAL이라 읽기 동시성 ↑)"""
        with self._sqlite_lock:
            cur = self._sqlite.cursor()
            try:
                cur.execute(query, params or [])
                if fetch:
                    return cur.fetchall()
                self._sqlite.commit()
                return cur
            except sqlite3.OperationalError as e:
                _logger.warning(f"SQLite 에러: {e}, 쿼리: {query[:80]}")
                raise

    def _exec_sqlite_one(self, query: str, params=None):
        """단일 행 조회"""
        with self._sqlite_lock:
            cur = self._sqlite.cursor()
            cur.execute(query, params or [])
            return cur.fetchone()

    def execute_safe(self, query, params=None):
        """DuckDB Thread-safe 실행 (OLAP용)"""
        with self._duck_lock:
            try:
                return self._duck.execute(query, params) if params else self._duck.execute(query)
            except (duckdb.ConnectionException, duckdb.IOException) as e:
                _logger.warning(f"DuckDB 재연결: {e}")
                try:
                    self._duck.close()
                except Exception:
                    pass
                self._duck = duckdb.connect(self._DUCKDB_PATH)
                return self._duck.execute(query, params) if params else self._duck.execute(query)

    # ═══════════════════════════════════════════
    #  테이블 초기화
    # ═══════════════════════════════════════════

    def _init_tables(self):
        # ── SQLite: OLTP 테이블 ──
        self._exec_sqlite("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                password TEXT,
                salt TEXT,
                nickname TEXT,
                role TEXT DEFAULT 'free',
                join_date TEXT,
                last_login TEXT,
                is_banned INTEGER DEFAULT 0,
                security_q_idx INTEGER DEFAULT 0,
                security_a_hash TEXT,
                session_token TEXT,
                prime_expire_date TEXT,
                login_fail_count INTEGER DEFAULT 0,
                lock_until TEXT
            )
        """)
        self._exec_sqlite("""
            CREATE TABLE IF NOT EXISTS inquiries (
                id TEXT,
                nickname TEXT,
                title TEXT,
                content TEXT,
                created_at TEXT
            )
        """)

        # ── DuckDB: OLAP 테이블 ──
        self.execute_safe("""
            CREATE TABLE IF NOT EXISTS daily_recommend (
                trade_date VARCHAR, code VARCHAR, name VARCHAR,
                close_price DOUBLE, display_score DOUBLE,
                final_score DOUBLE, ai_comment VARCHAR
            )
        """)

        try:
            cols = [r[1] for r in self.execute_safe(
                "PRAGMA table_info(price_snapshots)").fetchall()]
            if cols and "snap_date" in cols and "trade_date" not in cols:
                _logger.info(" 🔄 price_snapshots 스키마 마이그레이션: snap_date → trade_date")
                self.execute_safe("ALTER TABLE price_snapshots RENAME COLUMN snap_date TO trade_date")
        except Exception as _mig_err:
            try:
                self.execute_safe("DROP TABLE IF EXISTS price_snapshots")
            except Exception:
                pass

        self.execute_safe("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                trade_date VARCHAR, code VARCHAR, name VARCHAR,
                market VARCHAR, close_price DOUBLE, open_price DOUBLE,
                low_price DOUBLE, high_price DOUBLE
            )
        """)
        self.execute_safe("CREATE INDEX IF NOT EXISTS idx_rec_date ON daily_recommend (trade_date)")
        self.execute_safe("CREATE INDEX IF NOT EXISTS idx_trade_date ON price_snapshots (trade_date)")

    # ═══════════════════════════════════════════
    #  Gist 로드 / 업로드
    # ═══════════════════════════════════════════

    def ensure_gist_loaded(self):
        if self._gist_loaded:
            return
        try:
            self._load_users_from_gist()
            self._load_inquiries_from_gist()
            self._gist_loaded = True
        except Exception as e:
            _logger.warning(f"Gist 초기 로드 실패 (DB는 정상): {e}")

    def _download_gist_data(self) -> dict:
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
        if USER_DB_FILE in downloaded:
            self._insert_gist_users(downloaded[USER_DB_FILE])
        if INQUIRY_DB_FILE in downloaded:
            self._insert_gist_inquiries(downloaded[INQUIRY_DB_FILE])

    def _insert_gist_users(self, data):
        if not data:
            return
        try:
            if isinstance(data, dict) and "users" in data:
                for u in data["users"].values():
                    self._upsert_user_row(u)
                _logger.info(f" users 적용 완료 (Dict, {len(data['users'])}명)")
            elif isinstance(data, list):
                for item in data:
                    self._upsert_user_row(item)
                _logger.info(f" users 적용 완료 (List, {len(data)}명)")
        except Exception as e:
            _logger.warning(f" users INSERT 실패: {e}")

    def _upsert_user_row(self, u: dict):
        """SQLite UPSERT (ON CONFLICT)"""
        join_dt_str = u.get('created_at', u.get('join_date'))
        expire_val = u.get('prime_expire_date')
        role = u.get('role', 'free')
        if not expire_val and role in ['prime', 'pro'] and join_dt_str:
            try:
                clean_str = str(join_dt_str)[:19].replace("T", " ")
                join_dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
                expire_val = (join_dt + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        vals = (
            u.get('login_id', u.get('id')),
            u.get('password_hash', u.get('password')),
            u.get('salt'), u.get('nickname'), role,
            str(join_dt_str) if join_dt_str else None,
            str(u.get('last_login')) if u.get('last_login') else None,
            1 if u.get('is_banned') else 0,
            u.get('security_q_idx', 0), u.get('security_a_hash'),
            u.get('session_token'),
            str(expire_val) if expire_val else None,
        )
        self._exec_sqlite("""
            INSERT INTO users
            (id, password, salt, nickname, role, join_date, last_login,
             is_banned, security_q_idx, security_a_hash, session_token, prime_expire_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                password = excluded.password,
                salt = excluded.salt,
                nickname = excluded.nickname,
                role = excluded.role,
                join_date = COALESCE(users.join_date, excluded.join_date),
                is_banned = excluded.is_banned,
                security_q_idx = excluded.security_q_idx,
                security_a_hash = excluded.security_a_hash,
                prime_expire_date = excluded.prime_expire_date
        """, vals)

    def _insert_gist_inquiries(self, data):
        if not data or not isinstance(data, list):
            return
        try:
            for item in data:
                self._exec_sqlite(
                    "INSERT INTO inquiries VALUES (?,?,?,?,?)",
                    (item.get('id'), item.get('nickname'), item.get('title'),
                     item.get('content'), item.get('created_at'))
                )
            _logger.info(f" inquiries 적용 완료 ({len(data)}건)")
        except Exception as e:
            _logger.warning(f" inquiries INSERT 실패: {e}")

    def _load_users_from_gist(self):
        self._load_gist_to_table(USER_DB_FILE, "users")

    def _load_inquiries_from_gist(self):
        self._load_gist_to_table(INQUIRY_DB_FILE, "inquiries")

    def _load_gist_to_table(self, filename, tablename):
        if not GIST_ID or not GIST_TOKEN:
            _logger.warning(f" [Gist] {tablename} 로드 스킵 — Gist 인증 키 없음")
            return
        try:
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code != 200:
                _logger.error(f" [Gist] API 응답 실패: {resp.status_code}")
                return

            files = resp.json().get("files", {})
            if filename not in files:
                _logger.warning(f" [Gist] '{filename}' 파일이 Gist에 없음")
                return

            data = json.loads(files[filename]["content"])
            if not data:
                return

            if tablename == 'users':
                self._insert_gist_users(data)
            elif tablename == 'inquiries':
                self._insert_gist_inquiries(data)

        except Exception as e:
            _logger.error(f" [Gist] {tablename} 로드 실패: {e}", exc_info=True)

    # ═══════════════════════════════════════════
    #  [v6.0] Gist 업로드 — 디바운스 배치
    # ═══════════════════════════════════════════

    def _mark_gist_dirty(self, tablename: str):
        """[v6.0] 즉시 업로드 대신 dirty 플래그만 세움 → 배치 루프가 처리"""
        _gist_sync.mark_dirty(tablename)

    def _do_gist_upload(self, tablename: str, filename: str) -> bool:
        """실제 업로드 (배치 루프에서 호출). 성공 시 True, 실패 시 False."""
        if not GIST_ID or not GIST_TOKEN:
            return True  # Gist 미설정은 실패가 아님
        try:
            rows = self._exec_sqlite(f"SELECT * FROM {tablename}", fetch=True)
            if tablename == "users":
                cols = ["id", "password", "salt", "nickname", "role", "join_date",
                        "last_login", "is_banned", "security_q_idx", "security_a_hash",
                        "session_token", "prime_expire_date", "login_fail_count", "lock_until"]
            else:
                cols = ["id", "nickname", "title", "content", "created_at"]

            data = [dict(zip(cols, r)) for r in rows]
            json_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)

            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            payload = {"files": {filename: {"content": json_str}}}
            resp = requests.patch(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                _logger.debug(f" Gist 배치 업로드 완료: {tablename}")
                return True
            else:
                _logger.warning(f" Gist 배치 업로드 실패 ({resp.status_code}): {tablename}")
                return False
        except Exception as e:
            _logger.warning(f" Gist 배치 업로드 에러: {e}", exc_info=True)
            return False

    # ═══════════════════════════════════════════
    #  User Methods — SQLite OLTP
    # ═══════════════════════════════════════════

    def register_user(self, email, pw_hash, salt, nickname, q_idx, a_hash):
        try:
            check = self._exec_sqlite_one("SELECT id FROM users WHERE id = ?", (email,))
            if check:
                return False, "이미 존재하는 이메일입니다."

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._exec_sqlite("""
                INSERT INTO users
                (id, password, salt, nickname, role, join_date, last_login,
                 is_banned, security_q_idx, security_a_hash, session_token, prime_expire_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (email, pw_hash, salt, nickname, 'free', now, now, 0, q_idx, a_hash, "token", None))

            self._mark_gist_dirty("users")
            return True, "가입 완료! (체험권이 필요하시면 '문의 게시판'에 신청해주세요)"

        except Exception as e:
            _logger.error(f"회원가입 실패: {e}", exc_info=True)
            return False, f"DB Error: {e}"

    def get_user_by_id(self, email):
        try:
            row = self._exec_sqlite_one("SELECT * FROM users WHERE id = ?", (email,))
            if not row:
                return None
            cols = ["id", "password", "salt", "nickname", "role", "join_date",
                    "last_login", "is_banned", "security_q_idx", "security_a_hash",
                    "session_token", "prime_expire_date", "login_fail_count", "lock_until"]
            d = dict(zip(cols, row))
            # bool 변환 (SQLite는 0/1)
            d['is_banned'] = bool(d.get('is_banned', 0))
            return d
        except Exception as e:
            _logger.warning(f"유저 조회 실패 ({email}): {e}", exc_info=True)
            return None

    def update_login_timestamp(self, email):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._exec_sqlite("UPDATE users SET last_login = ? WHERE id = ?", (now, email))
        self._mark_gist_dirty("users")  # ← 즉시 업로드 대신 dirty 마킹

    def update_user_password(self, email, pw_hash, salt):
        try:
            self._exec_sqlite(
                "UPDATE users SET password = ?, salt = ?, login_fail_count = 0, lock_until = NULL WHERE id = ?",
                (pw_hash, salt, email)
            )
            self._mark_gist_dirty("users")
            return True
        except Exception as e:
            _logger.error(f" 비밀번호 업데이트 실패: {e}", exc_info=True)
            return False

    def record_login_failure(self, email):
        try:
            curr = self._exec_sqlite_one("SELECT login_fail_count FROM users WHERE id = ?", (email,))
            if not curr:
                return

            new_count = (curr[0] or 0) + 1
            lock_time = None
            if new_count >= 5:
                lock_time = (datetime.now() + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
                _logger.warning(f"🔒 {email} 계정 10분 잠금 발동")

            self._exec_sqlite(
                "UPDATE users SET login_fail_count = ?, lock_until = ? WHERE id = ?",
                (new_count, lock_time, email)
            )
            self._mark_gist_dirty("users")
        except Exception as e:
            _logger.warning(f" 실패 기록 오류: {e}", exc_info=True)

    def get_login_failures(self, email):
        try:
            res = self._exec_sqlite_one(
                "SELECT login_fail_count, lock_until FROM users WHERE id = ?", (email,))
            if res:
                lock = None
                if res[1]:
                    try:
                        lock = datetime.strptime(res[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    except Exception:
                        pass
                return res[0] or 0, lock
            return 0, None
        except Exception:
            return 0, None

    def reset_login_failures(self, email):
        self._exec_sqlite(
            "UPDATE users SET login_fail_count = 0, lock_until = NULL WHERE id = ?", (email,))
        self._mark_gist_dirty("users")

    # --- Admin Methods ---
    def get_all_users(self):
        try:
            cols = ["id", "password", "salt", "nickname", "role", "join_date",
                    "last_login", "is_banned", "security_q_idx", "security_a_hash",
                    "session_token", "prime_expire_date", "login_fail_count", "lock_until"]
            rows = self._exec_sqlite("SELECT * FROM users", fetch=True)
            result = []
            for r in rows:
                d = dict(zip(cols, r))
                d['login_id'] = d['id']
                d['is_banned'] = bool(d.get('is_banned', 0))
                result.append(d)
            return result
        except Exception as e:
            _logger.warning(f"전체 유저 조회 실패: {e}", exc_info=True)
            return []

    def update_user_role(self, email, new_role):
        self._exec_sqlite("UPDATE users SET role = ? WHERE id = ?", (new_role, email))
        self._mark_gist_dirty("users")
        return True

    def toggle_user_ban(self, email):
        curr = self._exec_sqlite_one("SELECT is_banned FROM users WHERE id = ?", (email,))
        if not curr:
            return False, "유저 없음"
        new_stat = 0 if curr[0] else 1
        self._exec_sqlite("UPDATE users SET is_banned = ? WHERE id = ?", (new_stat, email))
        self._mark_gist_dirty("users")
        return True, f"{'차단' if new_stat else '해제'} 완료"

    def update_user_subscription(self, email, role, expire_date_str):
        try:
            self._exec_sqlite(
                "UPDATE users SET role = ?, prime_expire_date = ? WHERE id = ?",
                (role, expire_date_str, email)
            )
            self._mark_gist_dirty("users")
        except Exception as e:
            _logger.warning(f"구독 업데이트 실패: {e}", exc_info=True)

    def grant_all_users_trial(self, days=14):
        try:
            new_expire = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            self._exec_sqlite(
                "UPDATE users SET role = 'prime', prime_expire_date = ? WHERE role != 'admin'",
                (new_expire,)
            )
            self._mark_gist_dirty("users")
            return True, f"모든 회원(관리자 제외)에게 {days}일 Prime 권한이 부여되었습니다."
        except Exception as e:
            return False, f"DB Error: {e}"

    # --- Inquiry Methods ---
    def get_all_inquiries(self):
        try:
            cols = ["id", "nickname", "title", "content", "created_at"]
            rows = self._exec_sqlite("SELECT * FROM inquiries", fetch=True)
            result = []
            for r in rows:
                d = dict(zip(cols, r))
                d['email'] = d['id']
                result.append(d)
            return result
        except Exception as e:
            _logger.warning(f"문의 조회 실패: {e}", exc_info=True)
            return []

    def save_inquiries(self, items):
        self._exec_sqlite("DELETE FROM inquiries")
        if items:
            for i in items:
                self._exec_sqlite(
                    "INSERT INTO inquiries VALUES (?, ?, ?, ?, ?)",
                    (i.get('email', i.get('id')), i.get('nickname'), i.get('title'),
                     i.get('content'), i.get('created_at'))
                )
        self._mark_gist_dirty("inquiries")
        return True

    # ═══════════════════════════════════════════
    #  OLAP Methods — DuckDB (변경 없음)
    # ═══════════════════════════════════════════

    def save_recommendations(self, df, trade_ymd=None):
        if df is None or df.empty:
            return
        try:
            try:
                table_info = self.execute_safe("PRAGMA table_info(daily_recommend)").fetchall()
                if len(table_info) > 0 and len(table_info) != 7:
                    _logger.warning(" 스키마 불일치 감지. 테이블을 재생성합니다.")
                    self.execute_safe("DROP TABLE daily_recommend")
            except Exception:
                pass

            self.execute_safe("""
                CREATE TABLE IF NOT EXISTS daily_recommend (
                    trade_date VARCHAR, code VARCHAR, name VARCHAR,
                    close_price DOUBLE, display_score DOUBLE,
                    final_score DOUBLE, ai_comment VARCHAR
                )
            """)

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
            save_df['display_score'] = pd.to_numeric(
                save_df.get('DISPLAY_SCORE', save_df.get('LDY_SCORE', 0)), errors='coerce').fillna(0)
            save_df['final_score'] = pd.to_numeric(save_df.get('FINAL_SCORE', 0), errors='coerce').fillna(0)
            save_df['ai_comment'] = save_df['AI_COMMENT'].astype(str).fillna("") if 'AI_COMMENT' in save_df.columns else ""

            target_cols = ['trade_date', 'code', 'name', 'close_price', 'display_score', 'final_score', 'ai_comment']
            target_df = save_df[target_cols]
            if target_df.empty:
                return

            date_val = target_df['trade_date'].iloc[0]
            self.execute_safe("DELETE FROM daily_recommend WHERE trade_date = ?", [date_val])

            self._duck.register("_tmp_target_df", target_df)
            try:
                self.execute_safe("INSERT INTO daily_recommend SELECT * FROM _tmp_target_df")
            finally:
                self._duck.unregister("_tmp_target_df")

            _logger.info(f" DB Saved: {len(target_df)} rows for {date_val}")
        except Exception as e:
            _logger.error(f" DB Save Failed: {e}", exc_info=True)

    def save_snapshot(self, df, trade_ymd):
        if df is None or df.empty:
            return
        try:
            try:
                cols = [r[1] for r in self.execute_safe("PRAGMA table_info(price_snapshots)").fetchall()]
                if cols and "trade_date" not in cols:
                    self.execute_safe("DROP TABLE price_snapshots")
            except Exception:
                pass

            self.execute_safe("""
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    trade_date VARCHAR, code VARCHAR, name VARCHAR,
                    market VARCHAR, close_price DOUBLE, open_price DOUBLE,
                    low_price DOUBLE, high_price DOUBLE
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

            self._duck.register("_tmp_snap", snap_db)
            try:
                self.execute_safe("DELETE FROM price_snapshots WHERE trade_date = ?", [formatted])
                self.execute_safe("INSERT INTO price_snapshots SELECT * FROM _tmp_snap")
            finally:
                self._duck.unregister("_tmp_snap")

            _logger.info(f" Snapshot Saved: {len(snap_db)} rows for {formatted}")
        except Exception as e:
            _logger.error(f" Snapshot Save Failed: {e}", exc_info=True)

    def close(self):
        try:
            self._sqlite.close()
        except Exception as e:
            _logger.warning(f" SQLite Close Error: {e}")
        try:
            self._duck.close()
        except Exception as e:
            _logger.warning(f" DuckDB Close Error: {e}")


# ═══════════════════════════════════════════════════
#  Thread-Safe 싱글톤 + TTL 기반 갱신
# ═══════════════════════════════════════════════════

_db_instance = None
_db_lock = threading.Lock()
_db_init_time = 0.0
_DB_TTL_SECONDS = 600
_bg_refresh_running = False


def get_db(force_refresh: bool = False) -> LDYDBManager:
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
            _logger.warning(f" 백그라운드 Gist 갱신 실패: {e}", exc_info=True)
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
            _gist_sync.start(db)  # ← [v6.0] 배치 동기화 시작
        while True:
            _time.sleep(interval_sec)
            _schedule_background_refresh()

    threading.Thread(target=_loop, daemon=True, name="gist-bg-loader").start()


def _reset_db_singleton():
    global _db_instance, _db_init_time
    with _db_lock:
        if _db_instance is not None:
            try:
                _db_instance.close()
            except Exception:
                pass
        _db_instance = None
        _db_init_time = 0.0
