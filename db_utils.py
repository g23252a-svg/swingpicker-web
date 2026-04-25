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
# [v22 Step W+X] 결제 기록 별도 Gist 파일 — inquiries와 컬럼 매핑 분리
PAYMENT_DB_FILE = "payments_db.json"

# [v22 Step X] 테이블 → Gist 파일명 매핑
# Sync manager + load 양쪽에서 일관 사용
TABLE_TO_GIST_FILE = {
    "users": USER_DB_FILE,
    "inquiries": INQUIRY_DB_FILE,
    "payments": PAYMENT_DB_FILE,
}

# ═══════════════════════════════════════════
#  [v6.0 핵심] Gist 디바운스 동기화
#  - 트랜잭션마다 즉시 쏘지 않고, 변경 플래그만 세움
#  - 백그라운드 루프가 60초 간격으로 dirty 테이블만 업로드
# ═══════════════════════════════════════════

_GIST_SYNC_INTERVAL = 60  # 초

class _GistSyncManager:
    """변경 감지 → 배치 업로드 (Rate Limit 폭탄 방지 + 실패 시 자동 재시도)"""

    def __init__(self):
        self._dirty: set[str] = set()  # {"users", "inquiries", "payments"}
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
                    # [v22 Step X] 테이블별 파일명 정확한 매핑
                    filename = TABLE_TO_GIST_FILE.get(tbl)
                    if not filename:
                        _logger.warning(
                            f"[Gist] 알 수 없는 테이블 '{tbl}' 동기화 스킵 "
                            f"(허용: {list(TABLE_TO_GIST_FILE.keys())})"
                        )
                        continue
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
        """SQLite Thread-safe 실행 (WAL이라 읽기 동시성 ↑).

        [v3.7.27 Phase 1] 트랜잭션 안전성 강화:
          - 쓰기 실패 시 자동 rollback (이전엔 커밋 반쯤 된 채 raise)
          - finally에서 커서 명시적 close
          - DB 부분 손상 방지
        """
        with self._sqlite_lock:
            cur = self._sqlite.cursor()
            try:
                cur.execute(query, params or [])
                if fetch:
                    result = cur.fetchall()
                    return result
                self._sqlite.commit()
                return cur
            except sqlite3.OperationalError as e:
                # [v3.7.27] rollback으로 부분 쓰기 방지
                try:
                    self._sqlite.rollback()
                except Exception as _re:
                    _logger.warning(f"SQLite rollback 실패: {_re}")
                _logger.warning(f"SQLite 에러: {e}, 쿼리: {query[:80]}")
                raise
            except Exception as e:
                # [v3.7.27] 일반 예외에도 rollback
                try:
                    self._sqlite.rollback()
                except Exception as _re:
                    _logger.warning(f"SQLite rollback 실패: {_re}")
                _logger.error(f"SQLite 예상외 에러: {e}, 쿼리: {query[:80]}")
                raise
            finally:
                # [v3.7.27] 커서 누수 방지 (fetch 모드에서만 close 필요)
                if fetch:
                    try:
                        cur.close()
                    except Exception:
                        pass

    def _exec_sqlite_one(self, query: str, params=None):
        """단일 행 조회 (읽기 전용).

        [v3.7.27] 커서 명시적 close로 누수 방지.
        """
        with self._sqlite_lock:
            cur = self._sqlite.cursor()
            try:
                cur.execute(query, params or [])
                return cur.fetchone()
            finally:
                try:
                    cur.close()
                except Exception:
                    pass

    def execute_safe(self, query, params=None):
        """DuckDB Thread-safe 실행 (OLAP용).

        [v3.7.27] 에러 경로 정리 + 재연결 시 로깅 강화.
        """
        with self._duck_lock:
            try:
                return self._duck.execute(query, params) if params else self._duck.execute(query)
            except (duckdb.ConnectionException, duckdb.IOException) as e:
                _logger.warning(f"DuckDB 재연결 시도: {e}")
                try:
                    self._duck.close()
                except Exception:
                    pass
                try:
                    self._duck = duckdb.connect(self._DUCKDB_PATH)
                    return self._duck.execute(query, params) if params else self._duck.execute(query)
                except Exception as _re:
                    _logger.error(f"DuckDB 재연결 실패: {_re}")
                    raise

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
        # [v22 Step Y] inquiries 테이블 마이그레이션 — 기존 컬럼 5개 → 9개
        # PRIMARY KEY 추가 (중복 방지) + 답변/상태/카테고리 컬럼
        try:
            existing_cols = [r[1] for r in self._exec_sqlite(
                "PRAGMA table_info(inquiries)").fetchall()]
            if existing_cols and "inquiry_id" not in existing_cols:
                _logger.info(" 🔄 inquiries 스키마 마이그레이션: 컬럼 4개 추가")
                # 기존 데이터 백업
                old_rows = self._exec_sqlite(
                    "SELECT id, nickname, title, content, created_at FROM inquiries",
                    fetch=True
                ) or []
                # 기존 테이블 DROP
                self._exec_sqlite("DROP TABLE IF EXISTS inquiries")
        except Exception as _e:
            _logger.debug(f"inquiries 마이그레이션 체크 실패 (무시): {_e}")
            old_rows = []

        self._exec_sqlite("""
            CREATE TABLE IF NOT EXISTS inquiries (
                inquiry_id TEXT PRIMARY KEY,
                id TEXT,
                nickname TEXT,
                title TEXT,
                content TEXT,
                created_at TEXT,
                category TEXT DEFAULT 'general',
                status TEXT DEFAULT 'open',
                admin_reply TEXT DEFAULT '',
                admin_reply_at TEXT DEFAULT ''
            )
        """)
        self._exec_sqlite(
            "CREATE INDEX IF NOT EXISTS idx_inq_email ON inquiries (id)"
        )
        self._exec_sqlite(
            "CREATE INDEX IF NOT EXISTS idx_inq_status ON inquiries (status)"
        )
        self._exec_sqlite(
            "CREATE INDEX IF NOT EXISTS idx_inq_created ON inquiries (created_at)"
        )

        # 마이그레이션 데이터 복원 (created_at + content 해시로 inquiry_id 생성)
        try:
            if 'old_rows' in dir() and old_rows:
                import hashlib
                for r in old_rows:
                    email, nickname, title, content, created_at = r
                    # 안정적인 inquiry_id 생성: created_at + content 해시
                    seed = f"{created_at}|{title}|{content}|{email}"
                    inquiry_id = hashlib.sha256(seed.encode()).hexdigest()[:16]
                    self._exec_sqlite(
                        """INSERT OR IGNORE INTO inquiries
                        (inquiry_id, id, nickname, title, content, created_at, category, status)
                        VALUES (?, ?, ?, ?, ?, ?, 'general', 'open')""",
                        (inquiry_id, email, nickname, title, content, created_at)
                    )
                _logger.info(f"  ✅ inquiries 마이그레이션 완료: {len(old_rows)}건 → 중복 자동 제거")
        except Exception as _mig_e:
            _logger.warning(f"inquiries 마이그레이션 실패 (무시 가능): {_mig_e}")

        # [v22 Step W] 결제 기록 테이블 — orderId UNIQUE로 중복 방지
        # status: success / failed / amount_mismatch / duplicate / refunded
        self._exec_sqlite("""
            CREATE TABLE IF NOT EXISTS payments (
                order_id TEXT PRIMARY KEY,
                payment_key TEXT,
                email TEXT,
                plan TEXT,
                amount INTEGER,
                status TEXT,
                method TEXT,
                approved_at TEXT,
                receipt_url TEXT,
                created_at TEXT,
                error_message TEXT
            )
        """)
        self._exec_sqlite(
            "CREATE INDEX IF NOT EXISTS idx_payments_email ON payments (email)"
        )
        self._exec_sqlite(
            "CREATE INDEX IF NOT EXISTS idx_payments_status ON payments (status)"
        )

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
            # [v20.6.5] 컬럼 수 불일치 → DROP+재생성
            elif cols and len(cols) != 8:
                _logger.info(f" 🔄 price_snapshots 컬럼 수 불일치: {len(cols)} → 8, DROP+재생성")
                self.execute_safe("DROP TABLE IF EXISTS price_snapshots")
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
            # [v22 Step X] payments 테이블도 Gist에서 복구
            self._load_payments_from_gist()
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
            # [v22 Step X] payments 파일도 함께 다운로드
            for fname in [USER_DB_FILE, INQUIRY_DB_FILE, PAYMENT_DB_FILE]:
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
        # [v22 Step X] payments 데이터 적용
        if PAYMENT_DB_FILE in downloaded:
            self._insert_gist_payments(downloaded[PAYMENT_DB_FILE])

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
        """[v22 Step Y] inquiries Gist 데이터 적용 (UPSERT — inquiry_id PK).
        
        기존 INSERT INTO 방식은 동일 row 매번 추가되어 중복 폭발 발생 (Step Y 이전 버그).
        INSERT OR REPLACE로 변경하여 inquiry_id 기준 UPSERT.
        하위 호환: 기존 데이터 (inquiry_id 없음)는 created_at+content 해시로 자동 생성.
        """
        if not data or not isinstance(data, list):
            return
        try:
            import hashlib
            for item in data:
                # inquiry_id 추출 또는 생성 (하위 호환)
                inquiry_id = item.get('inquiry_id', '')
                if not inquiry_id:
                    # 기존 데이터: created_at + title + content + email 해시
                    seed = (
                        f"{item.get('created_at', '')}|"
                        f"{item.get('title', '')}|"
                        f"{item.get('content', '')}|"
                        f"{item.get('id') or item.get('email', '')}"
                    )
                    inquiry_id = hashlib.sha256(seed.encode()).hexdigest()[:16]
                
                self._exec_sqlite(
                    """INSERT OR REPLACE INTO inquiries
                    (inquiry_id, id, nickname, title, content, created_at,
                     category, status, admin_reply, admin_reply_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        inquiry_id,
                        item.get('id') or item.get('email', ''),
                        item.get('nickname', '익명'),
                        item.get('title', ''),
                        item.get('content', ''),
                        item.get('created_at', ''),
                        item.get('category', 'general'),
                        item.get('status', 'open'),
                        item.get('admin_reply', ''),
                        item.get('admin_reply_at', ''),
                    )
                )
            _logger.info(f" inquiries 적용 완료 ({len(data)}건, UPSERT)")
        except Exception as e:
            _logger.warning(f" inquiries INSERT 실패: {e}", exc_info=True)

    def _insert_gist_payments(self, data):
        """[v22 Step X] payments 테이블 Gist 데이터 적용 (UPSERT — order_id PK)"""
        if not data or not isinstance(data, list):
            return
        try:
            for item in data:
                # INSERT OR REPLACE — order_id PK 기준 UPSERT
                self._exec_sqlite(
                    """INSERT OR REPLACE INTO payments
                    (order_id, payment_key, email, plan, amount, status,
                     method, approved_at, receipt_url, created_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item.get('order_id'),
                        item.get('payment_key'),
                        item.get('email'),
                        item.get('plan'),
                        item.get('amount'),
                        item.get('status'),
                        item.get('method', ''),
                        item.get('approved_at', ''),
                        item.get('receipt_url', ''),
                        item.get('created_at', ''),
                        item.get('error_message', ''),
                    )
                )
            _logger.info(f" payments 적용 완료 ({len(data)}건)")
        except Exception as e:
            _logger.warning(f" payments INSERT 실패: {e}", exc_info=True)

    def _load_users_from_gist(self):
        self._load_gist_to_table(USER_DB_FILE, "users")

    def _load_inquiries_from_gist(self):
        self._load_gist_to_table(INQUIRY_DB_FILE, "inquiries")

    def _load_payments_from_gist(self):
        """[v22 Step X] payments 테이블 Gist에서 로드"""
        self._load_gist_to_table(PAYMENT_DB_FILE, "payments")

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
            elif tablename == 'payments':
                # [v22 Step X] payments 테이블 적용
                self._insert_gist_payments(data)

        except Exception as e:
            _logger.error(f" [Gist] {tablename} 로드 실패: {e}", exc_info=True)

    # ═══════════════════════════════════════════
    #  [v6.0] Gist 업로드 — 디바운스 배치
    # ═══════════════════════════════════════════

    def _mark_gist_dirty(self, tablename: str):
        """[v6.0] 즉시 업로드 대신 dirty 플래그만 세움 → 배치 루프가 처리"""
        _gist_sync.mark_dirty(tablename)

    def _do_gist_upload(self, tablename: str, filename: str) -> bool:
        """[v22 Step X] 실제 업로드 — 테이블별 컬럼 정확히 분기.
        
        지원 테이블: users, inquiries, payments
        성공 시 True, 실패 시 False.
        """
        if not GIST_ID or not GIST_TOKEN:
            return True  # Gist 미설정은 실패가 아님
        try:
            rows = self._exec_sqlite(f"SELECT * FROM {tablename}", fetch=True)
            
            # [v22 Step X+Y] 테이블별 컬럼 정확한 분기
            if tablename == "users":
                cols = ["id", "password", "salt", "nickname", "role", "join_date",
                        "last_login", "is_banned", "security_q_idx", "security_a_hash",
                        "session_token", "prime_expire_date", "login_fail_count", "lock_until"]
            elif tablename == "inquiries":
                # [Step Y] inquiry_id PRIMARY KEY + 답변/상태/카테고리 컬럼
                cols = ["inquiry_id", "id", "nickname", "title", "content",
                        "created_at", "category", "status", "admin_reply", "admin_reply_at"]
            elif tablename == "payments":
                # CREATE TABLE 컬럼 순서와 일치 (db 패치 1)
                cols = ["order_id", "payment_key", "email", "plan", "amount",
                        "status", "method", "approved_at", "receipt_url",
                        "created_at", "error_message"]
            else:
                _logger.warning(
                    f" [Gist] 알 수 없는 테이블 '{tablename}' — 업로드 스킵"
                )
                return False

            data = [dict(zip(cols, r)) for r in rows]
            json_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)

            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            payload = {"files": {filename: {"content": json_str}}}
            resp = requests.patch(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                _logger.debug(f" Gist 배치 업로드 완료: {tablename} → {filename}")
                return True
            else:
                _logger.warning(
                    f" Gist 배치 업로드 실패 ({resp.status_code}): "
                    f"{tablename} → {filename}"
                )
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

    # ═══════════════════════════════════════════
    #  [v22 Step W] 결제 기록 함수
    # ═══════════════════════════════════════════
    def get_user_prime_expire(self, email):
        """[Step W] 사용자의 현재 Prime 만료일 조회 (조기 갱신용).
        
        Returns:
            datetime or None
        """
        try:
            row = self._exec_sqlite_one(
                "SELECT prime_expire_date, role FROM users WHERE id = ?", (email,)
            )
            if not row:
                return None
            expire_str, role = row
            if not expire_str:
                return None
            # role이 prime/pro가 아니면 만료된 것으로 간주
            if (role or "").lower() not in ("prime", "pro"):
                return None
            try:
                # "2026-04-30" 또는 "2026-04-30 00:00:00" 처리
                date_part = expire_str.split(" ")[0]
                return datetime.strptime(date_part, "%Y-%m-%d")
            except Exception:
                return None
        except Exception as e:
            _logger.warning(f"Prime 만료일 조회 실패: {e}")
            return None

    def is_payment_processed(self, order_id: str) -> bool:
        """[Step W] orderId가 이미 처리됐는지 확인 (DB 기반 중복 방지).
        
        메모리 set 보다 안정적: 서버 재시작/멀티 인스턴스에서도 작동.
        """
        try:
            row = self._exec_sqlite_one(
                "SELECT order_id FROM payments WHERE order_id = ? AND status = ?",
                (order_id, "success")
            )
            return row is not None
        except Exception as e:
            _logger.warning(f"결제 중복 체크 실패: {e}")
            return False

    def record_payment(
        self,
        order_id: str,
        payment_key: str,
        email: str,
        plan: str,
        amount: int,
        status: str,
        method: str = "",
        approved_at: str = "",
        receipt_url: str = "",
        error_message: str = "",
    ) -> bool:
        """[Step W] 결제 기록 저장 (성공/실패/금액불일치/중복/환불 모두).
        
        Args:
            status: success / failed / amount_mismatch / duplicate / refunded
        
        Returns:
            True if recorded successfully
        """
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._exec_sqlite(
                """INSERT OR REPLACE INTO payments
                (order_id, payment_key, email, plan, amount, status,
                 method, approved_at, receipt_url, created_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (order_id, payment_key, email, plan, amount, status,
                 method, approved_at, receipt_url, now_str, error_message)
            )
            self._mark_gist_dirty("payments")
            _logger.info(
                f"💳 결제 기록 저장: {order_id} / {email} / {plan} / "
                f"{amount:,}원 / {status}"
            )
            return True
        except Exception as e:
            _logger.error(f"결제 기록 저장 실패: {e}", exc_info=True)
            return False

    def get_payment(self, order_id: str) -> dict:
        """[Step W] 주문 ID로 결제 기록 조회"""
        try:
            row = self._exec_sqlite_one(
                """SELECT order_id, payment_key, email, plan, amount, status,
                          method, approved_at, receipt_url, created_at, error_message
                   FROM payments WHERE order_id = ?""",
                (order_id,)
            )
            if not row:
                return {}
            cols = ["order_id", "payment_key", "email", "plan", "amount", "status",
                    "method", "approved_at", "receipt_url", "created_at", "error_message"]
            return dict(zip(cols, row))
        except Exception as e:
            _logger.warning(f"결제 기록 조회 실패: {e}")
            return {}

    def get_user_payments(self, email: str, limit: int = 20) -> list:
        """[Step W] 사용자의 결제 이력 조회 (최근순)"""
        try:
            rows = self._exec_sqlite(
                """SELECT order_id, plan, amount, status, method,
                          approved_at, receipt_url, created_at
                   FROM payments WHERE email = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (email, limit), fetch=True
            )
            cols = ["order_id", "plan", "amount", "status", "method",
                    "approved_at", "receipt_url", "created_at"]
            return [dict(zip(cols, r)) for r in (rows or [])]
        except Exception as e:
            _logger.warning(f"사용자 결제 이력 조회 실패: {e}")
            return []

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
    # ═══════════════════════════════════════════
    #  [v22 Step Y] 문의 함수 — UPSERT 기반 (중복 방지)
    # ═══════════════════════════════════════════
    def get_all_inquiries(self):
        """[Step Y] 모든 문의 조회 (최근순)"""
        try:
            cols = ["inquiry_id", "id", "nickname", "title", "content",
                    "created_at", "category", "status", "admin_reply", "admin_reply_at"]
            rows = self._exec_sqlite(
                "SELECT inquiry_id, id, nickname, title, content, created_at, "
                "category, status, admin_reply, admin_reply_at "
                "FROM inquiries ORDER BY created_at DESC",
                fetch=True
            )
            result = []
            for r in rows or []:
                d = dict(zip(cols, r))
                d['email'] = d['id']
                result.append(d)
            return result
        except Exception as e:
            _logger.warning(f"문의 조회 실패: {e}", exc_info=True)
            return []

    def add_inquiry(self, inquiry_id: str, email: str, nickname: str,
                    title: str, content: str, created_at: str,
                    category: str = "general") -> bool:
        """[Step Y] 신규 문의 추가 — INSERT OR IGNORE (PRIMARY KEY로 중복 방지).
        
        Returns:
            True: 정상 등록
            False: 중복 또는 실패
        """
        try:
            # INSERT OR IGNORE — inquiry_id 중복 시 자동 무시
            cursor = self._exec_sqlite(
                """INSERT OR IGNORE INTO inquiries
                (inquiry_id, id, nickname, title, content, created_at,
                 category, status, admin_reply, admin_reply_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'open', '', '')""",
                (inquiry_id, email, nickname, title, content, created_at, category)
            )
            # rowcount 확인 — 0이면 중복
            if hasattr(cursor, 'rowcount') and cursor.rowcount == 0:
                _logger.info(f"📮 문의 중복 무시: {inquiry_id}")
                return False
            self._mark_gist_dirty("inquiries")
            _logger.info(f"📮 문의 등록: {inquiry_id} / {email} / {category}")
            return True
        except Exception as e:
            _logger.warning(f"문의 등록 실패: {e}", exc_info=True)
            return False

    def update_inquiry_reply(self, inquiry_id: str, admin_reply: str,
                              admin_reply_at: str = "") -> bool:
        """[Step Y] 관리자 답변 등록"""
        try:
            from datetime import datetime as _dt
            if not admin_reply_at:
                admin_reply_at = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            self._exec_sqlite(
                """UPDATE inquiries SET 
                    admin_reply = ?, admin_reply_at = ?, status = 'replied'
                   WHERE inquiry_id = ?""",
                (admin_reply, admin_reply_at, inquiry_id)
            )
            self._mark_gist_dirty("inquiries")
            return True
        except Exception as e:
            _logger.warning(f"문의 답변 실패: {e}", exc_info=True)
            return False

    def update_inquiry_status(self, inquiry_id: str, status: str) -> bool:
        """[Step Y] 문의 상태 변경 (open / in_progress / replied / closed)"""
        try:
            self._exec_sqlite(
                "UPDATE inquiries SET status = ? WHERE inquiry_id = ?",
                (status, inquiry_id)
            )
            self._mark_gist_dirty("inquiries")
            return True
        except Exception as e:
            _logger.warning(f"문의 상태 변경 실패: {e}")
            return False

    def delete_inquiry(self, inquiry_id: str) -> bool:
        """[Step Y] 문의 삭제 — inquiry_id 기반 (created_at 충돌 X)"""
        try:
            self._exec_sqlite(
                "DELETE FROM inquiries WHERE inquiry_id = ?", (inquiry_id,)
            )
            self._mark_gist_dirty("inquiries")
            return True
        except Exception as e:
            _logger.warning(f"문의 삭제 실패: {e}")
            return False

    def get_user_inquiries(self, email: str) -> list:
        """[Step Y] 사용자 본인 문의 조회"""
        try:
            cols = ["inquiry_id", "id", "nickname", "title", "content",
                    "created_at", "category", "status", "admin_reply", "admin_reply_at"]
            rows = self._exec_sqlite(
                "SELECT inquiry_id, id, nickname, title, content, created_at, "
                "category, status, admin_reply, admin_reply_at "
                "FROM inquiries WHERE id = ? ORDER BY created_at DESC",
                (email,), fetch=True
            )
            return [dict(zip(cols, r)) for r in (rows or [])]
        except Exception as e:
            _logger.warning(f"사용자 문의 조회 실패: {e}")
            return []

    def get_inquiry_stats(self) -> dict:
        """[Step Y] 문의 통계 (관리자 대시보드용)"""
        try:
            stats = {"total": 0, "open": 0, "in_progress": 0, "replied": 0, "closed": 0}
            rows = self._exec_sqlite(
                "SELECT status, COUNT(*) FROM inquiries GROUP BY status",
                fetch=True
            ) or []
            for r in rows:
                status, cnt = r
                if status in stats:
                    stats[status] = cnt
                stats["total"] += cnt
            return stats
        except Exception as e:
            _logger.warning(f"문의 통계 실패: {e}")
            return {"total": 0, "open": 0, "in_progress": 0, "replied": 0, "closed": 0}

    # ─── [Step Y] 하위 호환: 기존 save_inquiries (DEPRECATED) ───
    # 기존 코드가 호출할 수 있으므로 유지하되, 내부적으로 add_inquiry 사용 권장.
    # 사용 X — 절대 호출하지 마세요 (전체 DELETE+INSERT는 위험)
    def save_inquiries(self, items):
        """⚠️ DEPRECATED — add_inquiry 사용 권장.
        기존 호출자 호환을 위해 유지하지만 사용 X.
        """
        _logger.warning(
            "⚠️ save_inquiries() DEPRECATED — add_inquiry() 사용 권장. "
            "전체 DELETE+INSERT는 동시성 문제 + 중복 위험 있음."
        )
        return False  # 의도적으로 실패 — 새 코드는 add_inquiry 사용

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
            # [v20.6.5] 스키마 마이그레이션: 컬럼 수 불일치 시 DROP+재생성
            _EXPECTED_COLS = ["trade_date", "code", "name", "market",
                              "close_price", "open_price", "low_price", "high_price"]
            try:
                cols = [r[1] for r in self.execute_safe("PRAGMA table_info(price_snapshots)").fetchall()]
                if cols and (len(cols) != len(_EXPECTED_COLS) or "trade_date" not in cols):
                    _logger.info(f" 🔄 price_snapshots 스키마 불일치: {len(cols)}컬럼 → {len(_EXPECTED_COLS)}컬럼, DROP+재생성")
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
                # [v20.6.5] 명시 컬럼 INSERT — 스키마 불일치 방어
                self.execute_safe("""
                    INSERT INTO price_snapshots
                        (trade_date, code, name, market, close_price, open_price, low_price, high_price)
                    SELECT trade_date, code, name, market, close_price, open_price, low_price, high_price
                    FROM _tmp_snap
                """)
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
