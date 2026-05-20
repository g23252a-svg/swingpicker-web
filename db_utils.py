# db_utils.py — SQLite WAL + 배치 Gist 동기화 (v6.0)
# ═══════════════════════════════════════════════════
# [v6.0] DuckDB → SQLite WAL 마이그레이션
#   #1 OLTP 워크로드(유저 세션/로그인) → SQLite WAL 모드
#   #2 Gist 동기화 폭탄 제거 → 디바운스 배치 (60초 쿨다운)
#   #3 DuckDB는 OLAP 전용 (daily_recommend, price_snapshots)으로 분리 유지
# ═══════════════════════════════════════════════════

import hashlib  # [v22.3] inquiry_id deterministic 생성
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
# [Step AX] 관리자 감사 로그 — Gist 백업 (분쟁 대응 / Railway 재배포 보호)
ADMIN_ACTIONS_DB_FILE = "admin_actions_db.json"
# [Step BA] 결제 기록 — Gist 백업 (회귀 복구 / 결제 분쟁 대응)
PAYMENT_DB_FILE = "payments_db.json"
# [Hotfix-AB1] 약관 동의 기록 — Gist 백업 (법적 증빙 / 분쟁 대응)
TERMS_AGREEMENT_DB_FILE = "terms_agreements_db.json"

# [Step AX+BA] 테이블 → Gist 파일명 통일 매핑 (회귀 방지)
TABLE_TO_GIST_FILE = {
    "users": USER_DB_FILE,
    "inquiries": INQUIRY_DB_FILE,
    "admin_actions": ADMIN_ACTIONS_DB_FILE,
    "payments": PAYMENT_DB_FILE,  # [Step BA] 회귀 복구
    "terms_agreements": TERMS_AGREEMENT_DB_FILE,  # [Hotfix-AB1]
}

# [Step AX+BA] 테이블별 컬럼 정의 (Gist 직렬화용)
TABLE_COLUMNS = {
    "users": [
        "id", "password", "salt", "nickname", "role", "join_date",
        "last_login", "is_banned", "security_q_idx", "security_a_hash",
        "session_token", "prime_expire_date", "login_fail_count", "lock_until",
    ],
    "inquiries": [
        # [v22.3] v2 스키마 — 프론트(tab_inquiry.py / tab_pricing.py) 계약 일치
        "inquiry_id", "id", "nickname", "title", "content", "created_at",
        "category", "status", "admin_reply", "admin_reply_at",
    ],
    "admin_actions": [
        "id", "admin_email", "action_type", "target_email",
        "details", "timestamp",
    ],
    "payments": [
        "order_id", "payment_key", "email", "plan", "amount", "status",
        "method", "approved_at", "receipt_url", "created_at", "error_message",
    ],
    "terms_agreements": [
        "id", "email", "terms_version", "terms_type", "context",
        "ip_address", "user_agent", "agreed_at",
    ],
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
                    # [Step AX] TABLE_TO_GIST_FILE 통일 매핑 (admin_actions 포함)
                    filename = TABLE_TO_GIST_FILE.get(tbl)
                    if not filename:
                        _logger.warning(
                            f" [Gist] 알 수 없는 테이블 무시: {tbl}"
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
        self._exec_sqlite("""
            CREATE TABLE IF NOT EXISTS inquiries (
                id TEXT,
                nickname TEXT,
                title TEXT,
                content TEXT,
                created_at TEXT
            )
        """)

        # [v22.3] inquiries 테이블 v2 in-place 마이그레이션 (멱등 + 빠른 경로)
        self._migrate_inquiries_to_v2()

        # [v22 Step W+BA] 결제 기록 테이블 — orderId UNIQUE로 중복 방지
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
            "CREATE INDEX IF NOT EXISTS idx_payments_email "
            "ON payments (email)"
        )
        self._exec_sqlite(
            "CREATE INDEX IF NOT EXISTS idx_payments_status "
            "ON payments (status)"
        )

        # ── [Hotfix-AB1] 약관 동의 기록 테이블 ──
        # [Hotfix-AB1.1] NOT NULL DEFAULT로 NULL UNIQUE 우회 차단
        # UNIQUE(email, terms_version, terms_type, context) — 약관 종류별 세밀 추적
        self._exec_sqlite("""
            CREATE TABLE IF NOT EXISTS terms_agreements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                terms_version TEXT NOT NULL,
                terms_type TEXT NOT NULL DEFAULT 'all',
                context TEXT NOT NULL DEFAULT 'signup',
                ip_address TEXT NOT NULL DEFAULT '',
                user_agent TEXT NOT NULL DEFAULT '',
                agreed_at TEXT NOT NULL
            )
        """)
        self._exec_sqlite(
            "CREATE INDEX IF NOT EXISTS idx_terms_email_version "
            "ON terms_agreements (email, terms_version)"
        )
        # [Hotfix-AB1.1] v1 핫픽스의 옛 UNIQUE 인덱스 제거 (마이그레이션 호환)
        try:
            self._exec_sqlite("DROP INDEX IF EXISTS uq_terms_email_version_context")
        except Exception as _e:
            _logger.debug(f"옛 인덱스 제거 스킵: {_e}")
        # [Hotfix-AB1.1] terms_type 포함 UNIQUE — 약관 종류별 분리 저장 가능
        self._exec_sqlite(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_terms_email_version_type_context "
            "ON terms_agreements (email, terms_version, terms_type, context)"
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
            # [Step AY] admin_actions 복구 로드 — 감사 로그 영구 보존
            self._load_admin_actions_from_gist()
            # [Step BA] payments 복구 로드 — 결제 기록 영구 보존
            self._load_payments_from_gist()
            # [Hotfix-AB1.1] terms_agreements 복구 로드 — 약관 동의 영구 보존 (법적 증빙)
            self._load_terms_agreements_from_gist()
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
            # [Step AY+BA] 모든 백업 파일 포함
            for fname in [
                USER_DB_FILE, INQUIRY_DB_FILE,
                ADMIN_ACTIONS_DB_FILE, PAYMENT_DB_FILE,  # [BA] payments
                TERMS_AGREEMENT_DB_FILE,  # [Hotfix-AB1]
            ]:
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
        # [Step AY] admin_actions 복구 적용
        if ADMIN_ACTIONS_DB_FILE in downloaded:
            self._insert_gist_admin_actions(downloaded[ADMIN_ACTIONS_DB_FILE])
        # [Step BA] payments 복구 적용
        if PAYMENT_DB_FILE in downloaded:
            self._insert_gist_payments(downloaded[PAYMENT_DB_FILE])
        # [Hotfix-AB1] terms_agreements 복구 적용
        if TERMS_AGREEMENT_DB_FILE in downloaded:
            self._insert_gist_terms_agreements(downloaded[TERMS_AGREEMENT_DB_FILE])

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
        """SQLite UPSERT (ON CONFLICT).

        [v22.3.8] role 다운그레이드 원복 방지:
        Gist에 role='prime'이 남아있어도, 이미 SQLite가 'free'로 강등된 상태면
        강등을 유지함. 단방향 다운그레이드 보호.
        역으로 Gist에 'admin'/'prime'이 들어오고 SQLite가 'free'면 그건 정상
        업그레이드 → 그대로 반영.

        해결 시나리오:
        1. 관리자가 강등 버튼 → SQLite role='free' 변경
        2. Gist는 아직 'prime' (배치 upload 실패 또는 race)
        3. 앱 재시작 → ensure_gist_loaded → 이 함수 호출
        4. 이전: ON CONFLICT DO UPDATE → role='prime' 원복 (버그!)
        5. v22.3.8: 'prime'이 들어와도 SQLite의 'free'를 유지
        """
        join_dt_str = u.get('created_at', u.get('join_date'))
        expire_val = u.get('prime_expire_date')
        gist_role = u.get('role', 'free')
        if not expire_val and gist_role in ['prime', 'pro'] and join_dt_str:
            try:
                clean_str = str(join_dt_str)[:19].replace("T", " ")
                join_dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
                expire_val = (join_dt + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        # [v22.3.8] role 다운그레이드 원복 방지
        # SQLite에 이미 row 있고 role이 더 낮은(=권한 약한) 상태면 그것을 우선
        email_id = u.get('login_id', u.get('id'))
        existing_role = None
        existing_expire = None
        if email_id:
            row = self._exec_sqlite_one(
                "SELECT role, prime_expire_date FROM users WHERE id = ?",
                (email_id,),
            )
            if row:
                existing_role = row[0]
                existing_expire = row[1]

        # 권한 위계 (높음 → 낮음): admin > prime/pro > free > banned
        # banned는 별도 컬럼이라 role 위계에서 제외
        role_rank = {"admin": 3, "prime": 2, "pro": 2, "free": 1}

        if existing_role is not None:
            existing_rank = role_rank.get(
                str(existing_role).strip().lower(), 1
            )
            gist_rank = role_rank.get(
                str(gist_role).strip().lower(), 1
            )

            # SQLite 권한이 더 낮음(= 이미 강등됨) AND Gist 만료일이 미래 아님
            # 이런 경우 강등이 의도된 것이므로 Gist의 prime을 무시
            if existing_rank < gist_rank:
                # Gist의 만료일이 이미 지났다면 확실히 강등 의도 → 유지
                gist_expired = False
                if expire_val:
                    try:
                        exp_clean = str(expire_val).split(" ")[0]
                        gist_exp_dt = datetime.strptime(exp_clean, "%Y-%m-%d").date()
                        if gist_exp_dt < datetime.now().date():
                            gist_expired = True
                    except Exception as e:
                        _logger.debug(f"[gist-load] expire 파싱 실패: {e}")
                        pass

                if gist_expired:
                    _logger.info(
                        f" [GIST-LOAD] {email_id}: SQLite={existing_role!r} "
                        f"유지 (Gist={gist_role!r} 만료일 {expire_val} 이미 지남)"
                    )
                    # role과 expire_date는 SQLite 그대로, 나머지만 업데이트
                    vals = (
                        email_id,
                        u.get('password_hash', u.get('password')),
                        u.get('salt'), u.get('nickname'),
                        existing_role,  # ← SQLite 값 유지
                        str(join_dt_str) if join_dt_str else None,
                        str(u.get('last_login')) if u.get('last_login') else None,
                        1 if u.get('is_banned') else 0,
                        u.get('security_q_idx', 0), u.get('security_a_hash'),
                        u.get('session_token'),
                        existing_expire,  # ← SQLite 값 유지
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
                            is_banned = excluded.is_banned,
                            security_q_idx = excluded.security_q_idx,
                            security_a_hash = excluded.security_a_hash
                    """, vals)
                    return

        # 정상 UPSERT — Gist 값 반영
        vals = (
            email_id,
            u.get('password_hash', u.get('password')),
            u.get('salt'), u.get('nickname'), gist_role,
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
        """[v22.3] Gist → SQLite 복구 (멱등).

        이전 버그: 무조건 INSERT → 재시작/배경 갱신 때마다 누적
                  (10건 → 20 → 40 → 80 → 160건 폭발)
        현재: inquiry_id UNIQUE INDEX + INSERT OR IGNORE + 빈 글 차단
        입력: list[dict] 또는 {"inquiries": [...]} 둘 다 허용 (users 패턴 일치)
        """
        # dict 형태도 허용
        if isinstance(data, dict):
            data = data.get("inquiries", [])
        if not data or not isinstance(data, list):
            return

        inserted, skipped, empty = 0, 0, 0
        for item in data:
            if not isinstance(item, dict):
                continue
            email      = item.get("id") or item.get("email") or ""
            nickname   = item.get("nickname") or "익명"
            # [v22.3 hotfix] "None"/"null"/"nan"/"-" 문자열도 빈글로 처리
            title      = self._clean_inquiry_text(item.get("title"))
            content    = self._clean_inquiry_text(item.get("content"))
            created_at = item.get("created_at") or ""

            # 빈 글 차단 (DB 진입 자체 거부)
            if not title or not content:
                empty += 1
                continue

            # inquiry_id 결정 — Gist에 있으면 그대로, 없으면 deterministic 생성
            inquiry_id = (item.get("inquiry_id") or "").strip()
            if not inquiry_id:
                seed = f"{email}|{title}|{content}|{created_at}"
                inquiry_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

            cur = None
            try:
                cur = self._exec_sqlite(
                    """
                    INSERT OR IGNORE INTO inquiries
                        (id, nickname, title, content, created_at,
                         inquiry_id, category, status, admin_reply, admin_reply_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (email, nickname, title, content, created_at, inquiry_id,
                     item.get("category")       or "general",
                     item.get("status")         or "open",
                     item.get("admin_reply")    or "",
                     item.get("admin_reply_at") or ""),
                )
                if getattr(cur, "rowcount", 0) > 0:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                _logger.warning(f"[gist→inquiries] INSERT 실패: {e}")
                skipped += 1
            finally:
                # cursor 누수 방지
                if cur is not None:
                    try:
                        cur.close()
                    except Exception as _e:
                        # cursor close 실패는 무해 (이미 거래 끝남)
                        _logger.debug(f"[gist→inquiries] cursor close: {_e}")

        _logger.info(
            f"[gist→inquiries] 신규 {inserted} / 중복 {skipped} / 빈글차단 {empty}"
        )

    def _ensure_admin_actions_table(self):
        """[Step AY] admin_actions 테이블 멱등 생성 (Gist 로드 전 보장)
        
        tab_admin.py에도 동일 로직이 있지만, Gist 복구 시점이 더 이를 수 있어
        db_utils.py에서도 보장. 멱등이라 중복 호출 안전.
        """
        try:
            self._exec_sqlite("""
                CREATE TABLE IF NOT EXISTS admin_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_email TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    target_email TEXT,
                    details TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            self._exec_sqlite(
                "CREATE INDEX IF NOT EXISTS idx_admin_actions_target "
                "ON admin_actions(target_email, timestamp DESC)"
            )
            self._exec_sqlite(
                "CREATE INDEX IF NOT EXISTS idx_admin_actions_time "
                "ON admin_actions(timestamp DESC)"
            )
        except Exception as e:
            _logger.debug(f"admin_actions 테이블 보장 실패: {e}")

    def _insert_gist_admin_actions(self, data):
        """[Step AY] admin_actions Gist → SQLite 복구
        
        중복 방지: (admin_email, action_type, target_email, timestamp) 동일하면 스킵
        - id는 AUTOINCREMENT라 Gist에서 가져온 id와 충돌 가능 → 기존 row 검사
        - INSERT OR IGNORE 대신 명시적 검사 (id 컬럼 충돌 방지)
        """
        if not data or not isinstance(data, list):
            return
        try:
            self._ensure_admin_actions_table()
            inserted = 0
            skipped = 0
            for item in data:
                admin_email = item.get('admin_email', '')
                action_type = item.get('action_type', '')
                target_email = item.get('target_email', '') or ''
                timestamp = item.get('timestamp', '')
                details = item.get('details', '') or ''
                
                if not admin_email or not action_type or not timestamp:
                    skipped += 1
                    continue
                
                # 중복 검사 (자연 키 기반)
                existing = self._exec_sqlite(
                    "SELECT id FROM admin_actions "
                    "WHERE admin_email=? AND action_type=? "
                    "AND target_email=? AND timestamp=?",
                    (admin_email, action_type, target_email, timestamp),
                    fetch=True,
                )
                if existing:
                    skipped += 1
                    continue
                
                self._exec_sqlite(
                    "INSERT INTO admin_actions "
                    "(admin_email, action_type, target_email, details, timestamp) "
                    "VALUES (?,?,?,?,?)",
                    (admin_email, action_type, target_email, details, timestamp),
                )
                inserted += 1
            _logger.info(
                f" admin_actions 적용 완료 "
                f"(신규 {inserted}건 / 중복 {skipped}건)"
            )
        except Exception as e:
            _logger.warning(f" admin_actions INSERT 실패: {e}")

    def _load_users_from_gist(self):
        self._load_gist_to_table(USER_DB_FILE, "users")

    def _load_inquiries_from_gist(self):
        self._load_gist_to_table(INQUIRY_DB_FILE, "inquiries")

    def _load_admin_actions_from_gist(self):
        """[Step AY] admin_actions Gist 복구 로드 (감사 로그 보존)"""
        self._load_gist_to_table(ADMIN_ACTIONS_DB_FILE, "admin_actions")

    def _insert_gist_payments(self, data):
        """[Step BA] payments Gist → SQLite 복구 (결제 기록 보존)
        
        order_id가 PRIMARY KEY이므로 INSERT OR REPLACE로 멱등 보장.
        """
        if not data or not isinstance(data, list):
            return
        try:
            inserted = updated = 0
            for item in data:
                order_id = item.get('order_id', '')
                if not order_id:
                    continue
                
                # 기존 row 존재 여부 (통계용)
                existing = self._exec_sqlite_one(
                    "SELECT order_id FROM payments WHERE order_id = ?",
                    (order_id,)
                )
                
                self._exec_sqlite(
                    """INSERT OR REPLACE INTO payments
                    (order_id, payment_key, email, plan, amount, status,
                     method, approved_at, receipt_url, created_at, error_message)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        order_id,
                        item.get('payment_key', ''),
                        item.get('email', ''),
                        item.get('plan', ''),
                        item.get('amount', 0),
                        item.get('status', ''),
                        item.get('method', ''),
                        item.get('approved_at', ''),
                        item.get('receipt_url', ''),
                        item.get('created_at', ''),
                        item.get('error_message', ''),
                    )
                )
                if existing:
                    updated += 1
                else:
                    inserted += 1
            _logger.info(
                f" payments 적용 완료 (신규 {inserted}건 / 업데이트 {updated}건)"
            )
        except Exception as e:
            _logger.warning(f" payments INSERT 실패: {e}")

    def _load_payments_from_gist(self):
        """[Step BA] payments Gist 복구 로드 (결제 분쟁 대응)"""
        self._load_gist_to_table(PAYMENT_DB_FILE, "payments")

    # ═══════════════════════════════════════════════════
    # [Hotfix-AB1] 약관 동의 기록 — Gist 복구
    # ═══════════════════════════════════════════════════
    def _insert_gist_terms_agreements(self, data):
        """[Hotfix-AB1+AB1.1] terms_agreements Gist → SQLite 복구.

        UNIQUE(email, terms_version, terms_type, context)로 멱등 보장.
        [AB1.1] before/after row count로 신규 vs 기존 정확히 카운트.
        """
        if not data or not isinstance(data, list):
            return
        try:
            # [AB1.1] before count — 정확한 신규 수 측정
            before_row = self._exec_sqlite_one(
                "SELECT COUNT(*) FROM terms_agreements"
            )
            before_count = before_row[0] if before_row else 0

            seen = errored = 0
            for item in data:
                # [AB1.1] record_terms_agreement과 동일 정규화
                email = (item.get('email') or '').strip().lower()
                terms_version = (item.get('terms_version') or '').strip()
                terms_type = (item.get('terms_type') or 'all').strip() or 'all'
                context = (item.get('context') or 'signup').strip() or 'signup'
                if not email or not terms_version:
                    continue
                seen += 1

                try:
                    self._exec_sqlite(
                        """INSERT OR IGNORE INTO terms_agreements
                        (email, terms_version, terms_type, context,
                         ip_address, user_agent, agreed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            email,
                            terms_version,
                            terms_type,
                            context,
                            (item.get('ip_address') or '').strip(),
                            (item.get('user_agent') or '').strip(),
                            item.get('agreed_at', ''),
                        )
                    )
                except Exception:
                    errored += 1

            # [AB1.1] after count — 진짜 신규 수
            after_row = self._exec_sqlite_one(
                "SELECT COUNT(*) FROM terms_agreements"
            )
            after_count = after_row[0] if after_row else 0
            inserted = max(0, after_count - before_count)
            existing = seen - inserted - errored
            _logger.info(
                f" terms_agreements 적용 완료 "
                f"(신규 {inserted}건 / 기존 {existing}건 / 오류 {errored}건)"
            )
        except Exception as e:
            _logger.warning(f" terms_agreements INSERT 실패: {e}")


    def _load_terms_agreements_from_gist(self):
        """[Hotfix-AB1] terms_agreements Gist 복구 로드 (법적 증빙)"""
        self._load_gist_to_table(TERMS_AGREEMENT_DB_FILE, "terms_agreements")

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
            elif tablename == 'admin_actions':
                # [Step AY] admin_actions 복구 분기
                self._insert_gist_admin_actions(data)
            elif tablename == 'payments':
                # [Step BA] payments 복구 분기
                self._insert_gist_payments(data)
            elif tablename == 'terms_agreements':
                # [Hotfix-AB1.1] terms_agreements 복구 분기
                self._insert_gist_terms_agreements(data)

        except Exception as e:
            _logger.error(f" [Gist] {tablename} 로드 실패: {e}", exc_info=True)

    # ═══════════════════════════════════════════
    #  [v6.0] Gist 업로드 — 디바운스 배치
    # ═══════════════════════════════════════════

    def _mark_gist_dirty(self, tablename: str):
        """[v6.0] 즉시 업로드 대신 dirty 플래그만 세움 → 배치 루프가 처리"""
        _gist_sync.mark_dirty(tablename)

    @staticmethod
    def _quote_sqlite_ident(name: str) -> str:
        """[v22.3 hotfix] SQLite identifier 안전 인용 (예약어 / 특수문자 방어)."""
        return '"' + str(name).replace('"', '""') + '"'

    def _do_gist_upload(self, tablename: str, filename: str) -> bool:
        """실제 업로드 (배치 루프에서 호출). 성공 시 True, 실패 시 False.

        [v22.3 hotfix] SELECT * 금지 — TABLE_COLUMNS 순서로 명시 SELECT.
            기존 버그: ALTER ADD COLUMN 후 SELECT * 컬럼 순서와
                      TABLE_COLUMNS 순서가 다르면 zip()이 키-값을 어긋나게
                      매핑해서 Gist JSON이 통째로 밀림 (제목↔내용↔이메일).
            수정: SELECT col1, col2, ... 로 순서 강제 → JSON 무결성 보장.
        """
        if not GIST_ID or not GIST_TOKEN:
            return True  # Gist 미설정은 실패가 아님
        try:
            cols = TABLE_COLUMNS.get(tablename)
            if not cols:
                _logger.warning(
                    f" Gist 업로드: {tablename} 컬럼 매핑 없음 → 스킵 (안전)"
                )
                return False  # 매핑 없으면 업로드 자체 거부 (이전엔 inquiries 컬럼으로 폴백 → 위험)

            # [핵심] 명시 컬럼 SELECT — 순서 일치 보장
            select_cols = ", ".join(
                self._quote_sqlite_ident(c) for c in cols
            )
            quoted_table = self._quote_sqlite_ident(tablename)
            rows = self._exec_sqlite(
                f"SELECT {select_cols} FROM {quoted_table}",
                fetch=True,
            ) or []

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
                _logger.warning(
                    f" Gist 배치 업로드 실패 ({resp.status_code}): {tablename}"
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

            # [v22.3 hotfix-2] UTC 명시 — update_login_timestamp 등과 일관성 유지
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
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
        """[Step BB hotfix] last_login UTC 명시 저장.
        
        이전 버그: datetime.now()는 서버 로컬 시간 (Dockerfile에서 TZ=Asia/Seoul 설정)
        → KST naive datetime → tz_localize('UTC')로 잘못 간주 → +9시간 미래 표시
        
        수정: datetime.now(timezone.utc) — auth_user.py의 join_date와 일관
        → tz_localize('UTC')가 정확히 작동
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
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
                # [v22.3 hotfix-2] UTC 명시 — get_login_failures의 tz_localize('UTC')와 일관
                lock_time = (datetime.now(timezone.utc) + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
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
        """[v22.3.8] role 변경 + Gist 즉시 동기 push.

        이전 버그: dirty 플래그만 세우고 60초 배치 기다림 → 그 사이 컨테이너
                  재시작되면 Gist 옛 데이터(prime)가 SQLite를 덮어씀 → 강등 원복
        v22.3.8: role 변경은 권한 변화라 critical → 즉시 push 시도
                실패해도 dirty는 남아있어 배치 재시도 가능
        """
        self._exec_sqlite("UPDATE users SET role = ? WHERE id = ?", (new_role, email))
        self._mark_gist_dirty("users")
        # [v22.3.8] 즉시 push — 60초 배치 기다리지 않음
        try:
            filename = TABLE_TO_GIST_FILE.get("users")
            if filename:
                ok = self._do_gist_upload("users", filename)
                if ok:
                    _logger.info(
                        f" [update_user_role] {email} → {new_role} "
                        f"+ Gist 즉시 push 성공"
                    )
                else:
                    _logger.warning(
                        f" [update_user_role] {email} → {new_role} "
                        f"DB 변경 OK / Gist push 실패 (배치 재시도 대기)"
                    )
        except Exception as e:
            _logger.warning(
                f" [update_user_role] Gist 즉시 push 예외: {e} "
                f"(DB 변경은 완료, 배치 재시도 대기)"
            )
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
    #  [v22 Step W+BA] 결제 기록 메서드 — payments 테이블
    # ═══════════════════════════════════════════
    def get_user_prime_expire(self, email):
        """[Step W+BA] 사용자의 현재 Prime 만료일 조회 (조기 갱신용).
        
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
                date_part = str(expire_str).split(" ")[0]
                return datetime.strptime(date_part, "%Y-%m-%d")
            except Exception:
                return None
        except Exception as e:
            _logger.warning(f"Prime 만료일 조회 실패: {e}")
            return None

    def is_payment_processed(self, order_id: str) -> bool:
        """[Step W+BA] orderId가 이미 처리되었는지 확인 (DB 기반 중복 방지).
        
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
        """[Step W+BA] 결제 기록 저장 (성공/실패/금액불일치/중복/환불 모두).
        
        Args:
            status: success / failed / amount_mismatch / duplicate / refunded
        
        Returns:
            True if recorded successfully
        """
        try:
            # [v22.3 hotfix-2] UTC 명시 — payments.created_at 통일
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            self._exec_sqlite(
                """INSERT OR REPLACE INTO payments
                (order_id, payment_key, email, plan, amount, status,
                 method, approved_at, receipt_url, created_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (order_id, payment_key, email, plan, amount, status,
                 method, approved_at, receipt_url, now_str, error_message)
            )
            # [Step BA] Gist 백업 트리거
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
        """[Step W+BA] 주문 ID로 결제 기록 조회"""
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
        """[Step W+BA] 사용자의 결제 이력 조회 (최근순)"""
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

    # ═══════════════════════════════════════════════════
    # [Hotfix-AB1] 약관 동의 기록 (법적 증빙)
    # ═══════════════════════════════════════════════════
    def record_terms_agreement(
        self,
        email: str,
        terms_version: str,
        terms_type: str = "all",
        context: str = "signup",
        ip_address: str = "",
        user_agent: str = "",
    ) -> bool:
        """[Hotfix-AB1] 약관 동의 기록 저장.

        UNIQUE(email, terms_version, terms_type, context)로 동일 약관/컨텍스트 중복 방지.
        같은 (email, version, type, context)로 다시 호출하면 멱등 (UPDATE 안 함, 첫 동의 시점 보존).

        Returns: True if recorded successfully (이미 있어도 True)
        """
        # [Hotfix-AB1.1] 입력 정규화 — None / 공백 / 대소문자 통일
        email = (email or "").strip().lower()
        terms_version = (terms_version or "").strip()
        terms_type = (terms_type or "all").strip() or "all"
        context = (context or "signup").strip() or "signup"
        ip_address = (ip_address or "").strip()
        user_agent = (user_agent or "").strip()

        # [Hotfix-AB1.1] 필수 입력 검증 — 법적 증빙용이라 빈 값 차단
        if not email or not terms_version:
            _logger.warning(
                f"약관 동의 기록 실패: 필수 필드 누락 "
                f"(email={'있음' if email else '없음'}, "
                f"version={'있음' if terms_version else '없음'})"
            )
            return False

        try:
            # [Hotfix-AB1.1] UTC 통일 — 분쟁 시 타임존 모호성 제거
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            self._exec_sqlite(
                """INSERT OR IGNORE INTO terms_agreements
                (email, terms_version, terms_type, context,
                 ip_address, user_agent, agreed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (email, terms_version, terms_type, context,
                 ip_address, user_agent, now_str)
            )
            self._mark_gist_dirty("terms_agreements")
            _logger.info(
                f"📜 약관 동의 기록: {email} / {terms_version} / "
                f"{terms_type} / {context}"
            )
            return True
        except Exception as e:
            _logger.error(f"약관 동의 기록 실패: {e}", exc_info=True)
            return False

    def has_agreed_to_version(self, email: str, terms_version: str) -> bool:
        """[Hotfix-AB1] 사용자가 특정 버전 약관에 동의했는지 확인."""
        # [Hotfix-AB1.1] record_terms_agreement과 동일 정규화
        email = (email or "").strip().lower()
        terms_version = (terms_version or "").strip()
        if not email or not terms_version:
            return False
        try:
            row = self._exec_sqlite_one(
                """SELECT id FROM terms_agreements
                   WHERE email = ? AND terms_version = ?
                   LIMIT 1""",
                (email, terms_version)
            )
            return row is not None
        except Exception as e:
            _logger.warning(f"약관 동의 확인 실패: {e}")
            return False

    def get_user_agreement_history(self, email: str, limit: int = 20) -> list:
        """[Hotfix-AB1] 사용자의 약관 동의 이력 (분쟁 대응 / 관리자 조회용)."""
        email = (email or "").strip().lower()
        if not email:
            return []
        try:
            rows = self._exec_sqlite(
                """SELECT terms_version, terms_type, context,
                          ip_address, user_agent, agreed_at
                   FROM terms_agreements WHERE email = ?
                   ORDER BY agreed_at DESC LIMIT ?""",
                (email, limit), fetch=True
            )
            cols = ["terms_version", "terms_type", "context",
                    "ip_address", "user_agent", "agreed_at"]
            return [dict(zip(cols, r)) for r in (rows or [])]
        except Exception as e:
            _logger.warning(f"약관 동의 이력 조회 실패: {e}")
            return []


    def get_all_payments(self, status: str = None, limit: int = 1000) -> list:
        """[Step BA] 전체 결제 기록 조회 (관리자 매출 통계용).
        
        Args:
            status: 'success' / 'failed' / 'refunded' / None (모두)
        """
        try:
            if status:
                rows = self._exec_sqlite(
                    "SELECT order_id, payment_key, email, plan, amount, status, "
                    "method, approved_at, receipt_url, created_at, error_message "
                    "FROM payments WHERE status = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status, limit), fetch=True,
                )
            else:
                rows = self._exec_sqlite(
                    "SELECT order_id, payment_key, email, plan, amount, status, "
                    "method, approved_at, receipt_url, created_at, error_message "
                    "FROM payments ORDER BY created_at DESC LIMIT ?",
                    (limit,), fetch=True,
                )
            cols = TABLE_COLUMNS["payments"]
            return [dict(zip(cols, r)) for r in (rows or [])]
        except Exception as e:
            _logger.warning(f"전체 결제 조회 실패: {e}")
            return []

    def grant_all_users_trial(self, days=14):
        try:
            # [v22.3 hotfix-2] UTC 명시 — prime_expire_date 통일
            new_expire = (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
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
    #  Inquiry Methods v2 — [v22.3]
    #  ─────────────────────────────────────────
    #  프론트(tab_inquiry.py / tab_pricing.py) 계약:
    #    - add_inquiry / get_user_inquiries / get_inquiry_stats
    #    - update_inquiry_reply / delete_inquiry
    #  멱등성 보장:
    #    - inquiry_id UNIQUE INDEX + INSERT OR IGNORE
    #  데이터 무결성:
    #    - 빈 title/content DB 진입 차단
    #    - 서버사이드 길이 검증 (프론트 우회 차단)
    #  [v22.3 hotfix]
    #    - "None"/"null"/"nan"/"-" 문자열 무효화
    #    - 60초 자연 중복 차단 (inquiry_id 우회 방어)
    # ═══════════════════════════════════════════

    _INQUIRY_COLS = (
        "id", "nickname", "title", "content", "created_at",
        "inquiry_id", "category", "status", "admin_reply", "admin_reply_at",
    )

    # 서버사이드 입력 한계 (프론트 tab_inquiry.py와 동기화)
    _MAX_TITLE_LEN   = 100
    _MAX_CONTENT_LEN = 2000
    _MIN_CONTENT_LEN = 5

    # [v22.3 hotfix] 무효 텍스트 — 진짜 None / 빈 글뿐 아니라 문자열 "None" 등도 차단
    _BAD_TEXT = frozenset({"", "none", "null", "nan", "-"})

    @classmethod
    def _clean_inquiry_text(cls, value) -> str:
        """[v22.3 hotfix] 입력 텍스트 정규화 — 무효값은 빈 문자열로.

        대응:
          - 진짜 None → ""
          - 빈 문자열 / 공백만 → ""
          - 문자열 "None", "null", "NaN", "-" → ""  (대소문자 무관)
        """
        text = str(value or "").strip()
        if text.lower() in cls._BAD_TEXT:
            return ""
        return text

    def _row_to_inquiry(self, row):
        d = dict(zip(self._INQUIRY_COLS, row))
        d["email"] = d["id"]  # 프론트는 'email' 키도 사용
        return d

    def _close_cursor(self, cur):
        if cur is not None:
            try:
                cur.close()
            except Exception as _e:
                # cursor close 실패는 무해 (이미 거래 끝남)
                _logger.debug(f"[_close_cursor] cursor close: {_e}")

    # ── 조회 ──
    def get_all_inquiries(self, *, limit=None, offset=0):
        """관리자 — 전체 문의 (최신순). limit/offset 페이징 옵션."""
        try:
            sql = (
                f"SELECT {', '.join(self._INQUIRY_COLS)} FROM inquiries "
                "ORDER BY created_at DESC"
            )
            params: tuple = ()
            if limit is not None:
                sql += " LIMIT ? OFFSET ?"
                params = (int(limit), int(offset))
            rows = self._exec_sqlite(sql, params, fetch=True) or []
            return [self._row_to_inquiry(r) for r in rows]
        except Exception as e:
            _logger.warning(f"문의 조회 실패: {e}", exc_info=True)
            return []

    def get_user_inquiries(self, email: str):
        """일반 유저 — 본인 문의만 (idx_inquiries_email 사용)."""
        if not email:
            return []
        try:
            rows = self._exec_sqlite(
                f"SELECT {', '.join(self._INQUIRY_COLS)} FROM inquiries "
                "WHERE id = ? ORDER BY created_at DESC",
                (email,),
                fetch=True,
            ) or []
            return [self._row_to_inquiry(r) for r in rows]
        except Exception as e:
            _logger.warning(f"내 문의 조회 실패: {e}")
            return []

    def get_inquiry_stats(self):
        """관리자 통계 — total / open / in_progress / replied / closed."""
        out = {"total": 0, "open": 0, "in_progress": 0, "replied": 0, "closed": 0}
        try:
            rows = self._exec_sqlite(
                "SELECT COALESCE(NULLIF(TRIM(status),''), 'open'), COUNT(*) "
                "FROM inquiries GROUP BY 1",
                fetch=True,
            ) or []
            for status, cnt in rows:
                out["total"] += cnt
                if status in out:
                    out[status] = cnt
        except Exception as e:
            _logger.warning(f"문의 통계 실패: {e}")
        return out

    # ── 등록 ──
    def add_inquiry(self, *, inquiry_id, email, nickname, title, content,
                    created_at, category: str = "general") -> bool:
        """[v22.3] 신규 문의 등록 (3중 중복 방어).

        중복 방어:
          1. 무효 텍스트 차단 ("None"/"null"/"nan"/"-" 포함)
          2. 60초 자연 중복 차단 (같은 email+title+content 내 직전 글)
             → 프론트가 매번 새 inquiry_id 만들어도 막힘
          3. inquiry_id UNIQUE INDEX + INSERT OR IGNORE

        반환:
            True  — 정상 등록
            False — 중복 또는 입력 검증 실패
        """
        # [v22.3 hotfix] "None" 문자열도 무효화
        title   = self._clean_inquiry_text(title)
        content = self._clean_inquiry_text(content)

        # 서버사이드 검증
        if not inquiry_id or not title or not content:
            return False
        if len(title) > self._MAX_TITLE_LEN:
            _logger.info(f"add_inquiry 거부: title 길이 {len(title)}")
            return False
        if len(content) < self._MIN_CONTENT_LEN or len(content) > self._MAX_CONTENT_LEN:
            _logger.info(f"add_inquiry 거부: content 길이 {len(content)}")
            return False

        # [v22.3 hotfix] 60초 자연 중복 — inquiry_id 우회(매번 새 ID) 시에도 차단
        try:
            dup = self._exec_sqlite_one("""
                SELECT inquiry_id FROM inquiries
                 WHERE id = ?
                   AND title = ?
                   AND content = ?
                   AND datetime(created_at) >= datetime(?, '-60 seconds')
                 LIMIT 1
            """, (email or "", title, content, created_at))
            if dup:
                _logger.info(
                    f"add_inquiry 거부: 60초 내 동일 글 (기존 inquiry_id={dup[0]})"
                )
                return False
        except Exception as e:
            # 중복 검사 실패는 등록 자체를 막지 않음 (UNIQUE INDEX가 보호)
            _logger.warning(f"60초 중복 검사 실패 (계속 진행): {e}")

        cur = None
        try:
            cur = self._exec_sqlite(
                """
                INSERT OR IGNORE INTO inquiries
                    (id, nickname, title, content, created_at,
                     inquiry_id, category, status, admin_reply, admin_reply_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (email or "", nickname or "익명", title, content, created_at,
                 inquiry_id, category or "general", "open", "", ""),
            )
            if getattr(cur, "rowcount", 0) == 0:
                return False  # UNIQUE 충돌 = 더블 클릭
            self._mark_gist_dirty("inquiries")
            return True
        except Exception as e:
            _logger.warning(f"add_inquiry 실패: {e}", exc_info=True)
            return False
        finally:
            self._close_cursor(cur)

    # ── 답변 / 삭제 ──
    def update_inquiry_reply(self, inquiry_id: str, reply: str) -> bool:
        """관리자 답변 등록 → status를 'replied'로 자동 전환."""
        reply = (reply or "").strip()
        if not inquiry_id or not reply:
            return False
        cur = None
        try:
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            cur = self._exec_sqlite(
                "UPDATE inquiries SET admin_reply = ?, admin_reply_at = ?, "
                "status = 'replied' WHERE inquiry_id = ?",
                (reply, now_utc, inquiry_id),
            )
            if getattr(cur, "rowcount", 0) == 0:
                return False
            self._mark_gist_dirty("inquiries")
            return True
        except Exception as e:
            _logger.warning(f"update_inquiry_reply 실패: {e}")
            return False
        finally:
            self._close_cursor(cur)

    def delete_inquiry(self, inquiry_id: str) -> bool:
        """문의 삭제 (관리자 / 본인)."""
        if not inquiry_id:
            return False
        cur = None
        try:
            cur = self._exec_sqlite(
                "DELETE FROM inquiries WHERE inquiry_id = ?", (inquiry_id,)
            )
            if getattr(cur, "rowcount", 0) == 0:
                return False
            self._mark_gist_dirty("inquiries")
            return True
        except Exception as e:
            _logger.warning(f"delete_inquiry 실패: {e}")
            return False
        finally:
            self._close_cursor(cur)

    # ── 호환 (DEPRECATED) ──
    def save_inquiries(self, items, *, force: bool = False):
        """[DEPRECATED v22.3] 통째 교체 — 데이터 손실 위험.

        force=True 명시 호출만 허용. 신규 코드는 add_inquiry 사용.
        """
        _logger.warning(
            "save_inquiries() DEPRECATED — add_inquiry/update_inquiry_reply 권장"
        )
        if not force:
            existing = self._exec_sqlite_one("SELECT COUNT(*) FROM inquiries")
            if existing and existing[0] > 0:
                _logger.error(
                    f"save_inquiries(force=False) 거부 — 기존 {existing[0]}건 보호"
                )
                return False

        self._exec_sqlite("DELETE FROM inquiries")
        for i in (items or []):
            email   = i.get("email") or i.get("id") or ""
            # [v22.3 hotfix-2] deprecated 경로에도 무효 텍스트 차단
            title   = self._clean_inquiry_text(i.get("title"))
            content = self._clean_inquiry_text(i.get("content"))
            if not title or not content:
                continue
            created_at = i.get("created_at") or ""
            iid = i.get("inquiry_id") or hashlib.sha256(
                f"{email}|{title}|{content}|{created_at}".encode("utf-8")
            ).hexdigest()[:16]
            self._exec_sqlite(
                """INSERT OR IGNORE INTO inquiries
                   (id, nickname, title, content, created_at,
                    inquiry_id, category, status, admin_reply, admin_reply_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (email, i.get("nickname") or "익명", title, content, created_at,
                 iid, i.get("category") or "general",
                 i.get("status") or "open",
                 i.get("admin_reply") or "",
                 i.get("admin_reply_at") or ""),
            )
        self._mark_gist_dirty("inquiries")
        return True

    # ── 진단 (관리자 전용) ──
    def verify_inquiries_health(self) -> dict:
        """[v22.3] 관리자 진단 — 무효/중복/스키마 상태 한눈에.

        반환 dict 예시:
          {
            "schema_version": 2,
            "total": 12,
            "empty_title_or_content": 0,
            "missing_inquiry_id": 0,
            "duplicate_inquiry_id_groups": 0,
            "indexes_ok": True,
          }
        """
        out = {
            "schema_version": self._get_schema_version("inquiries"),
            "total": 0,
            "empty_title_or_content": 0,
            "missing_inquiry_id": 0,
            "duplicate_inquiry_id_groups": 0,
            "indexes_ok": False,
        }
        try:
            r = self._exec_sqlite_one("SELECT COUNT(*) FROM inquiries")
            out["total"] = int(r[0]) if r else 0

            r = self._exec_sqlite_one("""
                SELECT COUNT(*) FROM inquiries
                 WHERE LOWER(TRIM(COALESCE(title, '')))   IN ('', 'none', 'null', 'nan', '-')
                    OR LOWER(TRIM(COALESCE(content, ''))) IN ('', 'none', 'null', 'nan', '-')
            """)
            out["empty_title_or_content"] = int(r[0]) if r else 0

            r = self._exec_sqlite_one("""
                SELECT COUNT(*) FROM inquiries
                 WHERE COALESCE(inquiry_id, '') = ''
            """)
            out["missing_inquiry_id"] = int(r[0]) if r else 0

            rows = self._exec_sqlite("""
                SELECT inquiry_id, COUNT(*) FROM inquiries
                 WHERE COALESCE(inquiry_id, '') <> ''
                 GROUP BY inquiry_id
                HAVING COUNT(*) > 1
            """, fetch=True) or []
            out["duplicate_inquiry_id_groups"] = len(rows)

            idx_rows = self._exec_sqlite(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='inquiries'",
                fetch=True,
            ) or []
            idx_names = {r[0] for r in idx_rows}
            out["indexes_ok"] = (
                "idx_inquiries_inquiry_id" in idx_names
                and "idx_inquiries_email" in idx_names
                and "idx_inquiries_created_at" in idx_names
            )
        except Exception as e:
            _logger.warning(f"verify_inquiries_health 실패: {e}")
        return out

    # ═══════════════════════════════════════════
    #  Schema Versioning — 컴포넌트별 마이그레이션 관리
    # ═══════════════════════════════════════════

    _INQUIRY_TARGET_VERSION = 2

    def _ensure_schema_versions_table(self):
        self._exec_sqlite("""
            CREATE TABLE IF NOT EXISTS schema_versions (
                component  TEXT PRIMARY KEY,
                version    INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            )
        """)

    def _get_schema_version(self, component: str) -> int:
        self._ensure_schema_versions_table()
        row = self._exec_sqlite_one(
            "SELECT version FROM schema_versions WHERE component = ?",
            (component,),
        )
        return int(row[0]) if row else 0

    def _set_schema_version(self, component: str, version: int):
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self._exec_sqlite("""
            INSERT INTO schema_versions (component, version, applied_at)
            VALUES (?, ?, ?)
            ON CONFLICT(component) DO UPDATE SET
                version    = excluded.version,
                applied_at = excluded.applied_at
        """, (component, version, now_utc))

    def _ensure_inquiry_indexes(self):
        """[v22.3] 멱등 인덱스 — 매 재시작 호출 안전."""
        # UNIQUE: idempotency 보장 (부분 인덱스로 NULL 다중 허용)
        self._exec_sqlite(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_inquiries_inquiry_id "
            "ON inquiries(inquiry_id) WHERE COALESCE(inquiry_id, '') <> ''"
        )
        # 쿼리 성능: get_user_inquiries(email)
        self._exec_sqlite(
            "CREATE INDEX IF NOT EXISTS idx_inquiries_email "
            "ON inquiries(id)"
        )
        # 쿼리 성능: ORDER BY created_at DESC
        self._exec_sqlite(
            "CREATE INDEX IF NOT EXISTS idx_inquiries_created_at "
            "ON inquiries(created_at DESC)"
        )

    def _cleanup_invalid_inquiries(self) -> int:
        """[v22.3 hotfix] 무효/중복 문의 정리 — fast path에서도 호출.

        v2 마이그레이션 후 Gist에 폴루션이 새로 들어오는 경우를 위해
        매 재시작마다 실행 (LOWER(TRIM(...))는 인덱스 안 타지만
        평소엔 0~수 건만 매치되므로 비용 무시 가능).

        [v22.3 hotfix-2] 정리 후 Gist dirty 마킹 — 로컬/Gist 동기화 보장.

        반환: 삭제된 행 수 (0이면 dirty 마킹 스킵 → 무의미한 Gist 업로드 방지)
        """
        deleted_total = 0
        try:
            before = self._exec_sqlite_one("SELECT COUNT(*) FROM inquiries")
            before_count = before[0] if before else 0

            # 1. 무효 텍스트 정리 — "None" / "null" / "nan" / "-" 포함
            self._exec_sqlite("""
                DELETE FROM inquiries
                 WHERE LOWER(TRIM(COALESCE(title, '')))   IN ('', 'none', 'null', 'nan', '-')
                    OR LOWER(TRIM(COALESCE(content, ''))) IN ('', 'none', 'null', 'nan', '-')
            """)
            # 2. inquiry_id 중복 잔재 — fast path 진입 후 새로 생긴 중복 정리
            self._exec_sqlite("""
                DELETE FROM inquiries
                 WHERE COALESCE(inquiry_id, '') <> ''
                   AND rowid NOT IN (
                       SELECT MIN(rowid) FROM inquiries
                        WHERE COALESCE(inquiry_id, '') <> ''
                        GROUP BY inquiry_id
                   )
            """)

            after = self._exec_sqlite_one("SELECT COUNT(*) FROM inquiries")
            after_count = after[0] if after else 0
            deleted_total = max(0, before_count - after_count)

            # 정리된 결과를 Gist에 반영 — 안 그러면 다음 sync에서 폴루션이 다시 내려옴
            if deleted_total > 0:
                self._mark_gist_dirty("inquiries")
                _logger.info(
                    f"[cleanup-inquiries] {before_count} → {after_count} "
                    f"(-{deleted_total}) → Gist dirty 마킹"
                )
        except Exception as e:
            _logger.warning(f"[cleanup-inquiries] 실패: {e}")

        return deleted_total

    def _migrate_inquiries_to_v2(self):
        """[v22.3] inquiries 5컬럼 → 10컬럼 in-place 마이그레이션.

        ─────────────────────────────────────────
        멱등성 3중 안전망:
          1. schema_versions 가드 — 빠른 경로 (1회만 실행)
          2. ALTER/INDEX는 IF NOT EXISTS / 컬럼 존재 체크
          3. DELETE/UPDATE는 조건절로 0행 시 no-op
        [v22.3 hotfix]
          fast path에서도 _cleanup_invalid_inquiries 실행 →
          v2 도달 후 Gist에서 들어온 "None" 폴루션 즉시 청소
        ─────────────────────────────────────────
        """
        # ── [빠른 경로] 이미 v2: cleanup + 인덱스만 보강하고 종료 ──
        if self._get_schema_version("inquiries") >= self._INQUIRY_TARGET_VERSION:
            self._cleanup_invalid_inquiries()  # [hotfix] 매 재시작 정리
            self._ensure_inquiry_indexes()
            return

        _logger.info("[migrate-v2] inquiries v1 → v2 시작")

        # 1. 현재 컬럼 확인
        try:
            rows = self._exec_sqlite("PRAGMA table_info(inquiries)", fetch=True)
        except Exception as e:
            _logger.error(f"[migrate-v2] PRAGMA 실패 → 중단: {e}")
            return
        existing_cols = {r[1] for r in rows}

        # 2. 빠진 v2 컬럼 추가
        v2_cols = [
            ("inquiry_id",     "TEXT"),
            ("category",       "TEXT NOT NULL DEFAULT 'general'"),
            ("status",         "TEXT NOT NULL DEFAULT 'open'"),
            ("admin_reply",    "TEXT NOT NULL DEFAULT ''"),
            ("admin_reply_at", "TEXT NOT NULL DEFAULT ''"),
        ]
        for col, ddl in v2_cols:
            if col not in existing_cols:
                try:
                    self._exec_sqlite(
                        f"ALTER TABLE inquiries ADD COLUMN {col} {ddl}"
                    )
                    _logger.info(f"[migrate-v2] +{col}")
                except Exception as e:
                    _logger.warning(f"[migrate-v2] {col} ADD 실패: {e}")

        # 3. 빈 / 무효 텍스트 정리 (📌 None 문의 + "None" 문자열 동시 처리)
        try:
            before = self._exec_sqlite_one("SELECT COUNT(*) FROM inquiries")
            self._cleanup_invalid_inquiries()
            after = self._exec_sqlite_one("SELECT COUNT(*) FROM inquiries")
            if before and after and before[0] != after[0]:
                _logger.info(
                    f"[migrate-v2] 무효 문의 정리 {before[0]} → {after[0]} "
                    f"(-{before[0] - after[0]})"
                )
        except Exception as e:
            _logger.warning(f"[migrate-v2] 무효 글 정리 실패: {e}")

        # 4. 레거시 행에 deterministic inquiry_id 백필
        try:
            legacy = self._exec_sqlite("""
                SELECT rowid, COALESCE(id,''), COALESCE(title,''),
                       COALESCE(content,''), COALESCE(created_at,'')
                  FROM inquiries
                 WHERE COALESCE(inquiry_id, '') = ''
            """, fetch=True) or []
            for rowid, email, title, content, created_at in legacy:
                seed = f"{email}|{title}|{content}|{created_at}"
                iid = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
                self._exec_sqlite(
                    "UPDATE inquiries SET inquiry_id = ? "
                    "WHERE rowid = ? AND COALESCE(inquiry_id,'') = ''",
                    (iid, rowid),
                )
            if legacy:
                _logger.info(f"[migrate-v2] inquiry_id 백필 {len(legacy)}건")
        except Exception as e:
            _logger.warning(f"[migrate-v2] 백필 실패: {e}")

        # 5. inquiry_id 중복 제거 (가장 빠른 rowid만 유지)
        try:
            self._exec_sqlite("""
                DELETE FROM inquiries
                 WHERE COALESCE(inquiry_id, '') <> ''
                   AND rowid NOT IN (
                       SELECT MIN(rowid) FROM inquiries
                        WHERE COALESCE(inquiry_id, '') <> ''
                        GROUP BY inquiry_id
                   )
            """)
        except Exception as e:
            _logger.warning(f"[migrate-v2] 중복 제거 실패: {e}")

        # 6. 인덱스 (UNIQUE 포함) 생성
        self._ensure_inquiry_indexes()

        # 7. 스키마 버전 마킹 — 다음 재시작부터 빠른 경로
        self._set_schema_version("inquiries", self._INQUIRY_TARGET_VERSION)

        # 8. 정리 결과를 Gist에도 전파
        self._mark_gist_dirty("inquiries")

        _logger.info("[migrate-v2] 완료 → schema_versions['inquiries'] = 2")

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
