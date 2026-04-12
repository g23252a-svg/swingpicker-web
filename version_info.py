# -*- coding: utf-8 -*-
"""
version_info.py (v20.6.3 SSOT Complete)
- CHANGELOG: 전체 업데이트 이력 관리
- UI: 프리미엄 사이드바 뱃지, 업데이트 타임라인
- 100/100: dashboard.py에서 요청하는 모든 물자(변수/함수) 완비
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger("version_info")

# ----------------- 1. 진실의 원천 (CHANGELOG) -----------------
CHANGELOG: List[Dict[str, Any]] = [
    {
        "version": "21.3",
        "date": "2026-04-12",
        "type": "major",
        "title": "ELITE Ranking + Combo Optimizer + Auth Hardening + Live Price Engine",
        "items": [
            "🏆 **ELITE_SCORE SSOT:** 3축 밸런스(40%) + 현재가 RR(25%) + 밸런스(20%) + 진입갭(10%) + ROUTE(5%) 공식 랭킹 지표 승격",
            "🏆 **TOP_PICK 하드게이트:** ELITE≥60 + ARMED/ATTACK + RR≥0.5 + BAL≥50 + 거래대금≥50억 + 진입갭≤5% + 과열 제외 → 476→8종목 필터",
            "🎯 **조합 최적화 엔진:** combo_optimizer.py — 과거 데이터 기반 최고 승률 지표 조합 자동 탐색 (S≥90 T≥80 AI≥60 → 승률 78.8%)",
            "💎 **EV 이중 랭킹:** 승률 최적 + 기대값(EV) 최적 동시 탐색 — EV = 승률×평균수익 − (1−승률)×평균손실",
            "📊 **시장 대시보드 개선:** 엔진 상태 패널(매크로 리스크/Breadth/신뢰도) + ELITE Top 3 + 최적 조합 매칭 종목 리스트",
            "📊 **종목 테이블 재설계:** AI점수 1칸 → ELITE/종합/S/T/AI/균형/RR 7컬럼, 🏆 ELITE순 기본 정렬, 숫자 정렬 수정",
            "📊 **칸반 카드 강화:** TOP_PICK 🏆배지 + ELITE_REASON 직접 노출 + S/T/AI 미니 스코어",
            "💰 **Kelly 자동 연동:** 종목 선택 시 EST_WIN_RATE + RR_NOW_TP1 자동 매핑, 금액만 조절하면 자동 재계산",
            "💼 **포트폴리오 입력 UX:** 종목 검색 드롭다운(KRX 2,882종목) + 평단가/수량 입력 → 보유 카드 표시 + 개별 삭제",
            "📡 **LIVE_PRICE 엔진:** 시간외 가격 → RR + ELITE + BALANCE + TOP_PICK 전체 실시간 재계산",
            "🔐 **Auth SSOT 완성:** compute_access_status() 단일 함수 + require_premium() 서버측 강제 검증 + @premium_guard 데코레이터",
            "⏰ **만료 자동 강등:** downgrade_expired_users() — 만료 PRIME→FREE 자동 전환 + 관리자 강등 버튼",
            "👑 **관리자 화면 개선:** 실제 접근 상태(✅활성/❌만료/⚪무료/🔑관리자) 컬럼 + 상단 요약 배지",
            "🔧 **Feature Schema 통일:** _compute_feature_hash() → FEATURE_CONTRACT.schema_hash SSOT, ML 캐시 정상 로드",
            "📈 **성과 탭 복구:** .dockerignore data/ 제외 → 무거운 파일만 제외, 22일치 성과 차트 + 다크 테마 호버 수정",
            "🏆 **TOP_PICK 검증 리포트:** top_pick_validation_YYYYMMDD.json 자동 생성 (종목수/평균ELITE/RR/승률)",
            "📊 **CARRY 재분석:** 9%→96% refresh rate, DI패턴 top_df 보강, 진단 로그 강화",
            "🛡️ **NaN 크래시 방어:** _si() Safe Int + _nz() NaN-aware 폴백 — LIVE_PRICE NaN 시 종가 자동 전환",
            "🔓 **프리미엄 전체 표시:** prime/admin 50종목 제한 해제 → 전체 476종목 열람 가능",
            "🚀 **Railway 자동 배포:** GitHub 연결 복구, push 시 자동 deploy 활성화",
            "📱 **Google Play 등록:** 개발자 계정 생성 완료, 본인 확인 심사 중",
        ],
        "schema_min": 5
    },
    {
        "version": "20.6.5",
        "date": "2026-03-14",
        "type": "minor",
        "title": "Module Extraction + Pytest Migration — collector 구조 분할 시작",
        "items": [
            "🏗️ **trigger_engine.py 신규:** calculate_trigger_score + calc_volume_profile_v2 (198줄) collector.py에서 추출, pipeline_score.py 직접 import 전환",
            "🏗️ **investor_flow.py 신규:** fetch_investor_net_buying (55줄) collector.py에서 추출, pipeline_score/pipeline_data 직접 import 전환",
            "📉 **collector.py 경량화:** 2393줄 → 2148줄 (-245줄), 재수출로 하위 호환 유지",
            "🔌 **순환 의존 해소:** pipeline_score.py/pipeline_data.py가 trigger_engine/investor_flow 직접 import → collector 경유 불필요",
            "✅ **pytest 마이그레이션:** test_toxic_filter.py + test_stop_logic.py → pytest class 기반 (pytest -v / 스크립트 양방향 실행)",
        ],
        "schema_min": 5
    },
    {
        "version": "20.6.4",
        "date": "2026-03-14",
        "type": "patch",
        "title": "SSOT Hardening + Bare Except Purge + Stop Logic Tests",
        "items": [
            "🎯 **SSOT 강화:** scoring_engine.py 레거시+벡터 ROUTE 임계치 15건 → collector_config.IndicatorConfig로 통일 (행 단위/벡터 경로 모두 포함)",
            "🎯 **레거시 EBS SSOT:** calculate_ebs_independent() RSI/Vol_Quality 하드코딩 → config 참조",
            "🛡️ **bare except 정리:** 핵심 실행 경로 (collector/pipeline_calibrate/prefetch_flow/version_info/time_utils/strategy_lab) 16건 → 명시적 예외 타입",
            "🧠 **ML 가중치 오염 완화:** _calc_ml_weight partial failure 시 0점 종목을 coverage 계산에서 분리",
            "✅ **stop_logic 전용 테스트:** test_stop_logic.py 신규 — 6중 안전장치 + check_entry_filter API 정합성 검증 (25+ 케이스)",
            "🐛 **회귀 수정:** determine_state_dynamic() _cfg_ind 초기화 순서 버그 → EXIT_WARNING 검출 복구",
            "🧠 **ML center 완전 수정:** _calc_ml_weight() ml_center도 ml_active 기준 산출 (partial failure 보정 완성)",
        ],
        "schema_min": 5
    },
    {
        "version": "20.6.3",
        "date": "2026-03-10",
        "type": "minor",
        "title": "SSOT Complete + Deterministic Engine — Direct Macro SSOT + Adaptive Reason",
        "items": [
            "🎯 **SSOT 완성:** dashboard.py 독자 build_global_score() 제거 → scoring_engine.build_global_score() 단일 경로, CSV/Dashboard/Backtest 점수 100% 일치",
            "⚡ **Trigger 병렬 배치:** ThreadPoolExecutor 4-worker 병렬 처리 (iterrows 제거), sequential 대비 ~3x 속도 향상",
            "⚡ **ROUTE 벡터 판정:** apply(determine_state_dynamic) 제거 → _vec_determine_state_dynamic() 벡터 연산, 100% 단건 함수 일치 검증 완료",
            "📝 **점수 설명 컬럼:** SCORE_REASON_TOP1/TOP2 실제 점수순 정렬 (numpy argsort 완전 벡터화), 장세 연동 임계치 (BEAR=50, CAUTION=60, NORMAL=70)",
            "🔧 **예외 처리 등급화:** 투자 판단 축(ML/TRIGGER/SECTOR/FLOW) 실패 시 [AXIS:*] 태그 warning → run_health 후속 평가",
            "💾 **ML 캐시 개선:** feature_cache pickle → joblib + schema sidecar (feature_cols_hash), 구버전 자동 마이그레이션",
            "📡 **Macro SSOT 직접 전달:** MACRO_RISK/MARKET_BREADTH를 CSV + run_health JSON + run_meta JSON sidecar에 직접 저장",
            "📋 **run_meta sidecar:** 레거시 CSV용 메타 복원 경로 — macro_risk/가중치/scoring_axes 등 파이프라인 컨텍스트 보존",
            "📝 **주석/문서 동기화:** scoring_engine/pipeline_score/pipeline_finalize/ml_engine/run_health/version_info docstring을 v20.6.3 실제 구현과 일치시킴",
            "✅ **테스트 동봉:** test_v206_features.py — ROUTE 벡터 정확도/설명 컬럼 점수순/장세 연동/Macro SSOT 회귀 검증",
        ],
        "schema_min": 5
    },
    {
        "version": "20.5c",
        "date": "2026-03-08",
        "type": "minor",
        "title": "KIS API 수급 연동 완전체 — Run Status: OK 100/100 달성",
        "items": [
            "📡 **KIS API 수급 선수집 (prefetch_flow.py v2.0):** pykrx 전용 구조 → KIS API v2.0 완전 교체, `/uapi/domestic-stock/v1/quotations/foreign-institution-total` (tr_id: FHPTJ04400000) 단일 호출로 외인·기관 동시 수집",
            "🗂️ **flow 캐시 구조 확정:** `frg` / `inst` / `ant` 키 기반 저장, 주말 stale 오판 수정 (직전 평일 기준 비교)",
            "🔌 **collector 캐시 우선 로드:** `fetch_investor_net_buying()` 상단에 KIS 캐시 우선 로드 블록 삽입, pykrx 폴백은 캐시 미존재 시에만 실행",
            "✅ **end-to-end 검증 완료:** 수급 캐시 로드: 외인 30건 기관 30건 → 🟢 Run Status: OK (신뢰도: 100/100, 최대허용: ATTACK)",
            "🔧 **GitHub Secrets 연동:** KIS_APP_KEY / KIS_APP_SECRET 등록, prefetch_flow.yml 환경변수 연결 완료",
            "🐛 **load_cache 주말 버그 수정:** 일요일 실행 시 trade_ymd(금) 불일치 → stale=True 오판 해소",
        ],
        "schema_min": 5
    },
    {
        "version": "19.1.0",
        "date": "2026-03-01",
        "type": "minor",
        "title": "CI/CD 파이프라인 + 보안·안정성 긴급 패치 (3-Fix Hotfix)",
        "items": [
            "🔧 **GitHub Actions CI/CD:** Push/PR마다 순환참조 검사 + Silent Exception 검출 + 배포 전 안전 체크 자동화 (ci.yml, pre-deploy.yml)",
            "🔒 **관리자 암호화 강화 (Fix#3):** SHA256 단순 해싱 → bcrypt 12-round 전환, 레인보우 테이블 공격 완전 차단, 타이밍 공격 방어 내장 (bcrypt.checkpw)",
            "🧵 **Thread-Safety 수정 (Fix#1):** DataStore.scored 프로퍼티가 참조(Reference)만 반환하던 가짜 스레드 안전성 → .copy() 스냅샷 반환으로 Race Condition 근본 차단",
            "📢 **Silent Exception 로거 (Fix#4):** _authenticate_user 로그인 타임스탬프 + get_auth_status 구독 만료일 파싱 실패 시 logger.error/warning 기록 (기존 except:pass 제거)",
            "📊 **Silent Exception 자동 추적:** 전체 120건 except:pass 잔존 현황을 매 Push마다 CI에서 리포트, 점진적 제로화 추적 시작",
            "🛡️ **의존성 방향 검사:** AST 기반 import 그래프 분석으로 views→services→core 단방향 강제, NiceGUI 격리 규칙 자동 검증",
            "⚙️ **환경변수 분리:** BCRYPT_COST(운영12/테스트4), LOG_LEVEL, LOG_TO_FILE Railway 환경변수 추가",
        ],
        "schema_min": 5
    },
    {
        "version": "19.0.0",
        "date": "2026-02-22",
        "type": "major",
        "title": "Production-Grade Refactor: P2~P5 완전체 (276/276 Tests)",
        "items": [
            "🏗️ **P2 모듈화:** collector.py 3306줄→2730줄, 7개 모듈 분리 (collector_config, data_source, macro_filter, news_engine, telegram_sender, validation, scoring_engine)",
            "🗄️ **Parquet 캐시:** pickle RCE 취약점 제거 → Parquet 기반 안전 캐시 (allow_legacy_pickle=False 기본)",
            "⚙️ **Config 중앙화:** 40+개 상수 dataclass화 (collector_config.py), 매직넘버 자동검출 grep 패턴",
            "🔒 **섹터 이중 보상 잠금:** STRUCT 불변, TIMING만 +8/+4, FINAL=TIMING×w_t 수학적 검증",
            "📊 **P3 #13 자동 백테스트:** 폐루프 완성 (추천→실현수익률→승률테이블→Kelly), 7개 안전장치 (누수차단/진입청산고정/binning/min_n+스무딩/Kelly제한/비용반영/기업행위필터)",
            "📍 **P3 #14 포지션 트래킹:** 실시간 SL/TP/드로다운 감지, event_key 중복방지, positions.json SSOT, 원자적 저장(tmp→rename+락)",
            "🧠 **P3 #15 Multi-Timeframe:** 주봉/월봉 구조 필터 (STRUCT±10/15), 미완성봉 누수방지, Config화, TIMING 무관 수학적 검증",
            "🔄 **#16 Gemini 429 방어:** exponential backoff + jitter + Retry-After 우선 + cap + total_timeout(60s), news_engine SSOT 공유",
            "🗃️ **#17 DB 싱글톤:** thread-safe Double-Checked Locking, 쿼리 직렬화 _conn_lock, TTL(10분) gist 재로드, force_refresh",
            "🛡️ **P4 #18 평문 제거:** MASTER_ADMIN_PW 전역변수 del, _ADMIN_PW_SET(bool)만 유지, compare_digest timing-safe",
            "📝 **P4 #19 tomllib 전환:** SECTION_SUBKEY 네임스페이스 등록 (AUTH_MASTER_ADMIN_PW), 섹션 키 충돌 방지",
            "🧪 **P5 #20 단위테스트 확대:** TIMING/STRUCT/SuperTrend/EBS/safe_float 경계값+NULL방어, build_global_score 불변식",
            "📈 **P5 #21 일일 성과 리포트:** #13 realized_returns 재사용, report_key 중복방지, 점수대별 승률/Top 수익·손실",
            "✅ **276/276 ALL PASSED:** 9개 테스트 파일 (scoring_weights 54, ssot_import 14, toxic_filter 12, p2_refactor 53, auto_backtest 31, llm_retry 30, db_singleton 15, security 23, p5_final 44)",
        ],
        "schema_min": 5
    },
    {
        "version": "18.0.0",
        "date": "2026-02-17",
        "type": "major",
        "title": "Premium UI + ML v17.1 + Critical Bug Fixes",
        "items": [
            "🎨 **Premium Dark Theme:** Outfit 폰트, 글래스모피즘 카드, 그라디언트 메트릭, 탭 하이라이트 등 전면 UI 리뉴얼",
            "💎 **Hero Banner:** 기존 st.title → 그라디언트 히어로 배너 + AI 엔진 스펙 뱃지",
            "🃏 **Kanban 2.0:** 칸반 카드 글래스모피즘 + 점수별 색상 뱃지 + R:R 시각 바 + 호버 부유 효과",
            "🧠 **ML v17.1:** 피처 버전 메타데이터(trading_meta_v17.json) 저장 → 코드/모델 불일치 시 레거시 자동 폴백",
            "🔒 **Thread Safety:** load_model()에 threading.Lock 적용 → ThreadPoolExecutor 레이스 방지",
            "🐛 **매수가 중복 계산 Fix:** R:R 2.0→2.5 복원, 7%↑ 급등주 추격방지 정상화",
            "🚫 **OVERHEAT 필터:** RSI≥75 과열 종목이 정예군 Top120에서 자동 제외",
            "🤖 **ML 피처 확장:** 6→16개 (MFI, MACD_Hist, BB_Width, ATR, OBV_Slope 등)",
            "📊 **XGBoost 앙상블:** LSTM 60% + XGBoost 40% 가중평균, Focal Loss 적용",
            "🔑 **Secrets 매핑 테이블:** TOML 중첩 섹션 → 환경변수 확정 매핑 (telegram_token → TG_TOKEN)",
            "🗄️ **DB 컬럼 Fix:** users 테이블 14컬럼 vs INSERT 12값 불일치 → 컬럼명 명시로 해결",
            "🔍 **Gist 진단 패널:** 회원정보 미로드 시 10단계 진단 UI 자동 표시",
            "📈 **게이지 차트:** 투명 다크 배경, 4단계 구간 색상, Outfit 폰트 적용",
            "👑 **관리자 통계:** DAU/WAU를 3색 그라디언트 카드로 시각화",
            "🧪 **Unit Tests:** test_ml_engine.py 8개 테스트 케이스 (차원 불일치, 모델 없음, 정상 추론 등)",
        ],
        "schema_min": 5
    },
    {
        "version": "12.3.1",
        "date": "2026-02-14",
        "type": "major",
        "title": "Absolute Defense: Fix Import Issue",
        "items": [
            "🛡️ **Logic Isolation:** Import 시점의 UI 의존성 제거",
            "⚙️ **Robust Config:** 보급로(PRIME_TG_JOIN_URL) 변수 확립",
            "🚦 **Integrity Gate:** 무결성 검증 로직 함수화",
        ],
        "schema_min": 5
    }
]

# ----------------- 2. 핵심 로직 (Core) -----------------

def _get_conf(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val: return str(val)
    try:
        import streamlit as st
        if key in st.secrets: return str(st.secrets[key])
        if "core" in st.secrets and key in st.secrets["core"]: return str(st.secrets["core"][key])
    except Exception: pass  # [v20.6.4]
    return default

def _parse_version(v_str: str) -> Tuple[int, ...]:
    try: return tuple(map(int, (v_str.split('.'))))
    except (ValueError, TypeError): return (0, 0, 0)  # [v20.6.4]

# [📍 핵심 보급품] 앱 버전
APP_VERSION = CHANGELOG[0]["version"] if CHANGELOG else "18.0.0"
VERSION_TUPLE = _parse_version(APP_VERSION)

# [📍 핵심 보급품] 텔레그램 조인 URL
PRIME_TG_JOIN_URL = _get_conf("LDY_PRIME_JOIN_URL", "https://t.me/+DovDEluWnEJhOTY1")

def get_latest_log() -> Optional[Dict]:
    return CHANGELOG[0] if CHANGELOG else None

# [📍 핵심 보급품] 버전 라벨 함수
def get_version_label(include_build: bool = True) -> str:
    if include_build:
        return APP_VERSION
    return f"{VERSION_TUPLE[0]}.{VERSION_TUPLE[1]}"

def validate_integrity() -> bool:
    env_ver = _get_conf("LDY_APP_VERSION", APP_VERSION)
    if env_ver != APP_VERSION:
        logger.critical(f"🚨 VERSION CORRUPTION: Environment({env_ver}) != Core({APP_VERSION})")
        return False
    return True

# ----------------- 3. 유틸리티 -----------------

def _hex_to_rgb(hex_color: str) -> str:
    """#3B82F6 → '59,130,246' (CSS rgba용)"""
    h = hex_color.lstrip('#')
    try:
        return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
    except (ValueError, IndexError):  # [v20.6.4]
        return "100,100,100"

# ----------------- 4. UI 렌더링 (Streamlit 기반) -----------------

def show_toast_notification():
    import streamlit as st
    if "has_seen_version_toast" not in st.session_state:
        latest = get_latest_log()
        if latest: st.toast(f"🚀 v{APP_VERSION} 업데이트!", icon="💎")
        st.session_state["has_seen_version_toast"] = True


def render_sidebar_version_badge():
    """[v18.0] 사이드바 버전 뱃지 — 프리미엄 디자인"""
    import streamlit as st
    latest = get_latest_log()
    ver_type = latest['type'] if latest else "patch"
    ver_date = latest.get('date', '') if latest else ''
    item_count = len(latest.get('items', [])) if latest else 0

    colors = {"major": "#3B82F6", "minor": "#06B6D4", "patch": "#10B981"}
    bg_color = colors.get(ver_type, "#64748B")
    type_label = {"major": "MAJOR", "minor": "MINOR", "patch": "PATCH"}.get(ver_type, "UPDATE")

    st.sidebar.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(30,30,30,0.95), rgba(20,20,30,0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 4px solid {bg_color};
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 8px;
            font-family: 'Outfit', -apple-system, sans-serif;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.65rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.08em;">
                    LDY Pro Trader
                </span>
                <span style="
                    font-size:0.6rem; font-weight:600; color:{bg_color};
                    background:rgba({_hex_to_rgb(bg_color)},0.12);
                    padding:2px 8px; border-radius:10px;
                    letter-spacing:0.05em;
                ">{type_label}</span>
            </div>
            <div style="font-size:1.3rem; font-weight:700; color:#F1F5F9; letter-spacing:-0.02em;">
                v{APP_VERSION}
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:0.68rem; color:rgba(255,255,255,0.35);">
                <span>{ver_date}</span>
                <span>{item_count} changes</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_recent_updates(limit: int = 5):
    """[v18.0] 업데이트 노트 — 타임라인 스타일"""
    import streamlit as st

    st.markdown("""
        <style>
        .update-timeline {
            position: relative;
            padding-left: 28px;
        }
        .update-timeline::before {
            content: '';
            position: absolute;
            left: 8px;
            top: 4px;
            bottom: 0;
            width: 2px;
            background: linear-gradient(180deg, #3B82F6, rgba(59,130,246,0.05));
            border-radius: 1px;
        }
        .update-entry {
            position: relative;
            margin-bottom: 28px;
            padding-left: 18px;
        }
        .update-entry::before {
            content: '';
            position: absolute;
            left: -24px;
            top: 6px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 2px solid #3B82F6;
            background: #0E1117;
        }
        .update-entry.latest::before {
            background: #3B82F6;
            box-shadow: 0 0 10px rgba(59,130,246,0.5);
        }
        .update-ver {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 1.05rem;
            font-weight: 700;
            color: #F1F5F9;
        }
        .update-title {
            font-size: 0.88rem;
            color: #94A3B8;
            margin-top: 2px;
        }
        .update-date {
            font-size: 0.7rem;
            color: #475569;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 2px;
        }
        .update-tag {
            display: inline-block;
            font-size: 0.58rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 10px;
            margin-left: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            vertical-align: middle;
        }
        .tag-major { background: rgba(59,130,246,0.15); color: #3B82F6; }
        .tag-minor { background: rgba(6,182,212,0.15); color: #06B6D4; }
        .tag-patch { background: rgba(16,185,129,0.15); color: #10B981; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧩 System Intelligence Updates")

    # 타임라인 렌더링
    html_parts = ['<div class="update-timeline">']

    for i, log in enumerate(CHANGELOG[:limit]):
        ver = log['version']
        date = log.get('date', '')
        title = log.get('title', '')
        ver_type = log.get('type', 'patch')
        tag_class = f"tag-{ver_type}"
        latest_class = "latest" if i == 0 else ""
        n_items = len(log.get('items', []))

        html_parts.append(f"""
        <div class="update-entry {latest_class}">
            <div>
                <span class="update-ver">v{ver}</span>
                <span class="update-tag {tag_class}">{ver_type}</span>
            </div>
            <div class="update-date">{date} · {n_items} changes</div>
            <div class="update-title">{title}</div>
        </div>
        """)

    html_parts.append('</div>')
    st.markdown(''.join(html_parts), unsafe_allow_html=True)

    # 상세 내용은 Expander
    for i, log in enumerate(CHANGELOG[:limit]):
        with st.expander(f"v{log['version']} — {log['title']}", expanded=(i == 0)):
            for item in log['items']:
                st.markdown(item)
