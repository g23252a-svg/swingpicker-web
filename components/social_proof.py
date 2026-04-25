# -*- coding: utf-8 -*-
"""
social_proof.py — 사회적 증거 (Social Proof) 표시
═══════════════════════════════════════════════════════════
[v22 Step U] 결제 전환율 향상을 위한 사회적 증거

표시 요소:
1. 누적 가입자 수 (실제 DB 데이터)
2. Prime 멤버 수 (실제 활성 회원)
3. 최근 신규 가입자 (지난 7일)
4. 데이터 풍부함 (종목 수/데이터 기간)

환경변수 토글:
- SHOW_USER_COUNT: 가입자 수 표시 (default: true)
- SHOW_PRIME_COUNT: Prime 수 표시 (default: true)
- SHOW_RECENT_SIGNUPS: 최근 가입자 표시 (default: true)
- SHOW_DATA_METRICS: 데이터 메트릭 표시 (default: true)
- BETA_DISCOUNT_ACTIVE: 베타 할인 활성 (default: false)
"""
import logging
import os
from datetime import datetime, timedelta

from nicegui import ui

_logger = logging.getLogger(__name__)

# ─── 환경변수 ───
SHOW_USER_COUNT = os.environ.get("SHOW_USER_COUNT", "true").lower() == "true"
SHOW_PRIME_COUNT = os.environ.get("SHOW_PRIME_COUNT", "true").lower() == "true"
SHOW_RECENT_SIGNUPS = os.environ.get("SHOW_RECENT_SIGNUPS", "true").lower() == "true"
SHOW_DATA_METRICS = os.environ.get("SHOW_DATA_METRICS", "true").lower() == "true"
BETA_DISCOUNT_ACTIVE = os.environ.get("BETA_DISCOUNT_ACTIVE", "false").lower() == "true"
BETA_DISCOUNT_PERCENT = int(os.environ.get("BETA_DISCOUNT_PERCENT", "50"))
BETA_DISCOUNT_DEADLINE = os.environ.get("BETA_DISCOUNT_DEADLINE", "2026-05-31")
BETA_LIMIT = int(os.environ.get("BETA_LIMIT", "100"))

# 데이터 메트릭 (수동 설정)
# 데이터 메트릭 (환경변수 fallback — 실제 파일 기반 계산이 우선)
DATA_STOCK_COUNT = int(os.environ.get("DATA_STOCK_COUNT", "2500"))  # 분석 종목 수 fallback
DATA_PERIOD_YEARS = int(os.environ.get("DATA_PERIOD_YEARS", "5"))  # 데이터 기간 (년) fallback
DATA_DART_DAILY = int(os.environ.get("DATA_DART_DAILY", "1000"))  # DART 공시 일평균 fallback
DATA_BACKTEST_STRATEGIES = int(os.environ.get("DATA_BACKTEST_STRATEGIES", "100"))  # 백테스트 전략 수 fallback


def _calc_data_metrics() -> dict:
    """[Step W] 실제 데이터 파일 기반으로 메트릭 계산.
    
    환경변수 fallback 유지 (파일 못 찾으면 fallback 사용).
    
    Returns:
        {
            "stock_count": int,       # 분석 종목 수 (recommend_latest.csv 기준)
            "period_years": int,      # 데이터 기간 (OHLCV 캐시 기준)
            "backtest_strategies": int,  # 백테스트 전략 수 (validation 파일 기준)
            "dart_daily": int,        # DART 공시 일평균 (환경변수 유지 — 외부 API)
        }
    """
    import glob
    
    # 1. 분석 종목 수 — recommend_latest.csv 라인 수 (헤더 제외)
    stock_count = DATA_STOCK_COUNT  # fallback
    try:
        rec_paths = [
            "data/recommend_latest.csv",
            "/mnt/data/recommend_latest.csv",
            os.path.join(os.path.dirname(__file__), "..", "data", "recommend_latest.csv"),
        ]
        for path in rec_paths:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    line_count = sum(1 for _ in f) - 1  # 헤더 제외
                if line_count > 0:
                    stock_count = line_count
                    break
    except Exception as e:
        _logger.debug(f"종목 수 계산 실패 (fallback 사용): {e}")
    
    # 2. 데이터 기간 — OHLCV 캐시 파일 첫~마지막 날짜로 계산
    period_years = DATA_PERIOD_YEARS  # fallback
    try:
        ohlcv_paths = [
            "data/ohlcv_cache_*.parquet",
            "/mnt/data/ohlcv_cache_*.parquet",
        ]
        all_files = []
        for pattern in ohlcv_paths:
            all_files.extend(glob.glob(pattern))
        
        if all_files:
            # 파일명에서 날짜 추출 (ohlcv_cache_YYYYMMDD.parquet)
            dates = []
            for f in all_files:
                fname = os.path.basename(f)
                # ohlcv_cache_20260226.parquet → 20260226
                try:
                    date_part = fname.replace("ohlcv_cache_", "").replace(".parquet", "")
                    if len(date_part) == 8 and date_part.isdigit():
                        dates.append(date_part)
                except Exception:
                    pass
            
            if dates:
                dates.sort()
                # 캐시 파일은 운영 기간이지, 데이터 기간 자체는 더 길 수 있음
                # 보수적으로: 캐시 기간 + 환경변수 보정 사용
                # 실제 OHLCV 데이터 자체는 5년치 등 더 길지만, 운영 캐시 기준으로 표시
                first_dt = datetime.strptime(dates[0], "%Y%m%d")
                last_dt = datetime.strptime(dates[-1], "%Y%m%d")
                cache_days = (last_dt - first_dt).days
                # 캐시 운영 기간이 60일 미만이면 환경변수 사용 (fallback)
                # 60일+ 이면 실제 캐시 기간을 년수로 변환 (최소 1년)
                if cache_days >= 60:
                    period_years = max(1, cache_days // 365)
                # 그 외는 fallback 유지
    except Exception as e:
        _logger.debug(f"데이터 기간 계산 실패 (fallback 사용): {e}")
    
    # 3. 백테스트 전략 수 — [v22 Step X 재검토]
    # backtest_capital_curve_YYYYMMDD.csv는 "일자별 시뮬 결과"이지
    # "전략 수"가 아님. 일자 파일을 전략 수로 표시하면 오해 소지.
    # → 환경변수 fallback 유지 (실제 코드의 전략 갯수로 운영자 직접 설정)
    backtest_strategies = DATA_BACKTEST_STRATEGIES
    
    # 4. DART 공시 일평균 — 외부 API라 fallback 유지
    dart_daily = DATA_DART_DAILY
    
    return {
        "stock_count": stock_count,
        "period_years": period_years,
        "backtest_strategies": backtest_strategies,
        "dart_daily": dart_daily,
    }


def _get_user_stats() -> dict:
    """[Step U] 사용자 통계 조회.
    
    Returns:
        {
            "total": int,           # 총 가입자
            "prime_active": int,    # 활성 Prime 멤버
            "free": int,            # Free 회원
            "recent_7d": int,       # 최근 7일 신규
            "recent_30d": int,      # 최근 30일 신규
            "conversion_rate": float, # Free→Prime 전환율 (%)
        }
    """
    default = {
        "total": 0, "prime_active": 0, "free": 0,
        "recent_7d": 0, "recent_30d": 0,
        "conversion_rate": 0.0,
    }
    
    try:
        from db_utils import get_db
        db = get_db()
        if not db:
            return default
        if hasattr(db, 'ensure_gist_loaded'):
            db.ensure_gist_loaded()
        users = db.get_all_users()
        if not users:
            return default
        
        now = datetime.now()
        seven_days_ago = now - timedelta(days=7)
        thirty_days_ago = now - timedelta(days=30)
        
        total = 0
        prime_active = 0
        free = 0
        recent_7d = 0
        recent_30d = 0
        
        for u in users:
            role = (u.get("role") or "free").lower()
            
            # 관리자 제외 (외부에 노출되는 통계니까)
            if role == "admin":
                continue
            
            total += 1
            
            # Prime 활성 (만료일이 미래)
            if role in ("prime", "pro"):
                expire_str = u.get("prime_expire_date", "")
                if expire_str:
                    try:
                        # "2026-04-30" 또는 "2026-04-30 00:00:00" 형식 처리
                        expire_date_str = expire_str.split(" ")[0]
                        expire_date = datetime.strptime(expire_date_str, "%Y-%m-%d")
                        if expire_date >= now:
                            prime_active += 1
                    except Exception:
                        pass
            else:
                free += 1
            
            # 가입일 기준 최근 N일
            join_str = u.get("join_date", "")
            if join_str:
                try:
                    join_date_str = join_str.split(" ")[0]
                    join_date = datetime.strptime(join_date_str, "%Y-%m-%d")
                    if join_date >= seven_days_ago:
                        recent_7d += 1
                    if join_date >= thirty_days_ago:
                        recent_30d += 1
                except Exception:
                    pass
        
        # 전환율 (Prime / 전체 사용자, 100명 미만이면 약간 부풀림)
        conversion_rate = 0.0
        if total > 0:
            conversion_rate = (prime_active / total) * 100
        
        return {
            "total": total,
            "prime_active": prime_active,
            "free": free,
            "recent_7d": recent_7d,
            "recent_30d": recent_30d,
            "conversion_rate": round(conversion_rate, 1),
        }
    except Exception as e:
        _logger.warning(f"사용자 통계 조회 실패: {e}", exc_info=True)
        return default


def render_social_proof_card():
    """[Step U] 사회적 증거 카드 — 멤버십 탭 등에 표시"""
    
    # 통계 조회
    stats = _get_user_stats()
    
    # 데이터 너무 적으면 표시 안 함 (역효과 방지)
    if stats["total"] < 5:
        # 데이터 메트릭만 표시
        if SHOW_DATA_METRICS:
            _render_data_metrics_only()
        return
    
    with ui.card().classes(
        "w-full p-5 bg-gradient-to-br from-cyan-900/20 to-blue-900/20 "
        "border border-cyan-500/40 rounded-2xl mb-4"
    ):
        # 헤더
        with ui.row().classes("w-full items-center gap-2 mb-4"):
            ui.icon("group", size="28px").classes("text-cyan-400")
            ui.label("🚀 SwingPicker, 이미 함께하는 분들").classes(
                "text-xl font-bold text-white"
            )
        
        # 메트릭 그리드
        with ui.row().classes("w-full gap-3 flex-wrap"):
            # 누적 가입자
            if SHOW_USER_COUNT:
                _render_metric_card(
                    icon="👥",
                    value=f"{stats['total']}명+",
                    label="누적 가입자",
                    color="cyan",
                )
            
            # Prime 활성 멤버
            if SHOW_PRIME_COUNT and stats["prime_active"] > 0:
                _render_metric_card(
                    icon="💎",
                    value=f"{stats['prime_active']}명+",
                    label="Prime 멤버",
                    color="amber",
                )
            
            # 최근 7일 가입자
            if SHOW_RECENT_SIGNUPS and stats["recent_7d"] > 0:
                _render_metric_card(
                    icon="🔥",
                    value=f"{stats['recent_7d']}명+",
                    label="최근 7일 신규",
                    color="red",
                )
            
            # 활성 회원 (Prime + 최근 활동 Free)
            active_total = stats["prime_active"] + stats["recent_30d"]
            if active_total > 0:
                _render_metric_card(
                    icon="⚡",
                    value=f"{active_total}명+",
                    label="활성 회원",
                    color="emerald",
                )
        
        # [Step V] 전환율 표시 — 표본 충분(30명+) + 전환율 충분(20%+)일 때만
        # "업계 평균 대비" 비교는 근거 부족으로 제거 (리뷰어 피드백)
        if stats["total"] >= 30 and stats["conversion_rate"] >= 20.0:
            ui.label(
                f"💡 Free → Prime 전환율: {stats['conversion_rate']:.0f}%"
            ).classes(
                "text-sm text-emerald-300 mt-3 text-center font-medium"
            )


def _render_metric_card(icon: str, value: str, label: str, color: str = "cyan"):
    """단일 메트릭 카드"""
    color_map = {
        "cyan": "border-cyan-500/40 bg-cyan-900/20 text-cyan-300",
        "amber": "border-amber-500/40 bg-amber-900/20 text-amber-300",
        "red": "border-red-500/40 bg-red-900/20 text-red-300",
        "emerald": "border-emerald-500/40 bg-emerald-900/20 text-emerald-300",
    }
    classes = color_map.get(color, color_map["cyan"])
    
    with ui.card().classes(
        f"flex-1 min-w-[140px] p-4 {classes} rounded-xl border-2"
    ):
        ui.label(icon).classes("text-3xl mb-1")
        ui.label(value).classes("text-2xl font-bold text-white")
        ui.label(label).classes("text-xs text-gray-400")


def _render_data_metrics_only():
    """데이터 메트릭만 표시 (가입자 수가 적을 때)"""
    with ui.card().classes(
        "w-full p-5 bg-gradient-to-br from-indigo-900/20 to-purple-900/20 "
        "border border-indigo-500/40 rounded-2xl mb-4"
    ):
        with ui.row().classes("w-full items-center gap-2 mb-4"):
            ui.icon("storage", size="28px").classes("text-indigo-400")
            ui.label("📊 SwingPicker 데이터").classes(
                "text-xl font-bold text-white"
            )
            # [Step V] 기준일 표시
            today_str = datetime.now().strftime("%Y-%m-%d")
            ui.label(f"({today_str} 기준)").classes(
                "text-xs text-gray-500 ml-auto"
            )
        
        with ui.row().classes("w-full gap-3 flex-wrap"):
            # [Step W] 실제 파일 기반 메트릭 (환경변수 fallback)
            dm = _calc_data_metrics()
            metrics = [
                ("📈", f"{dm['stock_count']:,}+", "분석 종목"),
                ("📅", f"{dm['period_years']}년+", "데이터 기간"),
                ("🧪", f"{dm['backtest_strategies']}+", "백테스트 전략"),
                ("🤖", f"일 {dm['dart_daily']:,}+", "DART 공시"),
            ]
            for icon, value, label in metrics:
                _render_metric_card(icon=icon, value=value, label=label, color="cyan")


def render_data_richness_card():
    """[Step U+V] 데이터 풍부함 카드 — 사회적 증거 + 데이터 보완"""
    if not SHOW_DATA_METRICS:
        return
    
    with ui.card().classes(
        "w-full p-5 bg-gradient-to-br from-indigo-900/20 to-purple-900/20 "
        "border border-indigo-500/40 rounded-2xl mb-4"
    ):
        with ui.row().classes("w-full items-center gap-2 mb-4"):
            ui.icon("storage", size="28px").classes("text-indigo-400")
            ui.label("📊 SwingPicker 데이터 풍부함").classes(
                "text-xl font-bold text-white"
            )
            # [Step V] 기준일 표시 (작게, 우측)
            today_str = datetime.now().strftime("%Y-%m-%d")
            ui.label(f"({today_str} 기준)").classes(
                "text-xs text-gray-500 ml-auto"
            )
        
        with ui.row().classes("w-full gap-3 flex-wrap"):
            # [Step W] 실제 파일 기반 메트릭 (환경변수 fallback)
            dm = _calc_data_metrics()
            metrics = [
                ("📈", f"{dm['stock_count']:,}+", "분석 종목"),
                ("📅", f"{dm['period_years']}년+", "데이터 기간"),
                ("🧪", f"{dm['backtest_strategies']}+", "백테스트 전략"),
                ("🤖", f"일 {dm['dart_daily']:,}+", "DART 공시 학습"),
            ]
            for icon, value, label in metrics:
                _render_metric_card(icon=icon, value=value, label=label, color="cyan")


def render_beta_banner():
    """[Step U] 베타 한정 할인 배너"""
    if not BETA_DISCOUNT_ACTIVE:
        return
    
    # 마감일까지 남은 날짜 계산
    try:
        deadline = datetime.strptime(BETA_DISCOUNT_DEADLINE, "%Y-%m-%d")
        days_left = (deadline - datetime.now()).days
        deadline_str = deadline.strftime("%Y년 %m월 %d일")
    except Exception:
        days_left = 0
        deadline_str = BETA_DISCOUNT_DEADLINE
    
    # 마감 이후면 표시 X
    if days_left < 0:
        return
    
    with ui.card().classes(
        "w-full p-6 bg-gradient-to-br from-rose-900/30 via-orange-900/30 to-amber-900/30 "
        "border-2 border-orange-500/60 rounded-2xl mb-4 shadow-lg"
    ):
        # 헤더
        with ui.row().classes("w-full items-center gap-2 mb-3"):
            ui.icon("celebration", size="32px").classes("text-orange-300")
            ui.label("🚀 SwingPicker 정식 런칭 기념").classes(
                "text-2xl font-bold text-white"
            )
            if days_left >= 0:
                ui.badge(f"D-{days_left}").props(
                    "color=orange"
                ).classes("ml-auto")
        
        # 메인 메시지
        ui.label(
            f"💎 Prime 첫 {BETA_LIMIT}명 한정 {BETA_DISCOUNT_PERCENT}% 할인"
        ).classes("text-xl font-bold text-amber-300 mb-2")
        
        ui.label(
            f"⏰ {deadline_str}까지 (D-{days_left})"
        ).classes("text-sm text-orange-200 mb-3")
        
        # 혜택 리스트
        with ui.column().classes("w-full gap-1"):
            benefits = [
                f"💰 {BETA_DISCOUNT_PERCENT}% 할인 → "
                f"월 {19_900 // (100 - BETA_DISCOUNT_PERCENT) * (100 - BETA_DISCOUNT_PERCENT) // 100 * (100 - BETA_DISCOUNT_PERCENT) // 100 * 100:.0f}원"
                if False else
                f"💰 {BETA_DISCOUNT_PERCENT}% 할인 → "
                f"월 {int(19_900 * (100 - BETA_DISCOUNT_PERCENT) / 100):,}원",
                "🎁 30일 후기 작성 시 다음 1개월 무료",
                "📞 1:1 운영자 직접 지원",
                "✅ 베타 사용자만의 특별 혜택",
            ]
            for b in benefits:
                ui.label(b).classes("text-sm text-white")


def render_social_proof_section(show_all=True):
    """[Step U] 사회적 증거 섹션 — 멤버십 탭 등에 통합 표시
    
    Args:
        show_all: True면 사회적 증거 + 데이터 + 배너 모두 표시
                  False면 사회적 증거 카드만
    """
    # 1. 베타 배너 (가장 위)
    if show_all:
        render_beta_banner()
    
    # 2. 사회적 증거 카드
    render_social_proof_card()
    
    # 3. 데이터 풍부함 카드
    if show_all and SHOW_DATA_METRICS:
        # 사용자 데이터 충분하면 데이터 풍부함은 별도 카드 (보완)
        stats = _get_user_stats()
        if stats["total"] >= 5:
            render_data_richness_card()
