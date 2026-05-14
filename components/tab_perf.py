# -*- coding: utf-8 -*-
"""
tab_perf.py — 📈 시스템 성과 추세 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════════════
[v22 Step AK+AL+AM+AN] 전면 리팩토링 — 75 → 99점 목표

개선 사항 (Step AK):
1. ✅ 면책 + 백테스트 한계 안내 (법적 안전)
2. ✅ 사용자 친화 라벨 (METHOD/TOPK/보유기간)
3. ✅ 위험 강조 (MDD 빨간 카드)
4. ✅ 모바일 반응형 (높이/필터)
5. ✅ Research Workbench 통합 정리
6. ✅ 지표별 툴팁 + 설명

추가 개선 (Step AL):
7. ✅ latest CSV 중복 제거 (drop_duplicates)
8. ✅ 모바일 grid 실제 반응형 (grid-cols-2 md:grid-cols-3)
9. ✅ 차트에 MDD 추세 라인 추가 (빨간 점선)
10. ✅ 비용 차감 후 추정 수익률 (기본 0.4%)

추가 개선 (Step AM):
11. ✅ 메트릭 6종 — 승률/수익률/비용반영/도달률/낙폭/표본
12. ✅ 거래비용 가정 select (0.3%/0.4%/0.5%/0.7%)
13. ✅ KOSPI 알파 구현 (bench_cache_latest.json 활용)
14. ✅ 차트 보기 모드 (성과/위험/도달률/시장비교)

추가 개선 (Step AN):
15. ✅ _safe_float() helper — sel_cost.value 안전 변환
16. ✅ 차트 모드별 해설 (CHART_MODE_EXPLANATIONS)
17. ✅ KOSPI 근사치 명시 (행별 정확한 알파는 향후 백테스트 단계 작업)

향후 작업 (백엔드 단계 — 99점 → 100점):
- rank_validation_summary CSV에 KOSPI_RET_%/ALPHA_% 컬럼 추가
- 거래별 비용 차감 후 평균 (현재는 평균에서 일괄 차감)
"""
import glob
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from nicegui import ui

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

_logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ═══════════════════════════════════════════════════
#  사용자 친화 라벨
# ═══════════════════════════════════════════════════
METHOD_LABELS = {
    "ELITE_SCORE": "🏆 ELITE 점수 (검증 통과 종목)",
    "FINAL_SCORE": "🎯 최종 점수 (4축 통합)",
    "DISPLAY_SCORE": "📊 종합 점수 (3축 평균)",
    "RANK_SCORE": "📈 랭킹 점수 (내부 선별)",
    "AI_SCORE": "🤖 AI 점수 (단독)",
}

METHOD_DESCRIPTIONS = {
    "ELITE_SCORE": "구조 + 타이밍 + AI 3축 + RR + 밸런스 종합 — 가장 보수적 선별",
    "FINAL_SCORE": "ELITE에 ROUTE 가중치 추가한 최종 랭킹 지표",
    "DISPLAY_SCORE": "사용자에게 보이는 종합 점수 (3축 평균)",
    "RANK_SCORE": "내부 Top 선별용 — 실제 사용자 노출은 ELITE 우선",
    "AI_SCORE": "AI 컴포넌트만 분리 — 다른 지표 비교용",
}

TOPK_LABELS = {
    1: "상위 1개 (가장 보수적)",
    3: "상위 3개 (소수 정예)",
    5: "상위 5개 (균형)",
    10: "상위 10개 (분산)",
}

HOLD_LABELS = {
    1: "1영업일 (당일 매도)",
    3: "3영업일 (3일 보유)",
    5: "5영업일 (1주일)",
    10: "10영업일 (2주일)",
}

# [Step AL] 거래 비용 상수 — 슬리피지 + 수수료 + 세금 합산 추정
# 한국 주식 기준: 매수 수수료 0.015% + 매도 수수료 0.015% + 거래세 0.18% + 슬리피지 ~0.1%
# 단순화: 왕복 0.4% 가정 (보수적 추정)
DEFAULT_COST_PCT = 0.4
COST_DESCRIPTION = (
    "왕복 거래비용 추정치 — "
    "매수/매도 수수료 + 거래세(0.18%) + 슬리피지 합산 (~0.4%)"
)

# [Step AM] 거래비용 옵션 (사용자 선택)
COST_OPTIONS = {
    0.3: "0.3% (저비용)",
    0.4: "0.4% (기본 — 보수)",
    0.5: "0.5% (보수적)",
    0.7: "0.7% (스캘핑/고빈도)",
}

# [Step AN] 차트 모드별 해설 — 사용자가 모드 의미 즉시 이해
CHART_MODE_EXPLANATIONS = {
    "performance": (
        "📊 성과 보기: 승률(막대)과 평균 수익률(라인)의 일별 추세를 함께 봅니다. "
        "수익률 라인이 0% 위에 머무르면 안정적 성과."
    ),
    "risk": (
        "⚠️ 위험 보기: 평균 낙폭과 최악 낙폭의 일별 추세입니다. "
        "낙폭이 작을수록 실전 유지가 수월하며, 최악 낙폭은 손절가 설정의 기준."
    ),
    "hit": (
        "🎯 도달률 보기: 보유 중 +2% / +5% 한 번이라도 찍은 종목 비율입니다. "
        "익절 타이밍 설계에 참고하세요."
    ),
    "market": (
        "📈 시장 비교: 알파(=전략 수익률 - KOSPI 동기간 수익률)가 양수면 "
        "시장 평균을 초과한 성과입니다. 점선 위에 있는 날이 많을수록 일관된 알파."
    ),
}


def _safe_float(v, default: float = DEFAULT_COST_PCT) -> float:
    """[Step AN] 안전한 float 변환 — sel_cost.value가 str로 들어오는 케이스 방어.
    
    NiceGUI select에서 dict 옵션 사용 시 대부분 key가 그대로 오지만,
    환경에 따라 label 또는 string으로 올 수 있어 방어.
    """
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        # "0.4%" 같은 형식도 처리
        s = str(v).strip().replace("%", "").split()[0]
        return float(s)
    except (ValueError, IndexError, AttributeError):
        return default


# ═══════════════════════════════════════════════════
#  [Step AM] KOSPI 벤치마크 로더
# ═══════════════════════════════════════════════════
def _load_bench_cache() -> dict:
    """[Step AM] bench_cache_latest.json 로드 — KOSPI/KOSDAQ 보유기간별 수익률.
    
    파일 형식:
        {
            "KOSPI": {"1": -0.0, "3": 1.36, "5": 4.58, "10": 10.53, "20": 19.06},
            "KOSDAQ": {...}
        }
    
    Returns:
        {"KOSPI": {1: -0.0, 5: 4.58, ...}, ...} (정수 키로 변환)
        파일 없으면 빈 dict
    """
    import json
    
    dirs_to_try = [
        DATA_DIR,
        os.path.join(os.getcwd(), "data"),
        "data",
    ]
    
    for d in dirs_to_try:
        path = os.path.join(d, "bench_cache_latest.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                # 키를 string → int 변환 (H(영업일) 매칭용)
                result = {}
                for index_name, hold_data in raw.items():
                    if isinstance(hold_data, dict):
                        result[index_name] = {
                            int(k): float(v)
                            for k, v in hold_data.items()
                            if str(k).isdigit()
                        }
                _logger.info(
                    f"📊 KOSPI 벤치마크 로드: {list(result.keys())} "
                    f"(보유기간: {sorted(result.get('KOSPI', {}).keys())})"
                )
                return result
            except Exception as e:
                _logger.warning(f"bench_cache 로드 실패 ({path}): {e}")
                return {}
    
    _logger.info("bench_cache_latest.json 없음 — KOSPI 알파 미표시")
    return {}


def _get_kospi_return(bench_data: dict, hold_days: int) -> float:
    """[Step AM] 특정 보유기간의 KOSPI 수익률 추출.
    
    Returns: KOSPI 수익률(%) 또는 None
    """
    if not bench_data or "KOSPI" not in bench_data:
        return None
    return bench_data["KOSPI"].get(int(hold_days))


def _now_kst():
    return datetime.now(KST)


# ═══════════════════════════════════════════════════
#  데이터 로딩
# ═══════════════════════════════════════════════════
def _load_history() -> pd.DataFrame:
    """rank_validation_summary_*.csv 파일 병합"""
    # [v21.3] DATA_DIR 폴백 — Railway Docker 대응
    dirs_to_try = [
        DATA_DIR,
        os.path.join(os.getcwd(), "data"),
        "data",
    ]
    target_dir = None
    for d in dirs_to_try:
        pattern = os.path.join(d, "rank_validation_summary_*.csv")
        if glob.glob(pattern):
            target_dir = d
            break

    if not target_dir:
        _logger.warning(f"⚠️ rank_validation_summary 파일 없음 (검색: {dirs_to_try})")
        return pd.DataFrame()

    pattern = os.path.join(target_dir, "rank_validation_summary_*.csv")
    all_files = sorted(glob.glob(pattern))
    _logger.info(f"📊 성과 데이터 {len(all_files)}개 파일 발견 ({target_dir})")

    dfs = []
    for f in all_files:
        try:
            base = os.path.basename(f)
            ds = base.replace("rank_validation_summary_", "").replace(".csv", "")
            d = pd.read_csv(f, encoding='utf-8-sig')
            if "latest" in ds:
                d['Date'] = pd.to_datetime(_now_kst().strftime("%Y-%m-%d"))
            else:
                d['Date'] = pd.to_datetime(ds, format="%Y%m%d")
            dfs.append(d)
        except Exception as e:
            _logger.warning(f"⚠️ 성과 파일 읽기 실패: {f} → {e}")

    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True).sort_values('Date')
    
    # [v22 Step AL] latest CSV 중복 제거
    # rank_validation_summary_latest.csv가 오늘 날짜 파일과 중복될 가능성 방어
    # Date + METHOD + TOPK + H 조합으로 unique 보장 (keep="last" — latest 우선)
    dedup_cols = [c for c in ["Date", "METHOD", "TOPK", "H(영업일)"]
                  if c in result.columns]
    if dedup_cols:
        before = len(result)
        result = result.drop_duplicates(subset=dedup_cols, keep="last")
        after = len(result)
        if before != after:
            _logger.info(
                f"📊 중복 제거: {before} → {after}행 ({before - after}건 제거)"
            )
    
    _logger.info(f"📊 성과 데이터 로드: {len(result)}행")
    return result


# ═══════════════════════════════════════════════════
#  면책 카드 (가장 중요)
# ═══════════════════════════════════════════════════
def _render_disclaimer_card():
    """[Step AK] 백테스트 한계 + 면책 안내 — 법적 안전 핵심"""
    with ui.card().classes(
        "w-full p-4 bg-amber-900/20 border border-amber-500/40 rounded-xl mb-4"
    ):
        with ui.row().classes("w-full items-start gap-3"):
            ui.label("⚠️").classes("text-2xl")
            with ui.column().classes("flex-1 gap-1"):
                ui.label("백테스트 결과 안내").classes(
                    "text-base font-bold text-amber-300"
                )
                ui.label(
                    "이 페이지는 과거 시장 데이터 기반 알고리즘 검증 결과입니다."
                ).classes("text-sm text-gray-200 mb-1")
                
                with ui.column().classes("gap-0.5 mt-1"):
                    for line in [
                        "• 실제 거래가 아닌 시뮬레이션 (paper trading)",
                        "• 슬리피지 / 수수료 / 세금 미반영",
                        "• 단기 보유 시뮬레이션 (1~10영업일)",
                        "• 시장 상황(강세/약세) 구분 없이 평균값 표시",
                        "• 과거 성과는 미래 수익을 보장하지 않습니다",
                    ]:
                        ui.label(line).classes("text-xs text-gray-300")
                
                ui.label(
                    "💡 모든 투자 판단과 그에 따른 손익은 전적으로 본인 책임입니다."
                ).classes("text-xs text-amber-200 mt-2 font-bold")


# ═══════════════════════════════════════════════════
#  메트릭 6종 카드
# ═══════════════════════════════════════════════════
def _render_metrics_grid(
    cdf: pd.DataFrame,
    cost_pct: float = DEFAULT_COST_PCT,
    bench_data: dict = None,
    hold_days: int = None,
):
    """[Step AK+AL+AM] 메트릭 6종 — 승률/수익률/비용반영/도달률/낙폭.
    
    Args:
        cdf: 필터링된 데이터프레임
        cost_pct: 거래비용 % (사용자 선택)
        bench_data: KOSPI 벤치마크 dict
        hold_days: 보유기간 (KOSPI 알파 매칭용)
    """
    if cdf.empty:
        return
    
    # 안전한 평균 계산
    def safe_mean(col):
        if col not in cdf.columns:
            return None
        try:
            v = cdf[col].mean()
            return None if pd.isna(v) else v
        except Exception:
            return None
    
    win_rate = safe_mean('WIN_RATE_%')
    avg_ret = safe_mean('AVG_RET_%')
    hit_5 = safe_mean('HIT_5%_%')
    avg_mdd = safe_mean('AVG_MDD_%')
    worst_mdd = safe_mean('WORST_MDD_%')
    total_n = cdf['TOTAL_N'].sum() if 'TOTAL_N' in cdf.columns else 0
    
    # [Step AL+AM] 비용 차감 후 추정 수익률 (사용자 선택 비용률)
    avg_ret_after_cost = None
    if avg_ret is not None:
        avg_ret_after_cost = avg_ret - cost_pct
    
    # [Step AM] KOSPI 알파 계산
    kospi_ret = None
    alpha = None
    if bench_data and hold_days is not None:
        kospi_ret = _get_kospi_return(bench_data, hold_days)
        if kospi_ret is not None and avg_ret is not None:
            alpha = avg_ret - kospi_ret
    
    ui.label("📊 핵심 지표 (선택 조건 기준)").classes(
        "text-sm font-bold text-cyan-300 mt-3 mb-2"
    )
    
    # [Step AL+AM] 메트릭 카드 — 모바일 2열, 데스크톱 3열 반응형
    with ui.grid().classes(
        "w-full gap-3 grid-cols-2 md:grid-cols-3"
    ):
        # 1. 평균 승률
        _render_metric_card(
            icon="📊", label="평균 승률",
            value=f"{win_rate:.1f}%" if win_rate is not None else "—",
            color="amber",
            tooltip="보유 기간 종료 시 진입가 대비 +1% 이상 종목 비율",
        )
        
        # 2. 평균 수익률 (총 수익률 — 비용 미반영)
        _render_metric_card(
            icon="💰", label="평균 수익률 (총)",
            value=f"{avg_ret:+.2f}%" if avg_ret is not None else "—",
            color="blue",
            tooltip="모든 포지션의 산술 평균 수익률 (수수료/세금 미반영)",
        )
        
        # 3. [Step AL+AM] 비용 반영 추정 — 사용자 선택 비용률
        _render_metric_card(
            icon="💵", label=f"비용 반영 ({cost_pct:.1f}%)",
            value=(
                f"{avg_ret_after_cost:+.2f}%"
                if avg_ret_after_cost is not None else "—"
            ),
            color="emerald",
            tooltip=(
                f"평균 수익률에서 왕복 거래비용 {cost_pct:.1f}% 차감.\n"
                f"실제로는 종목/시장에 따라 변동 가능."
            ),
        )
        
        # 4. 5% 도달률
        _render_metric_card(
            icon="🎯", label="5% 도달률",
            value=f"{hit_5:.1f}%" if hit_5 is not None else "—",
            color="green",
            tooltip="보유 중 한 번이라도 +5% 이상 찍은 종목 비율",
        )
        
        # 5. 평균 최대 낙폭 (위험 강조)
        _render_metric_card(
            icon="⚠️", label="평균 낙폭",
            value=f"{avg_mdd:+.2f}%" if avg_mdd is not None else "—",
            color="orange",
            tooltip="MDD: 보유 중 진입가 대비 최저점까지의 평균 낙폭",
        )
        
        # 6. 최악 낙폭 (위험 강조)
        _render_metric_card(
            icon="🔴", label="최악 낙폭",
            value=f"{worst_mdd:+.2f}%" if worst_mdd is not None else "—",
            color="red",
            tooltip="기간 중 가장 컸던 단일 포지션 낙폭 (최악의 케이스)",
        )
    
    # [Step AM] KOSPI 알파 카드 (있을 때만 별도 표시 — 강조)
    if alpha is not None and kospi_ret is not None:
        ui.label(
            f"📈 시장 비교 (KOSPI 동기간 {hold_days}영업일)"
        ).classes("text-sm font-bold text-cyan-300 mt-3 mb-2")
        
        with ui.grid().classes(
            "w-full gap-3 grid-cols-1 md:grid-cols-3"
        ):
            # KOSPI 동기간 수익률
            _render_metric_card(
                icon="📊", label="KOSPI 수익률",
                value=f"{kospi_ret:+.2f}%",
                color="cyan",
                tooltip=f"동일 보유기간({hold_days}영업일) KOSPI 평균 수익률",
            )
            
            # 전략 수익률
            _render_metric_card(
                icon="💰", label="전략 수익률",
                value=f"{avg_ret:+.2f}%" if avg_ret is not None else "—",
                color="blue",
                tooltip="동일 조건 백테스트 전략 평균 수익률 (총)",
            )
            
            # 알파 (전략 - KOSPI)
            alpha_color = "green" if alpha > 0 else "red"
            alpha_icon = "🚀" if alpha > 0 else "⚠️"
            _render_metric_card(
                icon=alpha_icon, label="알파 (시장 초과)",
                value=f"{alpha:+.2f}%p",
                color=alpha_color,
                tooltip=(
                    f"전략 수익률 - KOSPI 수익률 = {avg_ret:+.2f}% - {kospi_ret:+.2f}% "
                    f"= {alpha:+.2f}%p\n"
                    "양수면 시장 초과 성과(알파+), 음수면 시장 미달."
                ),
            )
        
        # 알파 해설
        alpha_msg = (
            "✅ 전략이 시장(KOSPI) 평균을 초과 — 알파(+) 발생"
            if alpha > 0 else
            "⚠️ 전략이 시장(KOSPI) 평균에 미치지 못함 — 알파(-)"
        )
        ui.label(alpha_msg).classes(
            f"text-xs italic mt-1 text-center "
            f"{'text-emerald-300' if alpha > 0 else 'text-red-300'}"
        )
        
        # [Step AN] 근사치 안내 — 행별 정확한 알파 미지원
        ui.label(
            "ℹ️ 현재 KOSPI 알파는 동일 보유기간 평균값 기반 근사치입니다. "
            "행별(검증일별) 정확한 알파는 백테스트 데이터에 "
            "KOSPI_RET_% / ALPHA_% 컬럼 추가 시 제공됩니다."
        ).classes(
            "text-[11px] text-gray-500 italic mt-2 text-center leading-relaxed"
        )
    
    # [Step AL+AM] 비용 안내 + 시장 비교 안내
    with ui.column().classes("w-full gap-1 mt-3"):
        ui.label(
            f"💡 '비용 반영'은 왕복 {cost_pct:.1f}%(매수/매도 수수료 + "
            f"거래세 0.18% + 슬리피지) 차감한 보수적 추정치입니다."
        ).classes("text-xs text-gray-400 leading-relaxed")
        
        if not bench_data:
            # KOSPI 데이터 없을 때
            ui.label(
                "📌 KOSPI 알파는 bench_cache_latest.json 데이터가 있을 때 자동 표시됩니다."
            ).classes("text-xs text-gray-500 italic leading-relaxed")
    
    # 표본 + 기간 정보
    with ui.row().classes("w-full justify-center gap-4 mt-2 flex-wrap"):
        ui.label(f"📅 표본: {int(total_n):,}거래").classes(
            "text-xs text-gray-500"
        )
        if 'Date' in cdf.columns and not cdf.empty:
            try:
                d_min = cdf['Date'].min()
                d_max = cdf['Date'].max()
                if isinstance(d_min, pd.Timestamp):
                    ui.label(
                        f"📆 기간: {d_min.strftime('%Y-%m-%d')} ~ "
                        f"{d_max.strftime('%Y-%m-%d')}"
                    ).classes("text-xs text-gray-500")
            except Exception:
                pass
        # 2% 도달률 안내
        ui.label(
            "💡 2% 도달률 등 추가 지표는 아래 Research Workbench에서 확인"
        ).classes("text-xs text-gray-600 italic")


def _render_metric_card(icon: str, label: str, value: str,
                        color: str, tooltip: str = ""):
    """[Step AK] 단일 메트릭 카드"""
    color_map = {
        "amber": ("border-amber-700/40", "text-amber-400", "text-amber-300"),
        "blue": ("border-blue-700/40", "text-blue-400", "text-blue-300"),
        "green": ("border-emerald-700/40", "text-emerald-400", "text-emerald-300"),
        "emerald": ("border-emerald-600/50", "text-emerald-300", "text-emerald-200"),
        "cyan": ("border-cyan-700/40", "text-cyan-400", "text-cyan-300"),
        "orange": ("border-orange-700/40", "text-orange-400", "text-orange-300"),
        "red": ("border-red-700/40", "text-red-400", "text-red-300"),
    }
    border, label_color, value_color = color_map.get(color, color_map["blue"])
    
    card = ui.card().classes(
        f"p-3 bg-[#1a1a2e] border {border} rounded-xl"
    )
    with card:
        with ui.row().classes("w-full items-center gap-1"):
            ui.label(icon).classes("text-base")
            ui.label(label).classes(f"text-xs {label_color} font-medium")
        ui.label(value).classes(
            f"text-xl font-bold {value_color} mt-1"
        )
    if tooltip:
        card.tooltip(tooltip)


# ═══════════════════════════════════════════════════
#  [Step AM] 차트 보기 모드별 빌더
# ═══════════════════════════════════════════════════
def _build_chart_by_mode(
    cdf: pd.DataFrame,
    mode: str,
    bench_data: dict,
    hold_days: int,
    col_win: str = 'WIN_RATE_%',
    col_ret: str = 'AVG_RET_%',
):
    """[Step AM] 모드별 Plotly figure 생성.
    
    Modes:
        - performance: 승률 + 평균 수익률 (기본)
        - risk: 평균 낙폭 + 최악 낙폭
        - hit: 2% / 5% 도달률
        - market: 전략 vs KOSPI 비교
    """
    if not PLOTLY_OK or cdf.empty:
        return None
    
    common_layout = dict(
        height=380,
        autosize=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
        hoverlabel=dict(
            bgcolor="#1a1a2e", font_size=13,
            font_color="white", bordercolor="#444",
        ),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    grid_props = dict(
        gridcolor='rgba(255,255,255,0.05)',
    )
    
    # ─── 모드 1: 성과 보기 ───
    if mode == "performance":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=cdf['Date'], y=cdf[col_win], name="승률(%)",
            marker_color='#FFA726', opacity=0.6,
            hovertemplate="<b>%{x}</b><br>승률: %{y:.1f}%<extra></extra>",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=cdf['Date'], y=cdf[col_ret], name="평균 수익률(%)",
            mode='lines+markers',
            line=dict(color='#29B6F6', width=3),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>수익률: %{y:.2f}%<extra></extra>",
        ), secondary_y=True)
        fig.add_hline(y=0, line_dash="dot",
                      line_color="rgba(255,255,255,0.3)", secondary_y=True)
        fig.update_layout(**common_layout)
        fig.update_yaxes(title_text="승률 (%)", range=[0, 100],
                         gridcolor='rgba(255,255,255,0.1)', secondary_y=False)
        fig.update_yaxes(title_text="수익률 (%)",
                         gridcolor='rgba(255,255,255,0.05)', secondary_y=True)
        fig.update_xaxes(**grid_props)
        return fig
    
    # ─── 모드 2: 위험 보기 ───
    elif mode == "risk":
        fig = go.Figure()
        if 'AVG_MDD_%' in cdf.columns:
            fig.add_trace(go.Scatter(
                x=cdf['Date'], y=cdf['AVG_MDD_%'], name="평균 낙폭(%)",
                mode='lines+markers',
                line=dict(color='#F59E0B', width=2),
                marker=dict(size=5),
                hovertemplate="<b>%{x}</b><br>평균 낙폭: %{y:.2f}%<extra></extra>",
            ))
        if 'WORST_MDD_%' in cdf.columns:
            fig.add_trace(go.Scatter(
                x=cdf['Date'], y=cdf['WORST_MDD_%'], name="최악 낙폭(%)",
                mode='lines+markers',
                line=dict(color='#EF4444', width=2, dash='dot'),
                marker=dict(size=5),
                hovertemplate="<b>%{x}</b><br>최악 낙폭: %{y:.2f}%<extra></extra>",
            ))
        fig.add_hline(y=0, line_dash="dot",
                      line_color="rgba(255,255,255,0.3)")
        fig.update_layout(**common_layout)
        fig.update_yaxes(title_text="낙폭 (%)", **grid_props)
        fig.update_xaxes(**grid_props)
        return fig
    
    # ─── 모드 3: 도달률 보기 ───
    elif mode == "hit":
        fig = go.Figure()
        if 'HIT_2%_%' in cdf.columns:
            fig.add_trace(go.Scatter(
                x=cdf['Date'], y=cdf['HIT_2%_%'], name="2% 도달률(%)",
                mode='lines+markers',
                line=dict(color='#22D3EE', width=2),
                marker=dict(size=5),
                hovertemplate="<b>%{x}</b><br>2% 도달률: %{y:.1f}%<extra></extra>",
            ))
        if 'HIT_5%_%' in cdf.columns:
            fig.add_trace(go.Scatter(
                x=cdf['Date'], y=cdf['HIT_5%_%'], name="5% 도달률(%)",
                mode='lines+markers',
                line=dict(color='#10B981', width=3),
                marker=dict(size=6),
                hovertemplate="<b>%{x}</b><br>5% 도달률: %{y:.1f}%<extra></extra>",
            ))
        fig.update_layout(**common_layout)
        fig.update_yaxes(title_text="도달률 (%)", range=[0, 100], **grid_props)
        fig.update_xaxes(**grid_props)
        return fig
    
    # ─── 모드 4: 시장 비교 (KOSPI vs 전략) ───
    elif mode == "market":
        fig = go.Figure()
        # 전략 평균 수익률
        fig.add_trace(go.Scatter(
            x=cdf['Date'], y=cdf[col_ret], name="전략 평균 수익률(%)",
            mode='lines+markers',
            line=dict(color='#29B6F6', width=3),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>전략: %{y:.2f}%<extra></extra>",
        ))
        # KOSPI 동기간 수익률 (수평선)
        kospi_ret = _get_kospi_return(bench_data, hold_days) if hold_days else None
        if kospi_ret is not None:
            fig.add_hline(
                y=kospi_ret,
                line_dash="dash",
                line_color="#A78BFA",
                line_width=2,
                annotation_text=f"KOSPI {hold_days}영업일 수익률 {kospi_ret:+.2f}%",
                annotation_position="top right",
                annotation_font_color="#A78BFA",
            )
        fig.add_hline(y=0, line_dash="dot",
                      line_color="rgba(255,255,255,0.3)")
        fig.update_layout(**common_layout)
        fig.update_yaxes(title_text="수익률 (%)", **grid_props)
        fig.update_xaxes(**grid_props)
        return fig
    
    return None


# ═══════════════════════════════════════════════════
#  메인 렌더링
# ═══════════════════════════════════════════════════
def _load_backtest_validation() -> dict:
    """backtest_validation_latest.json 로드 — shadow 섹션 읽기용."""
    import json
    for d in (DATA_DIR, os.path.join(os.path.dirname(__file__), "..", "data")):
        path = os.path.join(d, "backtest_validation_latest.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _logger.warning(f"backtest_validation_latest.json 로드 실패: {e}")
                return {}
    return {}


def _render_shadow_lab_card():
    """[v3.9.1] Shadow 실험실 — ENTRY_MODE / STRUCT risk shadow 측정 결과 표시.

    추천 로직을 바꾸지 않고 "바꿨다면 어땠을지"를 매일 자동 측정한 결과.
    measurement-only — production 적용 아님을 명시.
    """
    j = _load_backtest_validation()
    em = j.get("entry_mode_shadow", {})
    sr = j.get("struct_risk_shadow", {})

    # 둘 다 없거나 비활성이면 카드 자체를 안 그림
    if not (em.get("enabled") or sr.get("enabled")):
        return

    with ui.row().classes("w-full items-center gap-2 mb-2 mt-2"):
        ui.label("🧪").classes("text-2xl")
        with ui.column().classes("gap-0 flex-1"):
            ui.label("Shadow 실험실 (measurement-only)").classes(
                "text-lg font-bold text-purple-300"
            )
            ui.label(
                "추천 로직을 바꾸지 않고 '바꿨다면 어땠을지'를 매일 자동 측정한 "
                "결과입니다. 운영 추천에는 아직 반영되지 않습니다."
            ).classes("text-xs text-gray-400")

    # ─── ENTRY_MODE shadow ───
    if em.get("enabled"):
        with ui.card().classes(
            "w-full p-3 bg-[#1a1a2e] border border-purple-700/30 "
            "rounded-lg mb-2"
        ):
            ui.label("🎯 ENTRY_MODE shadow — 강한 종목 미체결 chase 회수").classes(
                "text-sm font-bold text-purple-200 mb-1"
            )
            if "extra_fills" in em:
                with ui.row().classes("w-full flex-wrap gap-x-6 gap-y-1"):
                    _shadow_stat("추가 체결", f"{em['extra_fills']}건")
                    _shadow_stat("WIN", f"{em['extra_wins']}건",
                                 good=em["extra_wins"] > 0)
                    _shadow_stat("LOSS", f"{em['extra_losses']}건",
                                 bad=em["extra_losses"] > 0)
                    _shadow_stat("합산 수익", f"{em.get('extra_sum_ret', 0):+.2f}%",
                                 good=em.get("extra_sum_ret", 0) > 0,
                                 bad=em.get("extra_sum_ret", 0) < 0)
                    _shadow_stat("평균 RR", f"{em.get('avg_rr_chase', 0):.2f}")
                _shadow_gate_label(em.get("production_candidate", False))
            else:
                ui.label("측정 데이터 누적 중 — 표본 부족").classes(
                    "text-xs text-gray-500"
                )
            ui.label(f"규칙: {em.get('rule', '')}").classes(
                "text-[10px] text-gray-600 mt-1 leading-tight"
            )

    # ─── STRUCT risk shadow ───
    if sr.get("enabled"):
        with ui.card().classes(
            "w-full p-3 bg-[#1a1a2e] border border-purple-700/30 "
            "rounded-lg mb-2"
        ):
            ui.label(
                "🛡️ STRUCT risk shadow — STRUCT 70~85 제외 시뮬"
            ).classes("text-sm font-bold text-purple-200 mb-1")
            if "delta_ev" in sr:
                with ui.row().classes("w-full flex-wrap gap-x-6 gap-y-1"):
                    _shadow_stat(
                        "ΔEV",
                        f"{sr['delta_ev']:+.2f}",
                        good=sr["delta_ev"] > 0,
                        bad=sr["delta_ev"] < 0,
                    )
                    _shadow_stat(
                        "Δ비승리 평균손익",
                        f"{sr.get('delta_non_win_avg_ret', 0):+.2f}",
                        good=sr.get("delta_non_win_avg_ret", 0) > 0,
                        bad=sr.get("delta_non_win_avg_ret", 0) < 0,
                    )
                    _shadow_stat(
                        "Top3 구성변경",
                        f"{sr.get('changed_pick_rate', 0) * 100:.1f}%",
                    )
                    _shadow_stat(
                        "단일 백테스트",
                        "통과" if sr.get("single_backtest_ok") else "미통과",
                        good=sr.get("single_backtest_ok", False),
                    )
                _shadow_gate_label(
                    sr.get("production_candidate", False),
                    extra=" (RWF 검증 필요)" if sr.get("rwf_required") else "",
                )
            else:
                ui.label("측정 데이터 누적 중 — 표본 부족").classes(
                    "text-xs text-gray-500"
                )
            ui.label(f"규칙: {sr.get('rule', '')}").classes(
                "text-[10px] text-gray-600 mt-1 leading-tight"
            )


def _shadow_stat(label: str, value: str, good: bool = False, bad: bool = False):
    """shadow 카드용 미니 스탯 (라벨 + 값)."""
    color = "text-emerald-400" if good else ("text-rose-400" if bad else "text-gray-200")
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-[10px] text-gray-500")
        ui.label(value).classes(f"text-sm font-bold {color}")


def _shadow_gate_label(production_candidate: bool, extra: str = ""):
    """production 게이트 상태 라벨 — 항상 보수적 표시."""
    if production_candidate:
        ui.label(
            f"⚠️ production_candidate=True{extra} — 그래도 검토 후 적용"
        ).classes("text-[11px] text-amber-400 mt-1")
    else:
        ui.label(
            f"🔒 production 미적용{extra} — 측정 단계입니다"
        ).classes("text-[11px] text-gray-500 mt-1")


def render_tab_perf():
    """[Step AK+AL+AM] 시스템 성과 추세 — 면책 + 6개 메트릭 + 모바일 + KOSPI 알파"""
    
    # ─── 헤더 ───
    with ui.row().classes("w-full items-center justify-between mb-3 flex-wrap gap-2"):
        with ui.column().classes("gap-0"):
            ui.label("📈 시스템 성과 추세").classes(
                "text-2xl font-bold text-white"
            )
            ui.label(
                "백테스트 기반 알고리즘 검증 결과 (paper trading)"
            ).classes("text-xs text-gray-400")

    # ─── 면책 카드 (가장 먼저!) ───
    _render_disclaimer_card()

    # ─── 데이터 로드 ───
    history = _load_history()
    
    # [Step AM] KOSPI 벤치마크 로드
    bench_data = _load_bench_cache()

    if history.empty:
        with ui.card().classes(
            "w-full p-8 bg-[#1a1a2e] border border-gray-700 rounded-lg "
            "items-center"
        ):
            ui.label("📭").classes("text-4xl mb-2")
            ui.label("축적된 성과 데이터가 부족합니다.").classes(
                "text-gray-400 text-base font-bold"
            )
            ui.label(
                "데이터가 매일 자동 누적되며, 7일 이상 누적 후 표시됩니다."
            ).classes("text-xs text-gray-500 mt-1")
            # 디버그 정보
            ui.label(f"검색 경로: {DATA_DIR}").classes(
                "text-xs text-gray-600 mt-2"
            )
            import glob as _g
            _found = len(_g.glob(
                os.path.join(DATA_DIR, "rank_validation_summary_*.csv")
            ))
            ui.label(f"파일 수: {_found}").classes(
                "text-xs text-gray-600"
            )
        return

    col_win, col_ret = 'WIN_RATE_%', 'AVG_RET_%'
    if col_win not in history.columns or col_ret not in history.columns:
        ui.label("필요 컬럼 없음 — 데이터 형식 확인 필요").classes(
            "text-amber-400 p-4"
        )
        return

    # ─── 데이터 표본 안내 ───
    n_files = len(history.groupby('Date')) if 'Date' in history.columns else 0
    with ui.row().classes("w-full items-center gap-2 mb-3 flex-wrap"):
        ui.badge(f"📊 {n_files}일 누적").props("color=cyan").classes("text-xs")
        ui.badge(f"🔬 {len(history):,}개 검증 결과").props("color=indigo").classes("text-xs")
        if bench_data and "KOSPI" in bench_data:
            ui.badge("📈 KOSPI 벤치마크 사용").props("color=green").classes("text-xs")

    # ─── 필터 (사용자 친화 라벨) ───
    methods = sorted(history['METHOD'].unique()) if 'METHOD' in history.columns else []
    def_m = next(
        (m for m in ['ELITE_SCORE', 'FINAL_SCORE', 'DISPLAY_SCORE', 'RANK_SCORE'] if m in methods),
        methods[0] if methods else None,
    )
    method_options = {m: METHOD_LABELS.get(m, m) for m in methods}

    topks = sorted(history['TOPK'].unique().tolist()) if 'TOPK' in history.columns else []
    def_k = 5 if 5 in topks else (topks[0] if topks else None)
    topk_options = {int(k): TOPK_LABELS.get(int(k), f"상위 {k}개") for k in topks}

    holds = sorted(history['H(영업일)'].unique().tolist()) if 'H(영업일)' in history.columns else []
    def_h = 5 if 5 in holds else (holds[0] if holds else None)
    hold_options = {int(h): HOLD_LABELS.get(int(h), f"{h}영업일") for h in holds}

    # 모바일 친화 — flex-wrap + 충분한 너비
    with ui.row().classes("w-full gap-3 flex-wrap mb-3"):
        sel_m = ui.select(
            options=method_options,
            value=def_m,
            label="🏆 평가 방법",
        ).classes("flex-1 min-w-[200px]").props(
            "outlined dense"
        ) if methods else None

        sel_k = ui.select(
            options=topk_options,
            value=def_k,
            label="🎯 추천 종목 수",
        ).classes("flex-1 min-w-[160px]").props(
            "outlined dense"
        ) if topks else None

        sel_h = ui.select(
            options=hold_options,
            value=def_h,
            label="📅 보유 기간",
        ).classes("flex-1 min-w-[160px]").props(
            "outlined dense"
        ) if holds else None

    # ─── 평가 방법 설명 (선택된 method 기준) ───
    method_desc_label = ui.label("").classes(
        "text-xs text-gray-400 italic mb-3 pl-1"
    )
    
    # ─── [Step AM] 차트 보기 모드 + 거래비용 가정 ───
    with ui.row().classes("w-full gap-3 flex-wrap mb-3"):
        # 차트 보기 모드
        chart_mode_options = {
            "performance": "📊 성과 보기 (승률 + 수익률)",
            "risk": "⚠️ 위험 보기 (평균 + 최악 낙폭)",
            "hit": "🎯 도달률 보기 (2% / 5%)",
        }
        if bench_data and "KOSPI" in bench_data:
            chart_mode_options["market"] = "📈 시장 비교 (전략 vs KOSPI)"
        
        sel_chart_mode = ui.select(
            options=chart_mode_options,
            value="performance",
            label="📈 차트 보기",
        ).classes("flex-1 min-w-[200px]").props("outlined dense")
        
        # 거래비용 가정
        sel_cost = ui.select(
            options=COST_OPTIONS,
            value=DEFAULT_COST_PCT,
            label="💵 거래비용 가정",
        ).classes("flex-1 min-w-[180px]").props("outlined dense")
    
    chart_area = ui.column().classes("w-full")

    def _build_chart():
        chart_area.clear()
        cdf = history.copy()
        if sel_m and sel_m.value:
            cdf = cdf[cdf['METHOD'] == sel_m.value]
            # 설명 업데이트
            desc = METHOD_DESCRIPTIONS.get(sel_m.value, "")
            if desc:
                method_desc_label.set_text(f"💡 {desc}")
        if sel_k and sel_k.value is not None:
            cdf = cdf[cdf['TOPK'] == int(sel_k.value)]
        if sel_h and sel_h.value is not None:
            cdf = cdf[cdf['H(영업일)'] == int(sel_h.value)]
        cdf = cdf.sort_values('Date').tail(30)

        # Timestamp → 문자열 (NiceGUI orjson 직렬화 호환)
        if 'Date' in cdf.columns:
            cdf['Date'] = cdf['Date'].apply(
                lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else str(x)
            )
        
        # [Step AM] 현재 선택된 차트 모드 + 비용률
        chart_mode = sel_chart_mode.value if sel_chart_mode else "performance"
        # [Step AN] _safe_float로 sel_cost.value 안전 변환
        cost_pct = _safe_float(sel_cost.value, default=DEFAULT_COST_PCT) if sel_cost else DEFAULT_COST_PCT
        current_hold = int(sel_h.value) if sel_h and sel_h.value else None

        with chart_area:
            if cdf.empty:
                with ui.card().classes(
                    "w-full p-6 bg-[#1a1a2e] border border-gray-700 rounded-lg"
                ):
                    ui.label("📭 조건에 맞는 데이터가 없습니다.").classes(
                        "text-gray-400 text-sm text-center"
                    )
                    ui.label("필터를 다른 조건으로 변경해보세요.").classes(
                        "text-xs text-gray-500 text-center mt-1"
                    )
                return

            # ─── [Step AM] 차트 보기 모드별 trace 분기 ───
            if PLOTLY_OK:
                fig = _build_chart_by_mode(
                    cdf=cdf,
                    mode=chart_mode,
                    bench_data=bench_data,
                    hold_days=current_hold,
                    col_win=col_win,
                    col_ret=col_ret,
                )
                if fig:
                    ui.plotly(fig).classes("w-full")
            else:
                ui.label("⚠️ Plotly 미설치 — 차트 표시 불가").classes(
                    "text-amber-400 p-4"
                )
            
            # ─── [Step AN] 차트 모드별 해설 (차트 직후) ───
            mode_explanation = CHART_MODE_EXPLANATIONS.get(chart_mode, "")
            if mode_explanation:
                with ui.card().classes(
                    "w-full p-2 bg-[#1a1a2e]/50 border border-cyan-700/20 "
                    "rounded-lg mt-2"
                ):
                    ui.label(mode_explanation).classes(
                        "text-xs text-cyan-100 leading-relaxed"
                    )

            # ─── 메트릭 (cost_pct + bench 전달) ───
            _render_metrics_grid(
                cdf,
                cost_pct=cost_pct,
                bench_data=bench_data,
                hold_days=current_hold,
            )
            
            # ─── 추가 안내 ───
            with ui.card().classes(
                "w-full p-3 bg-[#0a0a14] border border-gray-700/30 "
                "rounded-lg mt-3"
            ):
                ui.label(
                    "💡 위 지표는 모두 백테스트 시뮬레이션 결과입니다. "
                    "실제 거래 시 슬리피지/수수료/세금이 추가로 차감됩니다 "
                    "(통상 0.3~0.5% 수준)."
                ).classes("text-xs text-gray-400 leading-relaxed")
                
                # [Step AN] 시장 비교 모드일 때 KOSPI 근사치 안내
                if chart_mode == "market":
                    ui.label(
                        "ℹ️ 현재 KOSPI 비교는 보유기간별 평균 수익률을 수평선으로 "
                        "표시한 근사치입니다. 행별 정확한 알파는 향후 백테스트 "
                        "데이터에 KOSPI_RET_% / ALPHA_% 컬럼이 추가되면 제공됩니다."
                    ).classes("text-xs text-gray-500 italic mt-1 leading-relaxed")

    # [Step AM] 차트 모드 + 비용 select도 변경 시 차트 재빌드
    for w in [sel_m, sel_k, sel_h, sel_chart_mode, sel_cost]:
        if w:
            w.on("update:model-value", lambda _: _build_chart())
    _build_chart()

    # [v3.9.1] Shadow 실험실 — ENTRY_MODE / STRUCT risk shadow 측정 결과
    try:
        ui.separator().classes("my-6")
        _render_shadow_lab_card()
    except Exception as e:
        _logger.warning(f"shadow lab 카드 렌더 실패 (페이지 영향 없음): {e}")

    # ─── Research Workbench 통합 정리 ───
    try:
        from research_tab import render_research_tab
        ui.separator().classes("my-6")
        
        with ui.row().classes("w-full items-center gap-2 mb-2"):
            ui.label("🔬").classes("text-2xl")
            with ui.column().classes("gap-0 flex-1"):
                ui.label("심화 분석 (Research Workbench)").classes(
                    "text-lg font-bold text-cyan-300"
                )
                ui.label(
                    "위 차트는 핵심 지표 요약입니다. "
                    "더 깊이 분석하려면 아래 도구를 사용하세요."
                ).classes("text-xs text-gray-400")
        
        render_research_tab(data_dir=DATA_DIR)
    except ImportError:
        # research_tab 없어도 정상 작동
        pass
    except Exception as _rt_err:
        with ui.card().classes(
            "w-full p-3 bg-amber-900/20 border border-amber-500/30 rounded-lg mt-3"
        ):
            ui.label(
                f"⚠️ Research 탭 로드 중 오류가 발생했습니다."
            ).classes("text-sm text-amber-300")
            ui.label(f"({str(_rt_err)[:100]})").classes(
                "text-xs text-gray-500"
            )
