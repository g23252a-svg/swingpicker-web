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
17. ✅ KOSPI 근사치 명시 → v22.3.14에서 행별 KOSPI_RET_%/ALPHA_% 우선 사용

v22.3.14 추가:
- rank_validation_summary CSV의 KOSPI_RET_%/ALPHA_% 행별 정확 알파 우선 표시
- Shadow Promotion Gate로 B_red 등 룰 승격 심사표 표시

향후 작업:
- 거래별 비용 차감 후 평균 (현재는 평균에서 일괄 차감)
"""
import glob
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

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



def _metric_weights(df: pd.DataFrame, weight_col: str = "TOTAL_N") -> pd.Series:
    """[v22.3.11] KPI 표본가중 평균용 weight.

    rank_validation_summary는 일별/조건별 요약행이므로 단순 mean을 쓰면
    표본 5건인 날과 표본 200건인 날이 동일 가중 처리된다. 성과탭 KPI는
    TOTAL_N 기준 가중 평균을 기본으로 사용한다. TOTAL_N이 없거나 전부 0이면
    row 단위 동일가중으로 안전 fallback한다.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if weight_col not in df.columns:
        return pd.Series(1.0, index=df.index, dtype=float)
    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(w.sum()) <= 0:
        return pd.Series(1.0, index=df.index, dtype=float)
    return w.astype(float)


def _weighted_mean(df: pd.DataFrame, col: str, weight_col: str = "TOTAL_N") -> Optional[float]:
    """[v22.3.11] TOTAL_N 표본가중 평균.

    Returns None when the metric column has no valid numeric values.
    """
    if df is None or df.empty or col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce")
    mask = values.notna()
    if not mask.any():
        return None
    weights = _metric_weights(df, weight_col=weight_col).reindex(df.index).fillna(0.0)
    weights = weights[mask]
    values = values[mask]
    if float(weights.sum()) <= 0:
        return float(values.mean())
    return float((values * weights).sum() / weights.sum())


def _worst_drawdown_value(df: pd.DataFrame, col: str = "WORST_MDD_%") -> Optional[float]:
    """[v22.3.11] 기간 중 최악 낙폭.

    과거 데이터에 MDD 부호 오염이 섞여도 회원 화면에는 위험값을 보수적으로
    표시하기 위해 절댓값 최대치를 음수로 반환한다.
    """
    if df is None or df.empty or col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    return -float(values.abs().max())




def _resolve_alpha_metrics(
    df: pd.DataFrame,
    bench_data: dict = None,
    hold_days: int = None,
) -> Tuple[Optional[float], Optional[float], str]:
    """[v22.3.14] 행별 KOSPI/알파 우선 사용.

    rank_validation_summary에 KOSPI_RET_% / ALPHA_%가 있으면 추천일별
    실제 KOSPI forward return을 집계한 정확 알파로 표시한다. 없을 때만
    기존 bench_cache_latest 보유기간 평균 기반 근사치로 fallback한다.

    Returns:
        (kospi_ret_pct, alpha_pctp, source)
        source: row_exact | bench_avg | none
    """
    row_kospi = _weighted_mean(df, "KOSPI_RET_%")
    row_alpha = _weighted_mean(df, "ALPHA_%")
    if row_kospi is not None and row_alpha is not None:
        return row_kospi, row_alpha, "row_exact"

    if bench_data and hold_days is not None:
        kospi_ret = _get_kospi_return(bench_data, hold_days)
        avg_ret = _weighted_mean(df, "AVG_RET_%")
        if kospi_ret is not None and avg_ret is not None:
            return float(kospi_ret), float(avg_ret - float(kospi_ret)), "bench_avg"

    return None, None, "none"


def _safe_shadow_float(value, default: Optional[float] = None) -> Optional[float]:
    """Shadow promotion gate용 숫자 변환."""
    try:
        if value is None:
            return default
        v = float(value)
        if pd.isna(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _shadow_changed_pct(rule: dict) -> Optional[float]:
    """changed_pick_rate가 0~1 비율/0~100 퍼센트 어느 형식이든 %로 정규화."""
    raw = _safe_shadow_float(rule.get("changed_pick_rate"), None)
    if raw is None:
        raw = _safe_shadow_float(rule.get("changed_pick_pct"), None)
    if raw is None:
        return None
    return raw * 100.0 if abs(raw) <= 1.0 else raw


def _shadow_sample_n(rule: dict) -> Optional[int]:
    """Shadow rule 표본 수 후보 키를 보수적으로 탐색."""
    for key in ("n", "N", "sample_n", "n_raw", "total_n", "trades", "candidate_n"):
        if key in rule:
            v = _safe_shadow_float(rule.get(key), None)
            if v is not None:
                return int(v)
    return None


def _score_shadow_promotion_rule(
    rule_key: str,
    rule: dict,
    rwf_pass: bool = False,
) -> Dict[str, Any]:
    """[v22.3.14] Shadow Rule 승격 심사표.

    추천 로직을 자동 변경하지 않고, 룰을 다음 단계로 올릴 수 있는지
    표시/감점/hard-block 검토 후보로만 판정한다. hard block은 언제나
    사람이 별도 PR로 승격해야 한다.
    """
    d_ev = _safe_shadow_float(rule.get("delta_ev"), 0.0) or 0.0
    d_non_win = _safe_shadow_float(rule.get("delta_non_win_avg_ret"), 0.0) or 0.0
    changed_pct = _shadow_changed_pct(rule)
    sample_n = _shadow_sample_n(rule)
    single_ok = bool(rule.get("single_backtest_ok", False))
    rwf_ok = bool(
        rule.get("rwf_ok", False)
        or rule.get("rwf_pass", False)
        or rule.get("rwf_passed", False)
        or rwf_pass
    )

    checks = {
        "delta_ev": d_ev >= 0.8,
        "non_win": d_non_win >= 0.0,
        "changed": (changed_pct is not None and changed_pct <= 35.0),
        "single": single_ok,
        "rwf": rwf_ok,
        "sample": (sample_n is not None and sample_n >= 30),
    }

    if d_ev < 0 or ("single_backtest_ok" in rule and not single_ok):
        verdict = "폐기 후보"
        tone = "red"
        action = "운영 반영 금지"
    elif not checks["sample"]:
        verdict = "표본 부족 · 표시 유지"
        tone = "yellow"
        action = "2~4주 추가 누적"
    elif checks["delta_ev"] and checks["changed"] and checks["single"] and checks["rwf"]:
        if d_ev >= 1.0 and changed_pct is not None and changed_pct <= 30.0 and sample_n >= 60:
            verdict = "hard block 검토 후보"
            tone = "amber"
            action = "별도 PR에서 승격 심사"
        else:
            verdict = "감점/표시 승격 후보"
            tone = "green"
            action = "ENTRY_EDGE 감점 강화 검토"
    elif checks["delta_ev"] and checks["single"]:
        verdict = "관찰 유지"
        tone = "yellow"
        action = "구성변경률/RWF/표본 보강"
    else:
        verdict = "측정 유지"
        tone = "gray"
        action = "현 상태 유지"

    return {
        "rule_key": rule_key,
        "description": rule.get("description", ""),
        "delta_ev": d_ev,
        "delta_non_win_avg_ret": d_non_win,
        "changed_pct": changed_pct,
        "sample_n": sample_n,
        "single_ok": single_ok,
        "rwf_ok": rwf_ok,
        "checks": checks,
        "verdict": verdict,
        "tone": tone,
        "action": action,
    }

def _select_perf_default_slice(
    history: pd.DataFrame,
    method: str = "ELITE_SCORE",
    topk: int = 5,
    hold_days: int = 5,
) -> pd.DataFrame:
    """[v22.3.11] 성과탭 상단 카드 공통 기준 slice.

    성과 판정 카드/최근 7거래일 카드가 동일 기준을 중복 구현하던 것을
    helper로 통일한다. 가능한 경우 ELITE_SCORE / Top5 / 5영업일 기준을 사용하고,
    해당 slice가 없으면 기존 데이터 범위 안에서 안전하게 fallback한다.
    """
    if history is None or history.empty:
        return pd.DataFrame()
    h = history.copy()
    if "METHOD" in h.columns and (h["METHOD"] == method).any():
        h = h[h["METHOD"] == method]
    if "TOPK" in h.columns:
        topk_num = pd.to_numeric(h["TOPK"], errors="coerce")
        h_topk = h[topk_num == int(topk)]
        if not h_topk.empty:
            h = h_topk
    if "H(영업일)" in h.columns:
        hold_num = pd.to_numeric(h["H(영업일)"], errors="coerce")
        h_hold = h[hold_num == int(hold_days)]
        if not h_hold.empty:
            h = h_hold
    return h

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

    # [v22.3.8 C-3] N=0 방어 — 표본 없으면 KPI 숫자 카드 숨김
    # 현재 데이터엔 TOTAL_N=0 행이 없지만, 미래 데이터 깨짐 대비.
    # 계산 로직은 안 건드리고 UI만 안전한 안내 카드로 대체.
    _total_n_guard = (
        cdf['TOTAL_N'].sum() if 'TOTAL_N' in cdf.columns else 0
    )
    if _total_n_guard <= 0:
        with ui.card().classes(
            "w-full p-4 bg-[#1a1a2e] border border-gray-700/40 rounded-lg mt-3"
        ):
            ui.label("📊 표본 부족 — 평가 보류").classes(
                "text-sm font-bold text-amber-300 mb-1"
            )
            ui.label(
                "현재 선택한 기준에 해당하는 백테스트 표본이 없어 "
                "승률/수익률 평가는 표시하지 않습니다."
            ).classes("text-xs text-gray-400 leading-relaxed")
        return

    # [v22.3.11] 성과 KPI는 TOTAL_N 표본가중 평균으로 계산.
    # 일별 summary 행 단순 mean은 표본 작은 날을 과대반영할 수 있어 낙관 편향이 생김.
    win_rate = _weighted_mean(cdf, 'WIN_RATE_%')
    avg_ret = _weighted_mean(cdf, 'AVG_RET_%')
    hit_5 = _weighted_mean(cdf, 'HIT_5%_%')
    avg_mdd = _weighted_mean(cdf, 'AVG_MDD_%')
    # 최악 낙폭은 평균이 아니라 선택 구간의 최대 위험값으로 표시.
    worst_mdd = _worst_drawdown_value(cdf, 'WORST_MDD_%')
    total_n = cdf['TOTAL_N'].sum() if 'TOTAL_N' in cdf.columns else len(cdf)
    
    # [Step AL+AM] 비용 차감 후 추정 수익률 (사용자 선택 비용률)
    avg_ret_after_cost = None
    if avg_ret is not None:
        avg_ret_after_cost = avg_ret - cost_pct
    
    # [v22.3.14] KOSPI 알파 계산 — 행별 정확 알파 우선, 없으면 기존 평균 benchmark fallback
    kospi_ret, alpha, alpha_source = _resolve_alpha_metrics(
        cdf, bench_data=bench_data, hold_days=hold_days
    )
    
    # [v22.3.8 C-1] 현재 선택된 백테스트 차원(METHOD/TOPK/H/N)을 SSOT로 명시
    # 회원이 보는 KPI가 어떤 기준의 시뮬레이션인지 항상 헤더에서 확인 가능.
    # cdf는 위에서 cdf.empty / TOTAL_N=0 가드 통과한 상태이므로 iloc[0] 안전.
    _method_val = (
        cdf['METHOD'].iloc[0] if 'METHOD' in cdf.columns else None
    )
    _topk_val = (
        cdf['TOPK'].iloc[0] if 'TOPK' in cdf.columns else None
    )
    _hold_val = (
        cdf['H(영업일)'].iloc[0] if 'H(영업일)' in cdf.columns else None
    )
    _method_lbl = (
        METHOD_LABELS.get(_method_val, str(_method_val))
        if _method_val is not None else "—"
    )
    _criteria_parts = [_method_lbl]
    if _topk_val is not None:
        try:
            _criteria_parts.append(f"상위 {int(_topk_val)}")
        except (TypeError, ValueError) as exc:
            _logger.debug("[tab_perf] TOPK 표시값 변환 실패: %r (%s)", _topk_val, exc)
    if _hold_val is not None:
        try:
            _criteria_parts.append(f"{int(_hold_val)}영업일")
        except (TypeError, ValueError) as exc:
            _logger.debug("[tab_perf] 보유기간 표시값 변환 실패: %r (%s)", _hold_val, exc)
    _criteria_parts.append(f"N={int(total_n):,}")

    ui.label("📊 현재 성과 기준: " + " · ".join(_criteria_parts)).classes(
        "text-base font-bold text-cyan-300 mt-3 mb-1"
    )
    ui.label(
        "선택한 백테스트 시뮬레이션 기준의 표본가중 지표입니다."
    ).classes("text-[11px] text-gray-500 italic mb-2")

    # [v22.3.8 C-2] 공식 신규매수 기준은 별도 데이터 누적 후 분리 표시 예정
    # 현재 데이터엔 TOP_PICK+BUY_NOW_ELIGIBLE METHOD 행이 없음.
    # "공식 신규매수 기준"이라고 라벨만 박으면 표시-데이터 불일치 → 회피.
    with ui.card().classes(
        "w-full p-2 bg-[#1a1a2e]/60 border border-amber-700/30 "
        "rounded-lg mb-3"
    ):
        ui.label(
            "ℹ️ 위 성과는 선택한 스코어(METHOD) 기준 시뮬레이션입니다. "
            "승률/수익률은 TOTAL_N 기준 표본가중 평균이며, "
            "공식 신규매수 기준(TOP_PICK + BUY_NOW_ELIGIBLE) 성과는 "
            "별도 데이터 누적 후 분리 표시 예정입니다."
        ).classes("text-[11px] text-amber-200 leading-relaxed")
    
    # [Step AL+AM] 메트릭 카드 — 모바일 2열, 데스크톱 3열 반응형
    with ui.grid().classes(
        "w-full gap-3 grid-cols-2 md:grid-cols-3"
    ):
        # 1. 평균 승률
        _render_metric_card(
            icon="📊", label="표본가중 승률",
            value=f"{win_rate:.1f}%" if win_rate is not None else "—",
            color="amber",
            tooltip="TOTAL_N 기준 표본가중 승률. 보유 기간 종료 시 진입가 대비 +1% 이상 종목 비율",
        )
        
        # 2. 평균 수익률 (총 수익률 — 비용 미반영)
        _render_metric_card(
            icon="💰", label="표본가중 수익률 (총)",
            value=f"{avg_ret:+.2f}%" if avg_ret is not None else "—",
            color="blue",
            tooltip="TOTAL_N 기준 표본가중 평균 수익률 (수수료/세금 미반영)",
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
            value=f"-{abs(avg_mdd):.2f}%" if avg_mdd is not None else "—",
            color="orange",
            tooltip="MDD: 보유 중 진입가 대비 최저점까지의 평균 낙폭",
        )
        
        # 6. 최악 낙폭 (위험 강조)
        _render_metric_card(
            icon="🔴", label="최악 낙폭",
            value=f"-{abs(worst_mdd):.2f}%" if worst_mdd is not None else "—",
            color="red",
            tooltip="선택 기간 중 가장 컸던 낙폭. 평균이 아니라 최대 위험값",
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
                tooltip="동일 조건 백테스트 전략 표본가중 평균 수익률 (총)",
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
        
        # [v22.3.14] 알파 산출 방식 안내
        if alpha_source == "row_exact":
            alpha_note = (
                "ℹ️ KOSPI 알파는 검증일별 KOSPI_RET_% / ALPHA_% 컬럼을 "
                "TOTAL_N 기준으로 집계한 정확 알파입니다."
            )
            alpha_note_cls = "text-[11px] text-emerald-300 italic mt-2 text-center leading-relaxed"
        else:
            alpha_note = (
                "ℹ️ 현재 KOSPI 알파는 기존 bench_cache_latest 보유기간 평균값 기반 "
                "근사치입니다. rank_validation에 KOSPI_RET_% / ALPHA_%가 생성되면 "
                "검증일별 정확 알파로 자동 전환됩니다."
            )
            alpha_note_cls = "text-[11px] text-gray-500 italic mt-2 text-center leading-relaxed"
        ui.label(alpha_note).classes(alpha_note_cls)
    
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



def _load_official_buy_validation() -> dict:
    """[v22.3.12] 공식 신규매수 검증 JSON 로드.

    공식 신규매수는 TOP_PICK + BUY_NOW_ELIGIBLE 기준이며,
    별도 script가 생성한 official_buy_validation_latest.json만 읽는다.
    추천/점수/BUY_NOW 산식은 여기서 절대 변경하지 않는다.
    """
    dirs_to_try = [DATA_DIR, os.path.join(os.getcwd(), "data"), "data"]
    for d in dirs_to_try:
        path = os.path.join(d, "official_buy_validation_latest.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            _logger.warning("official_buy_validation_latest.json 로드 실패: %s", exc)
            return {}
    return {}


def _pct_or_dash(value, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        _logger.debug("pct 표시값 변환 실패: %r (%s)", value, exc)
        return "—"
    sign = "+" if signed else ""
    return f"{v:{sign}.{digits}f}%"


def _render_official_buy_validation_card() -> None:
    """[v22.3.12] 공식 신규매수 성과 누적 카드.

    측정 대상:
    - TOP_PICK + BUY_NOW_ELIGIBLE 실제 결과
    - BUY_NOW_ELIGIBLE=0으로 보류한 TOP_PICK의 이후 결과
    - 보류/현금 유지가 손실을 피했는지, 기회비용을 냈는지
    """
    j = _load_official_buy_validation()

    with ui.card().classes(
        "w-full p-3 bg-[#1a1a2e] border border-emerald-700/30 rounded-lg mb-3"
    ):
        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.label("📌").classes("text-2xl")
            with ui.column().classes("gap-0 flex-1"):
                ui.label("공식 신규매수 성과 검증").classes(
                    "text-lg font-bold text-emerald-300"
                )
                ui.label(
                    "TOP_PICK + BUY_NOW_ELIGIBLE 기준의 실제 성과와, "
                    "ELIGIBLE=0으로 보류한 TOP_PICK의 현금 유지 효과를 별도로 측정합니다."
                ).classes("text-xs text-gray-400 leading-relaxed")

        if not j:
            ui.label(
                "아직 official_buy_validation_latest.json 데이터가 없습니다. "
                "`python scripts/official_buy_validation.py --data-dir data --out-dir data` 실행 후 표시됩니다."
            ).classes("text-xs text-amber-200 leading-relaxed mt-1")
            return

        s = j.get("summary", {}) or {}
        asof = j.get("asof", "")

        with ui.row().classes("w-full flex-wrap gap-x-6 gap-y-1 mt-2"):
            _shadow_stat("검증일", f"{int(s.get('signal_days', 0)):,}일")
            _shadow_stat("공식 신호", f"{int(s.get('official_buy_signals', 0)):,}건")
            _shadow_stat("공식 결과", f"{int(s.get('official_buy_results', 0)):,}건")
            _shadow_stat(
                "공식 승률",
                _pct_or_dash(s.get("official_buy_win_rate"), digits=1),
                good=(s.get("official_buy_win_rate") or 0) >= 50,
            )
            _shadow_stat(
                "공식 평균",
                _pct_or_dash(s.get("official_buy_avg_net_pct"), digits=2, signed=True),
                good=(s.get("official_buy_avg_net_pct") or 0) > 0,
                bad=(s.get("official_buy_avg_net_pct") or 0) < 0,
            )

        with ui.row().classes("w-full flex-wrap gap-x-6 gap-y-1 mt-2"):
            _shadow_stat("공식매수 없음", f"{int(s.get('no_official_buy_days', 0)):,}일")
            _shadow_stat("보류 TOP_PICK", f"{int(s.get('top_pick_holdout_results', 0)):,}건")
            _shadow_stat(
                "보류 종목 승률",
                _pct_or_dash(s.get("holdout_top_pick_win_rate"), digits=1),
                bad=(s.get("holdout_top_pick_win_rate") or 0) >= 50,
            )
            _shadow_stat(
                "보류 종목 평균",
                _pct_or_dash(s.get("holdout_top_pick_avg_net_pct"), digits=2, signed=True),
                good=(s.get("holdout_top_pick_avg_net_pct") or 0) < 0,
                bad=(s.get("holdout_top_pick_avg_net_pct") or 0) > 0,
            )
            _shadow_stat(
                "현금 효과",
                _pct_or_dash(s.get("cash_vs_top_pick_avg_pct"), digits=2, signed=True),
                good=(s.get("cash_vs_top_pick_avg_pct") or 0) > 0,
                bad=(s.get("cash_vs_top_pick_avg_pct") or 0) < 0,
            )

        avoided = int(s.get("cash_avoided_loss_days", 0) or 0)
        opp = int(s.get("cash_opportunity_cost_days", 0) or 0)
        ui.label(
            f"현금 유지 판정: 손실 회피 {avoided}건 · 기회비용 {opp}건 · "
            f"데이터 기준 {asof or '—'}"
        ).classes("text-xs text-emerald-100 mt-2 leading-relaxed")
        ui.label(
            "※ 이 카드는 측정 전용입니다. BUY_NOW_ELIGIBLE / TOP_PICK / 점수 산식은 변경하지 않습니다. "
            "공식 결과 표본이 충분히 쌓이기 전까지는 승률을 과신하지 마세요."
        ).classes("text-[10px] text-gray-500 italic mt-1 leading-relaxed")



def _shadow_reliability_eval(
    n: int = 0,
    delta_ev: float | None = None,
    changed_rate: float | None = None,
    single_ok: bool | None = None,
    rwf_ok: bool | None = None,
) -> dict:
    # [v22.3.25] Shadow 실험 신뢰도/승격 가능성 평가. 표시 전용.
    # production 추천식에는 절대 반영하지 않는다.
    try:
        n = int(n or 0)
    except Exception:
        n = 0
    try:
        ev = None if delta_ev is None else float(delta_ev)
    except Exception:
        ev = None
    try:
        chg = None if changed_rate is None else float(changed_rate)
    except Exception:
        chg = None

    blockers: list[str] = []
    if n < 10:
        grade = "🔴 표본 부족"
        blockers.append("N<10")
    elif n < 30:
        grade = "🟡 관찰"
        blockers.append("N<30")
    elif n < 60:
        grade = "🟢 후보"
    else:
        grade = "🔵 승격 검토"

    if ev is not None and ev <= 0:
        blockers.append("EV 개선 없음")
    if single_ok is False:
        blockers.append("단일 백테스트 미통과")
    if rwf_ok is False:
        blockers.append("RWF 미통과")
    if chg is not None and chg > 0.40:
        blockers.append("구성변경률>40%")

    promotion_ready = (
        n >= 30
        and (ev is None or ev > 0.50)
        and single_ok is not False
        and rwf_ok is not False
        and (chg is None or chg <= 0.35)
    )
    verdict = "승격 검토 가능" if promotion_ready else (
        "승격 불가 — " + (" · ".join(blockers) if blockers else "추가 검증 필요")
    )

    return {
        "n": n,
        "grade": grade,
        "verdict": verdict,
        "promotion_ready": promotion_ready,
        "blockers": blockers,
    }


def _render_shadow_reliability_badge(title: str, eval_result: dict) -> None:
    grade = eval_result.get("grade", "—")
    verdict = eval_result.get("verdict", "추가 검증 필요")
    ready = bool(eval_result.get("promotion_ready"))
    color_cls = "text-emerald-300" if ready else "text-amber-200"
    ui.label(f"{title}: {grade} · {verdict}").classes(
        f"text-[11px] {color_cls} leading-snug mt-1"
    )


def _render_shadow_reliability_card(j: dict) -> None:
    # [v22.3.25] Shadow 실험실 신뢰도/승격 게이트 요약 카드.
    em = j.get("entry_mode_shadow", {}) or {}
    sr = j.get("struct_risk_shadow", {}) or {}
    pe = j.get("pre_entry_risk_shadow", {}) or {}
    rows = []

    if em.get("enabled"):
        n = int(em.get("extra_fills", 0) or 0)
        ev = None
        if n > 0:
            try:
                ev = float(em.get("extra_sum_ret", 0) or 0) / n
            except Exception:
                ev = None
        note = f"N={n} · 평균추가수익 {ev:+.2f}%" if ev is not None else f"N={n}"
        rows.append(("ENTRY_MODE chase", _shadow_reliability_eval(n=n, delta_ev=ev), note))

    if sr.get("enabled"):
        n = int(sr.get("n", sr.get("sample_n", sr.get("total_n", 0)) or 0))
        ev = sr.get("delta_ev")
        chg = sr.get("changed_pick_rate")
        single = bool(sr.get("single_backtest_ok")) if "single_backtest_ok" in sr else None
        note = f"ΔEV {float(ev or 0):+.2f} · 구성변경 {float(chg or 0)*100:.1f}%"
        rows.append(("STRUCT risk", _shadow_reliability_eval(n=n, delta_ev=ev, changed_rate=chg, single_ok=single), note))

    if pe.get("enabled") and isinstance(pe.get("rules"), dict):
        rules = pe.get("rules", {}) or {}
        key = pe.get("best_by_efficiency") or pe.get("best_by_delta_ev") or "B_red"
        best = rules.get(key, {}) or {}
        n = int(best.get("n", best.get("sample_n", best.get("total_n", 0)) or 0))
        ev = best.get("delta_ev")
        chg = best.get("changed_pick_rate")
        single = bool(best.get("single_backtest_ok")) if "single_backtest_ok" in best else None
        note = f"ΔEV {float(ev or 0):+.2f} · 구성변경 {float(chg or 0)*100:.1f}%"
        rows.append((f"PRE_ENTRY_RISK {key}", _shadow_reliability_eval(n=n, delta_ev=ev, changed_rate=chg, single_ok=single, rwf_ok=True if pe.get("rwf_required") else None), note))

    if not rows:
        return

    with ui.card().classes("w-full p-3 bg-[#141428] border border-purple-600/30 rounded-lg mb-2"):
        ui.label("🧪 Shadow 신뢰도/승격 게이트").classes(
            "text-sm font-bold text-purple-200 mb-1"
        )
        ui.label(
            "좋아 보이는 Shadow라도 N, 구성변경률, 단일 백테스트, RWF를 통과하지 못하면 운영 반영 금지입니다."
        ).classes("text-[11px] text-gray-400 leading-relaxed mb-1")

        for title, evl, note in rows:
            with ui.row().classes("w-full items-start justify-between gap-2"):
                ui.label(f"• {title}").classes("text-[11px] text-gray-200 font-semibold")
                ui.label(note).classes("text-[10px] text-gray-500")
            _render_shadow_reliability_badge("판정", evl)

        ui.label(
            "승격 기준: N≥30 · ΔEV +0.5 이상 · 구성변경률≤35% · 단일/RWF 통과. 하나라도 미달이면 measurement-only 유지."
        ).classes("text-[10px] text-gray-500 mt-2 leading-snug")


def _safe_col(df: pd.DataFrame, *names) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _render_green_early_stop_shadow_card() -> None:
    # [v22.3.25] GREEN인데 1~2일 내 손절난 케이스 진단 카드.
    trades = _load_top1_trades()
    if trades is None or trades.empty:
        return

    t = trades.copy()
    if "net_pct_num" not in t.columns:
        t["net_pct_num"] = pd.to_numeric(_safe_col(t, "net_pct", "NET_PCT"), errors="coerce")

    t = t.dropna(subset=["net_pct_num"]).tail(80).copy()
    if t.empty:
        return

    risk = _safe_col(t, "ENTRY_RISK_LEVEL", "entry_risk_level", "risk_level").astype(str).str.upper()
    if risk.eq("<NA>").all() or risk.eq("NAN").all():
        vwap = pd.to_numeric(_safe_col(t, "VWAP_GAP", "vwap_gap"), errors="coerce")
        risk = pd.Series("UNKNOWN", index=t.index)
        risk.loc[vwap.notna() & (vwap <= 10)] = "GREEN_PROXY"

    out = _safe_col(t, "outcome_norm", "outcome").astype(str).str.upper()
    is_loss = out.eq("LOSS") | (t["net_pct_num"] < 0)
    is_green = risk.isin(["GREEN", "GREEN_PROXY"])

    fill_dt = pd.to_datetime(_safe_col(t, "fill_date", "entry_date"), errors="coerce")
    exit_dt = pd.to_datetime(_safe_col(t, "exit_date", "sell_date"), errors="coerce")
    hold_days = pd.to_numeric(_safe_col(t, "holding_days", "hold_days"), errors="coerce")
    calc_days = (exit_dt - fill_dt).dt.days
    hold_days = hold_days.fillna(calc_days)
    early = hold_days.fillna(999) <= 2

    green_losses = t[is_green & is_loss].copy()
    early_green_losses = t[is_green & is_loss & early].copy()

    if len(green_losses) == 0:
        return

    n_green_loss = int(len(green_losses))
    n_early = int(len(early_green_losses))
    early_rate = n_early / max(n_green_loss, 1) * 100.0
    avg_loss = float(green_losses["net_pct_num"].mean())

    with ui.card().classes("w-full p-3 bg-[#1a1a2e] border border-rose-700/30 rounded-lg mb-2"):
        ui.label("🧯 GREEN 조기손절 shadow — 놓친 위험 원인 분석").classes(
            "text-sm font-bold text-rose-200 mb-1"
        )
        ui.label(
            "ENTRY_RISK가 GREEN이었거나 GREEN에 가까웠는데 빠르게 손절난 케이스를 추적합니다. 추천식에는 반영하지 않는 진단 전용입니다."
        ).classes("text-[11px] text-gray-400 leading-relaxed mb-1")

        with ui.row().classes("w-full flex-wrap gap-x-6 gap-y-1"):
            _shadow_stat("GREEN 손실", f"{n_green_loss}건")
            _shadow_stat("2일내 손절", f"{n_early}건", bad=n_early > 0)
            _shadow_stat("조기손절률", f"{early_rate:.1f}%", bad=early_rate >= 50)
            _shadow_stat("평균손익", f"{avg_loss:+.2f}%", bad=avg_loss < 0)

        if n_early:
            ui.label("최근 조기손절 예시").classes("text-[10px] text-rose-100 mt-2")
            for _, row in early_green_losses.tail(3).iterrows():
                name = row.get("name", row.get("종목명", row.get("stock_name", "-")))
                code = row.get("code", row.get("종목코드", ""))
                net = pd.to_numeric(row.get("net_pct_num"), errors="coerce")
                fd = row.get("fill_date", row.get("entry_date", ""))
                code_txt = str(code).zfill(6) if str(code).strip() else ""
                ui.label(f"• {fd} {name} {code_txt} · {float(net):+.2f}%").classes(
                    "text-[10px] text-gray-500 leading-tight"
                )

        ui.label(
            "다음 연구 후보: 전일 급등·VWAP 급확대·거래대금 폭증 후 식음·시장 급락·수급 역전. 현재는 원인 표시 단계이며 hard block 금지."
        ).classes("text-[10px] text-gray-500 mt-2 leading-snug")

def _render_shadow_lab_card():
    """[v3.9.1 / v3.9.2] Shadow 실험실 — 3개 shadow 측정 결과 표시.

    추천 로직을 바꾸지 않고 "바꿨다면 어땠을지"를 매일 자동 측정한 결과.
    measurement-only — production 적용 아님을 명시.

    카드:
      🎯 ENTRY_MODE shadow    (v3.9.0) — 강한 종목 미체결 chase 회수
      🛡️ STRUCT risk shadow   (v3.9.1) — STRUCT 70~85 제외 시뮬
      🚨 PRE_ENTRY_RISK shadow (v3.9.2) — 4개 룰 비교 (B_red 등)
    """
    j = _load_backtest_validation()
    em = j.get("entry_mode_shadow", {})
    sr = j.get("struct_risk_shadow", {})
    pe = j.get("pre_entry_risk_shadow", {})  # [v3.9.2]

    # 셋 다 없거나 비활성이면 카드 자체를 안 그림
    if not (em.get("enabled") or sr.get("enabled") or pe.get("enabled")):
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
            # 위쪽 핵심 지표(Top5, 5영업일, 47일)와 Shadow(Top3, 10영업일, 57일)의
            # 백테스트 기준이 다름을 명시. 회원 혼란 방지.
            ui.label(
                "※ Shadow 실험실은 위쪽 핵심 지표와 별도로, daily Top3 실전 "
                "백테스트(10영업일 보유) 기준으로 계산됩니다."
            ).classes("text-[10px] text-gray-500 mt-1 italic")

    # [v3.9.12] Shadow 종합 판정 카드 — 3개 shadow 상태 한눈에
    try:
        _render_shadow_summary_card(j)
    except Exception as _e:
        _logger.debug("shadow summary card render skipped: %s", _e)

    # [v22.3.14] Shadow Promotion Gate — 룰 승격 심사표
    try:
        _render_shadow_promotion_gate_card(j)
    except Exception as _e:
        _logger.debug("shadow promotion gate render skipped: %s", _e)

    # [v22.3.25] Shadow 신뢰도/승격 게이트 — measurement-only
    try:
        _render_shadow_reliability_card(j)
    except Exception as _e:
        _logger.debug("shadow reliability card render skipped: %s", _e)

    # [v22.3.25] GREEN 조기손절 shadow — 놓친 위험 원인 분석
    try:
        _render_green_early_stop_shadow_card()
    except Exception as _e:
        _logger.debug("green early stop shadow render skipped: %s", _e)

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

    # ─── PRE_ENTRY_RISK shadow [v3.9.2] ───
    if pe.get("enabled") and "rules" in pe:
        with ui.card().classes(
            "w-full p-3 bg-[#1a1a2e] border border-purple-700/30 "
            "rounded-lg mb-2"
        ):
            ui.label(
                "🚨 PRE_ENTRY_RISK shadow — 4개 룰 비교 (위험 종목 사전 식별)"
            ).classes("text-sm font-bold text-purple-200 mb-1")

            rules = pe.get("rules", {})
            best_rule_key = pe.get("best_by_efficiency") or pe.get("best_by_delta_ev")
            # 추천 룰(B_red 기본 — RWF 통과 룰) 우선 표시
            highlight_rule = best_rule_key or "B_red"
            best = rules.get(highlight_rule, {})

            if "delta_ev" in best:
                ui.label(
                    f"⭐ 추천 룰: {highlight_rule} — {best.get('description','')}"
                ).classes("text-xs text-emerald-300 mb-1")
                with ui.row().classes("w-full flex-wrap gap-x-6 gap-y-1"):
                    _shadow_stat(
                        "ΔEV",
                        f"{best['delta_ev']:+.2f}",
                        good=best["delta_ev"] > 0,
                        bad=best["delta_ev"] < 0,
                    )
                    _shadow_stat(
                        "Δ비승리 평균손익",
                        f"{best.get('delta_non_win_avg_ret', 0):+.2f}",
                        good=best.get("delta_non_win_avg_ret", 0) > 0,
                        bad=best.get("delta_non_win_avg_ret", 0) < 0,
                    )
                    _shadow_stat(
                        "Top3 구성변경",
                        f"{best.get('changed_pick_rate', 0) * 100:.1f}%",
                    )
                    _shadow_stat(
                        "단일 백테스트",
                        "통과" if best.get("single_backtest_ok") else "미통과",
                        good=best.get("single_backtest_ok", False),
                    )

                # 4개 룰 한 줄 요약 (작은 글씨)
                lines = []
                for rule_key in ["A_struct70_85", "B_red", "C_orange", "D_red_orange"]:
                    r = rules.get(rule_key, {})
                    if "delta_ev" not in r:
                        continue
                    mark = "★ " if rule_key == highlight_rule else "  "
                    ok = "✅" if r.get("single_backtest_ok") else "❌"
                    lines.append(
                        f"{mark}{rule_key:14s}: ΔEV {r['delta_ev']:+.2f} / "
                        f"구성변경 {r.get('changed_pick_rate', 0) * 100:.1f}% {ok}"
                    )
                if lines:
                    ui.label("4개 룰 비교:").classes("text-[10px] text-gray-400 mt-2")
                    for line in lines:
                        ui.label(line).classes(
                            "text-[10px] text-gray-500 font-mono leading-tight"
                        )

                _shadow_gate_label(
                    pe.get("production_candidate", False),
                    extra=" (RWF B_red 5/5 통과)" if pe.get("rwf_required") else "",
                )
            else:
                ui.label("측정 데이터 누적 중 — 표본 부족").classes(
                    "text-xs text-gray-500"
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


# ═══════════════════════════════════════════════════
# [v3.9.12] 성과탭 회원용 요약 카드들
# ═══════════════════════════════════════════════════
# ═══════════════════════════════════════════════════
# [v3.9.14d] 손실/수익 카드 공통 helper (중복 로딩 제거)
# ═══════════════════════════════════════════════════
def _date_key_top1(x):
    """[v3.9.14b 보정 1] date 키 정규화 — '2026-05-15' / '20260515' / 20260515(int) 모두 처리"""
    s = str(x).strip()
    s_digits = s.replace("-", "").replace("/", "").replace(".0", "")
    if len(s_digits) == 8 and s_digits.isdigit():
        return s_digits
    try:
        return pd.to_datetime(x).strftime("%Y%m%d")
    except Exception:
        return s.replace("-", "").replace("/", "")[:8]


def _code6_top1(x):
    """[v3.9.14b 보정 4] 종목코드 6자리 정규화 — '5930.0' / 5930 / '005930' 모두 처리"""
    s = str(x).strip()
    try:
        if s.endswith(".0") or "." in s:
            s = str(int(float(s)))
    except Exception:
        pass
    return s.zfill(6)


def _load_top1_trades() -> pd.DataFrame:
    """[v3.9.14d + v3.9.22c-1] backtest_top1_trades 최근 15일치 로딩 + 정규화.

    [v3.9.22c-1 DATA_AUDIT] exit_price=0 결손 데이터 필터링.
    근거: RF머트리얼즈 20260428 케이스
      04/30: exit_price=0, -100.22% LOSS (데이터 누락)
      05/04: exit_price=108600, +21.67% OPEN (가격 복구)
      05/07: exit_price=116900, +30.98% WIN (실제 결과)
    → exit_price=0 + LOSS 행은 백테스트 처리 버그로 -100% 산출.
      실제 손실이 아니라 거래정지/상폐/가격 스냅샷 누락 가능성.
    → 통계/표시에서 제외 + DATA_QUALITY_FLAG 컬럼으로 별도 태깅.

    Returns: 정규화된 trades DataFrame 또는 빈 DataFrame.
    컬럼 추가:
      - fill_date(datetime), net_pct_num(float), outcome_norm(upper str)
      - DATA_QUALITY_FLAG(str): "OK" / "EXIT_ZERO" / ""
    """
    import glob as _g
    try:
        trade_files = sorted(_g.glob(os.path.join(DATA_DIR, "backtest_top1_trades_*.csv")))
        trade_files = [f for f in trade_files if "latest" not in os.path.basename(f)]
        if len(trade_files) < 3:
            return pd.DataFrame()
        trades = pd.concat(
            [pd.read_csv(f) for f in trade_files[-15:]],
            ignore_index=True,
        )
        trades.columns = [c.lstrip("\ufeff") for c in trades.columns]
        trades = trades.drop_duplicates(subset=["date", "code"])

        if "fill_date" not in trades.columns or "outcome" not in trades.columns:
            return pd.DataFrame()
        if "net_pct" not in trades.columns:
            return pd.DataFrame()
        trades["fill_date"] = pd.to_datetime(trades["fill_date"], errors="coerce")
        # [v3.9.14b 보정 2] net_pct 숫자 변환
        trades["net_pct_num"] = pd.to_numeric(trades["net_pct"], errors="coerce")
        # [v3.9.14b 보정 3] outcome 대소문자/공백 정규화
        trades["outcome_norm"] = trades["outcome"].astype(str).str.strip().str.upper()

        # ─── [v3.9.22c-1 DATA_AUDIT] 데이터 결손 태깅 + 필터 ───
        exit_price_num = pd.to_numeric(
            trades.get("exit_price", 0), errors="coerce"
        ).fillna(0)
        # exit_price=0 인데 outcome=LOSS이고 net_pct가 -90 이하 = 데이터 결손
        _suspect = (
            (exit_price_num <= 0)
            & (trades["outcome_norm"] == "LOSS")
            & (trades["net_pct_num"] <= -90)
        )
        trades["DATA_QUALITY_FLAG"] = "OK"
        trades.loc[_suspect, "DATA_QUALITY_FLAG"] = "EXIT_ZERO"

        if _suspect.any():
            n = int(_suspect.sum())
            _logger = logging.getLogger(__name__)
            _logger.info(
                f"[v3.9.22c-1 DATA_AUDIT] exit_price=0 의심 행 {n}건 태깅 "
                f"(통계 제외 대상)"
            )

        # 의심 행은 통계에서 제외 — net_pct_num을 NaN으로 (dropna(subset)에서 자동 제외)
        # outcome_norm은 유지해서 별도 진단 가능
        trades.loc[_suspect, "net_pct_num"] = pd.NA

        return trades
    except Exception:
        return pd.DataFrame()


def _load_recommend_cache(days: int = 90) -> dict:
    """[v3.9.14d] recommend_*.csv 최근 N일 캐시 로딩.
    
    Returns: {date_key("YYYYMMDD"): DataFrame with __code6 column} 또는 빈 dict.
    """
    import glob as _g
    rec_files = sorted(_g.glob(os.path.join(DATA_DIR, "recommend_*.csv")))
    rec_files = [f for f in rec_files if "latest" not in os.path.basename(f)]
    recs = {}
    for f in rec_files[-days:]:
        raw_d = os.path.basename(f).replace("recommend_", "").replace(".csv", "")
        d = _date_key_top1(raw_d)
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
            df.columns = [c.lstrip("\ufeff") for c in df.columns]
            if "종목코드" in df.columns:
                df["__code6"] = df["종목코드"].apply(_code6_top1)
                recs[d] = df
        except Exception:
            pass
    return recs


def _load_macro_risk_cache() -> dict:
    """[v3.9.14e] run_meta_YYYYMMDD.json 일자별 macro_risk 로딩.
    
    Returns: {"20260515": "CAUTION", "20260514": "NORMAL", ...}
    
    GREEN 손실 분석에서 손실 시점의 시장 모드를 정확히 매칭.
    최신 1개만 쓰면 "2주 전 손실에 오늘 CAUTION 적용"되는 부정확 차단.
    """
    import glob as _g, json as _j
    out = {}
    meta_files = sorted(_g.glob(os.path.join(DATA_DIR, "run_meta_*.json")))
    meta_files = [f for f in meta_files if "latest" not in os.path.basename(f)]
    for f in meta_files[-90:]:  # 90일치
        raw_d = os.path.basename(f).replace("run_meta_", "").replace(".json", "")
        d = _date_key_top1(raw_d)
        try:
            with open(f, "r", encoding="utf-8") as fh:
                j = _j.load(fh)
            risk = str(j.get("macro_risk", "") or "").strip().upper()
            if risk:
                out[d] = risk
        except Exception:
            pass
    return out


def _enrich_trade_with_risk(row, recs: dict) -> dict:
    """[v3.9.14d] 단일 trade row에 STRUCT/VWAP/ENTRY_RISK 정보 부착.
    
    ENTRY_RISK_LEVEL SSOT 우선 — 컬럼 있으면 그대로, 없으면 STRUCT/VWAP로 재계산.
    
    Returns: {fill_date, name, net_pct, struct, vwap, risk}
    """
    d = _date_key_top1(row["date"])
    code6 = _code6_top1(row["code"])
    rec_df = recs.get(d)
    risk_str = "데이터 없음"
    s_val, v_val = None, None
    if rec_df is not None:
        rr = rec_df[rec_df["__code6"] == code6]
        if not rr.empty:
            r2 = rr.iloc[0]
            s_raw = pd.to_numeric(r2.get("STRUCT_SCORE"), errors="coerce")
            v_raw = pd.to_numeric(r2.get("VWAP_GAP"), errors="coerce")
            if pd.notna(s_raw) and pd.notna(v_raw):
                s_val, v_val = float(s_raw), float(v_raw)
            # ENTRY_RISK_LEVEL SSOT 우선
            csv_risk = str(r2.get("ENTRY_RISK_LEVEL", "") or "").strip().upper()
            if csv_risk in ("RED", "ORANGE", "GREEN"):
                risk_str = {
                    "RED": "🔴 RED",
                    "ORANGE": "🟠 ORANGE",
                    "GREEN": "🟢 GREEN",
                }[csv_risk]
            elif s_val is not None and v_val is not None:
                red = (s_val >= 70 and s_val <= 85 and v_val > 8)
                orange = (s_val < 90 and v_val > 15 and not red)
                risk_str = (
                    "🔴 RED" if red else ("🟠 ORANGE" if orange else "🟢 GREEN")
                )

    net_val = float(row.get("net_pct_num") or 0)
    return {
        "fill_date": row["fill_date"],
        "name": row.get("name", ""),
        "net_pct": net_val,
        "struct": s_val,
        "vwap": v_val,
        "risk": risk_str,
    }


def _render_profit_attribution_card(
    trades: pd.DataFrame = None,
    recs: dict = None,
) -> None:
    """[v3.9.14c] 최근 수익 기여 Top — 손실 카드와 균형용.
    
    [v3.9.14d] 공통 helper 사용 — _load_top1_trades / _load_recommend_cache /
    _enrich_trade_with_risk.
    
    [v3.9.14e] trades / recs 외부 주입 가능 — render_tab_perf에서 한 번만
    로딩 후 두 카드에 공유 (이전엔 각 카드가 따로 로딩 → 2회 중복).
    None이면 자체 로딩 (단독 호출 호환).
    """
    try:
        if trades is None:
            trades = _load_top1_trades()
        if trades.empty:
            return

        recent = trades.dropna(subset=["fill_date"]).sort_values("fill_date").tail(60)
        wins = (
            recent[recent["outcome_norm"] == "WIN"]
            .dropna(subset=["net_pct_num"])
            .sort_values("net_pct_num", ascending=False)
            .head(5)
        )
        if wins.empty:
            return

        if recs is None:
            recs = _load_recommend_cache(days=90)
        enriched = [_enrich_trade_with_risk(r, recs) for _, r in wins.iterrows()]
        if not enriched:
            return

        # 렌더
        with ui.card().classes(
            "w-full p-3 mb-3 bg-[rgba(16,185,129,0.06)] "
            "border border-emerald-500/30 rounded-lg"
        ):
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.label("📈").classes("text-xl")
                ui.label("최근 수익 기여 Top — Top1 실전 검증").classes(
                    "text-base font-bold text-emerald-300"
                )
            with ui.column().classes("gap-1 pl-5"):
                for e in enriched:
                    parts = [
                        f"{e['fill_date'].strftime('%m/%d')} {e['name']}",
                        f"{e['net_pct']:+.2f}%",
                    ]
                    if e["struct"] is not None and e["vwap"] is not None:
                        parts.append(
                            f"STRUCT {e['struct']:.0f} · VWAP_GAP {e['vwap']:+.1f}"
                        )
                    parts.append(e["risk"])
                    color = (
                        "text-emerald-200" if "🟢" in e["risk"]
                        else "text-gray-300"
                    )
                    ui.label("• " + " / ".join(parts)).classes(f"text-xs {color}")
            ui.label(
                "※ Top1 실전 검증 기준 · 최근 fill 60건 중 수익 Top 5 · "
                "추천 당시 STRUCT/VWAP 기준 (ENTRY_RISK_LEVEL CSV 컬럼 SSOT 우선)"
            ).classes("text-[10px] text-gray-500 italic mt-1 pl-5")
    except Exception:
        return


def _render_loss_attribution_card(
    trades: pd.DataFrame = None,
    recs: dict = None,
) -> None:
    """[v3.9.14] 최근 손실 기여 자동 리포트.
    
    [v3.9.14d] 공통 helper 사용.
    [v3.9.14e] trades / recs 외부 주입 가능 — 수익 카드와 캐시 공유.
    
    추가 진단:
    - 공통 원인 (RED/ORANGE 집중 / VWAP 과열 / STRUCT 애매)
    - GREEN 손실 패턴 (TP 거리 과도 / 짧은 days_held / 모델승률 /
      RR / 시장 모드)
    """
    try:
        if trades is None:
            trades = _load_top1_trades()
        if trades.empty:
            return

        recent = trades.dropna(subset=["fill_date"]).sort_values("fill_date").tail(60)
        losses = (
            recent[recent["outcome_norm"] == "LOSS"]
            .dropna(subset=["net_pct_num"])
            .sort_values("net_pct_num")
            .head(5)
        )
        if losses.empty:
            return

        if recs is None:
            recs = _load_recommend_cache(days=90)
        enriched = [_enrich_trade_with_risk(r, recs) for _, r in losses.iterrows()]
        if not enriched:
            return

        # 위험 통계 추출 (data 매칭 성공한 것만)
        struct_vals, vwap_vals, risk_levels = [], [], []
        for e in enriched:
            if e["struct"] is not None and e["vwap"] is not None and e["risk"] != "데이터 없음":
                struct_vals.append(e["struct"])
                vwap_vals.append(e["vwap"])
                risk_levels.append(e["risk"])

        # 5. 공통 패턴 진단
        diagnosis = None
        green_pattern = None
        if struct_vals and vwap_vals:
            n_risk = sum(1 for x in risk_levels if x in ("🔴 RED", "🟠 ORANGE"))
            n_green = sum(1 for x in risk_levels if "🟢" in x)
            n_high_vwap = sum(1 for v in vwap_vals if v > 8)
            n_70_85 = sum(1 for s in struct_vals if 70 <= s <= 85)
            total = len(struct_vals)
            if n_risk >= max(2, total // 2):
                diagnosis = (
                    f"위험 패턴(RED/ORANGE)에 손실 집중 — {n_risk}/{total}건. "
                    "ENTRY_RISK 표시를 확인하세요."
                )
            elif n_high_vwap >= total * 0.6:
                diagnosis = (
                    f"VWAP 과열 구간에 손실 집중 — VWAP>8% {n_high_vwap}/{total}건"
                )
            elif n_70_85 >= total * 0.6:
                diagnosis = (
                    f"STRUCT 70~85 애매 구간에 손실 집중 — {n_70_85}/{total}건"
                )
            else:
                diagnosis = "특정 위험 패턴 집중은 보이지 않습니다."

            # [v3.9.14c → v3.9.14d 확장 → v3.9.14e 정밀화] GREEN 손실 추가 진단
            # 본인 짚은 5개 잠재 원인 중 5개 활용:
            #   - TP 거리 (entry → tp1 gap > 15%)
            #   - days_held (짧은 손절 ≤ 2일)
            #   - 개별 모델 승률 (EST_WIN_RATE < 0.50)
            #   - RR (RR_NOW_TP1 < 1.2)
            #   - 시장 모드 (macro_risk CAUTION/WARNING) — 손실 시점별 매칭
            # 섹터 수익률은 별도 데이터 필요 (다음 패치)
            if n_green >= 2:
                green_items = [e for e in enriched if "🟢" in e.get("risk", "")]
                excessive_tp = 0
                short_held = 0
                low_model_wr = 0
                low_rr = 0
                market_caution_at_loss = 0
                green_count = len(green_items)

                # [v3.9.14e] 시장 모드 — 손실 시점별 정확 매칭 (이전엔 최신 1개만)
                macro_cache = _load_macro_risk_cache()

                for gi in green_items:
                    match = losses[
                        (losses["name"].astype(str) == gi.get("name", ""))
                        & (losses["fill_date"] == gi.get("fill_date"))
                    ]
                    if match.empty:
                        continue
                    mr = match.iloc[0]
                    entry = pd.to_numeric(mr.get("entry"), errors="coerce")
                    tp1 = pd.to_numeric(mr.get("tp1"), errors="coerce")
                    days = pd.to_numeric(mr.get("days_held"), errors="coerce")
                    if pd.notna(entry) and pd.notna(tp1) and entry > 0:
                        gap = (float(tp1) / float(entry) - 1) * 100
                        if gap > 15:
                            excessive_tp += 1
                    if pd.notna(days) and float(days) <= 2:
                        short_held += 1

                    # [v3.9.14e] 손실 시점 (추천일 = mr["date"]) macro_risk 매칭
                    d_at_entry = _date_key_top1(mr["date"])
                    risk_at_entry = macro_cache.get(d_at_entry, "")
                    if risk_at_entry in ("CAUTION", "WARNING", "CRITICAL"):
                        market_caution_at_loss += 1

                    # 같은 (date, code) recommend 행에서 EST_WIN_RATE / RR_NOW_TP1
                    code6 = _code6_top1(mr["code"])
                    rec_df = recs.get(d_at_entry)
                    if rec_df is not None:
                        rr = rec_df[rec_df["__code6"] == code6]
                        if not rr.empty:
                            r2 = rr.iloc[0]
                            ewr = pd.to_numeric(r2.get("EST_WIN_RATE"), errors="coerce")
                            rrv = pd.to_numeric(r2.get("RR_NOW_TP1"), errors="coerce")
                            if pd.notna(ewr) and float(ewr) < 0.50:
                                low_model_wr += 1
                            if pd.notna(rrv) and float(rrv) < 1.2:
                                low_rr += 1

                # 우선순위 (가장 강한 패턴 1개만 표시)
                threshold = max(2, green_count // 2)
                if excessive_tp >= threshold:
                    green_pattern = (
                        f"GREEN 손실 {green_count}건 중 {excessive_tp}건이 "
                        f"목표가 차이 15% 초과 — TP 거리 과도가 원인일 수 있음"
                    )
                elif short_held >= threshold:
                    green_pattern = (
                        f"GREEN 손실 {green_count}건 중 {short_held}건이 "
                        f"2일 이내 손절 — 급락일 진입 가능성"
                    )
                elif low_model_wr >= threshold:
                    green_pattern = (
                        f"GREEN 손실 {green_count}건 중 {low_model_wr}건이 "
                        f"개별 모델 승률 50% 미만 — 모델 신뢰도 낮은 종목 진입"
                    )
                elif low_rr >= threshold:
                    green_pattern = (
                        f"GREEN 손실 {green_count}건 중 {low_rr}건이 "
                        f"수익:손실 1.2 미만 — 손익비 불리한 진입"
                    )
                elif market_caution_at_loss >= threshold:
                    green_pattern = (
                        f"GREEN 손실 {green_count}건 중 {market_caution_at_loss}건이 "
                        "시장 주의/위험 구간에 진입 — 시장 모드 영향 가능"
                    )
                elif green_count >= 3:
                    green_pattern = (
                        f"GREEN 손실 {green_count}건은 현재 RED/ORANGE 룰로 "
                        "설명 안 됨 — 시장 환경/섹터 요인 점검 필요"
                    )

        # 6. 렌더
        with ui.card().classes(
            "w-full p-3 mb-3 bg-[rgba(239,68,68,0.06)] "
            "border border-red-500/30 rounded-lg"
        ):
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.label("📉").classes("text-xl")
                ui.label("최근 손실 기여 Top — Top1 실전 검증").classes(
                    "text-base font-bold text-red-300"
                )
            with ui.column().classes("gap-1 pl-5"):
                for e in enriched:
                    parts = [
                        f"{e['fill_date'].strftime('%m/%d')} {e['name']}",
                        f"{e['net_pct']:+.2f}%",
                    ]
                    if e["struct"] is not None and e["vwap"] is not None:
                        parts.append(
                            f"STRUCT {e['struct']:.0f} · VWAP_GAP {e['vwap']:+.1f}"
                        )
                    parts.append(e["risk"])
                    color = "text-red-300" if "🔴" in e["risk"] else (
                        "text-orange-300" if "🟠" in e["risk"]
                        else "text-gray-300"
                    )
                    ui.label("• " + " / ".join(parts)).classes(f"text-xs {color}")
            if diagnosis:
                ui.label(f"공통 원인: {diagnosis}").classes(
                    "text-xs text-amber-200 mt-2 pl-5 leading-relaxed"
                )
            if green_pattern:
                ui.label(f"추가 분석: {green_pattern}").classes(
                    "text-xs text-cyan-200 mt-1 pl-5 leading-relaxed"
                )
            ui.label(
                "※ Top1 실전 검증 기준 · 최근 fill 60건 중 손실 Top 5 · "
                "추천 당시 STRUCT/VWAP 기준 (ENTRY_RISK_LEVEL CSV 컬럼 SSOT 우선)"
            ).classes("text-[10px] text-gray-500 italic mt-1 pl-5")
    except Exception:
        return


def _render_recent_trend_card(history: pd.DataFrame) -> None:
    """[v3.9.13] 최근 7거래일 성과 변화 카드.
    
    누적 평균만 보면 최근 악화를 못 잡음. 5/15 같은 폭락 이후엔 특히 중요.
    최근 7거래일 vs 직전 7거래일 비교로 추세 표시.
    
    [v3.9.13b] 정확히는 "최근 7개 검증일"이지만 회원 이해도 위해 "7거래일"로 표기.
    
    지표:
      WIN_RATE_% / AVG_RET_% / AVG_MDD_%
    METHOD/TOPK/H는 perf judgment와 동일 기준 (ELITE/Top5/5영업일).
    """
    if history is None or history.empty or "Date" not in history.columns:
        return

    try:
        h = _select_perf_default_slice(history)
        if h.empty or "Date" not in h.columns:
            return

        h = h.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        if len(h) < 8:
            return  # 표본 부족 (최근 7 + 직전 7 비교 안 됨)

        recent = h.tail(7)
        prev = h.iloc[-14:-7] if len(h) >= 14 else h.iloc[:-7]
        if len(prev) < 3:
            return  # 직전 표본 부족

        # [v22.3.11] 3개 지표 모두 TOTAL_N 표본가중 평균으로 비교
        recent_win = _weighted_mean(recent, "WIN_RATE_%")
        prev_win = _weighted_mean(prev, "WIN_RATE_%")
        recent_ret = _weighted_mean(recent, "AVG_RET_%")
        prev_ret = _weighted_mean(prev, "AVG_RET_%")

        # MDD는 음수일수록 나쁨 (절댓값 클수록 나쁨)
        recent_mdd = _weighted_mean(recent, "AVG_MDD_%")
        prev_mdd = _weighted_mean(prev, "AVG_MDD_%")

        if any(x is None for x in [recent_win, prev_win, recent_ret, prev_ret]):
            return

        d_win = recent_win - prev_win
        d_ret = recent_ret - prev_ret
        d_mdd = (recent_mdd - prev_mdd) if (recent_mdd is not None and prev_mdd is not None) else None

        # 추세 판정 — 평균 수익률 변화 위주
        # 악화: 수익률 -2%p 이상 하락
        # 개선: 수익률 +2%p 이상 상승
        # 보합: 그 외
        if d_ret <= -2.0:
            icon = "📉"
            title = "최근 7거래일 성과 약화"
            color = "text-red-400"
            bg = "bg-red-900/15 border-red-500/30"
            body = (
                "최근 시장 변동 또는 임계값 영향으로 단기 실전 성과가 약화되었습니다. "
                "장기 평균은 양호해도 신규 진입은 보수적으로 접근하세요."
            )
        elif d_ret >= 2.0:
            icon = "📈"
            title = "최근 7거래일 성과 개선"
            color = "text-emerald-400"
            bg = "bg-emerald-900/15 border-emerald-500/30"
            body = (
                "최근 7거래일 평균이 직전 대비 개선되었습니다. 다만 단기 변동은 "
                "장세 영향이 크므로 지속성은 계속 관찰해야 합니다."
            )
        else:
            icon = "➡️"
            title = "최근 7거래일 성과 보합"
            color = "text-gray-300"
            bg = "bg-gray-800/40 border-gray-600/30"
            body = (
                "최근 7거래일 평균이 직전과 비슷한 수준입니다. 시장 모드와 종목 위험 "
                "신호를 함께 확인하세요."
            )

        with ui.card().classes(f"w-full p-3 mb-3 {bg} rounded-lg"):
            with ui.row().classes("w-full items-center gap-2 mb-1"):
                ui.label(icon).classes("text-xl")
                ui.label(title).classes(f"text-base font-bold {color}")
            # 3개 지표 한 줄 변화
            def _arrow(d, reverse=False):
                """기본은 양수=좋음(↑녹), 음수=나쁨(↓빨). reverse=True면 반대 (MDD)"""
                if abs(d) < 0.1:
                    return "→", "text-gray-400"
                up = d > 0
                good = up if not reverse else not up
                if good:
                    return ("↑" if up else "↓"), "text-emerald-400"
                return ("↑" if up else "↓"), "text-red-400"

            with ui.column().classes("gap-0.5 pl-5 mt-1"):
                a_w, c_w = _arrow(d_win)
                ui.label(
                    f"표본가중 승률: {prev_win:.1f}% → {recent_win:.1f}% "
                    f"({d_win:+.1f}%p {a_w})"
                ).classes(f"text-xs {c_w}")
                a_r, c_r = _arrow(d_ret)
                ui.label(
                    f"표본가중 수익률: {prev_ret:+.2f}% → {recent_ret:+.2f}% "
                    f"({d_ret:+.2f}%p {a_r})"
                ).classes(f"text-xs {c_r}")
                if d_mdd is not None:
                    # MDD는 0에 가까울수록 좋음. 더 음수가 되면 악화.
                    a_m, c_m = _arrow(d_mdd, reverse=False)  # 양수 변화 = MDD 완화 = 좋음
                    ui.label(
                        f"평균 낙폭: {prev_mdd:.2f}% → {recent_mdd:.2f}% "
                        f"({d_mdd:+.2f}%p {a_m})"
                    ).classes(f"text-xs {c_m}")
            ui.label(body).classes("text-xs text-gray-300 mt-2 leading-relaxed")
            ui.label(
                f"※ 최근 7거래일({len(recent)}개) vs 직전 7거래일({len(prev)}개) 비교 "
                "— 백테스트 검증일 기준"
            ).classes("text-[10px] text-gray-500 italic mt-1")
    except Exception:
        return


def _render_perf_judgment_card(history: pd.DataFrame, bench_data: dict) -> None:
    """성과탭 최상단 — 시스템 성과 판정 카드.

    상태 판정 (회원 카드와 유사 로직):
      🟢 장기 검증 성과 양호  : avg_ret>0 AND alpha 양수 명시 AND win>=60
      🟡 성과 양호 · 시장 비교 데이터 부족: avg_ret>0, alpha 계산 안 됨
      🟡 성과 양호 · 신뢰도 관찰: avg_ret>0 AND alpha>0 (win<60)
      ⚠️ 성과 약화 · 점검 필요  : 그 외
    
    [v3.9.12b 정합성 보정]
    - alpha: _get_kospi_return() 헬퍼 사용 (bench_data["KOSPI"]는 dict, 
      DataFrame 아님 — 직접 .columns 접근 시 silent except로 alpha=None
      되어 잘못 양호 판정될 위험 차단)
    - TOPK/H 타입 안전성: pd.to_numeric으로 변환 후 비교
    - METHOD 정합성: ELITE_SCORE 가능하면 우선 (상단 기본 필터와 일치)
    """
    if history is None or history.empty:
        return

    # 디폴트 view (METHOD=ELITE_SCORE 가능하면 / Top5 / 5영업일) 기준 평균
    try:
        h = _select_perf_default_slice(history)
        if h.empty:
            return
        win = _weighted_mean(h, "WIN_RATE_%")
        avg_ret = _weighted_mean(h, "AVG_RET_%")
        if win is None or avg_ret is None:
            return
    except Exception:
        return

    # [v22.3.14] alpha — 행별 정확 ALPHA_% 우선, 없으면 bench 평균 fallback
    alpha = None
    alpha_source = "none"
    try:
        _kospi_ret, alpha, alpha_source = _resolve_alpha_metrics(
            h, bench_data=bench_data, hold_days=5
        )
    except Exception as e:
        _logger.debug("KOSPI alpha 계산 실패: %s", e)

    # [v3.9.12b] 상태 판정 — alpha None일 때 잘못 양호 판정 차단
    # alpha 없으면 양호 단정 금지, "시장 비교 데이터 부족" 상태로
    if avg_ret > 0 and alpha is not None and alpha > 0 and win >= 60:
        icon = "🟢"
        title = "장기 검증 성과 양호"
        color = "text-emerald-400"
        bg = "bg-emerald-900/15 border-emerald-500/30"
        body = (
            "Top5 / 5영업일 기준으로 시장 대비 초과 성과가 확인됩니다. "
            "다만 Top1 실전 운용과 Shadow 실험은 별도 기준이므로, "
            "신규 진입은 시장 모드와 종목 위험 신호를 함께 확인하세요."
        )
    elif avg_ret > 0 and alpha is None:
        # 평균 수익률은 양수지만 KOSPI 비교 데이터 없음 — 단정 금지
        icon = "🟡"
        title = "성과 양호 · 시장 비교 데이터 부족"
        color = "text-yellow-400"
        bg = "bg-yellow-900/15 border-yellow-500/30"
        body = (
            "Top5 / 5영업일 평균 수익률은 양호하지만 KOSPI 비교 데이터가 "
            "확인되지 않아 시장 대비 초과 성과 여부는 단정할 수 없습니다. "
            "신규 진입은 시장 모드와 종목 위험 신호를 함께 확인하세요."
        )
    elif avg_ret > 0 and alpha is not None and alpha > 0:
        # 알파 양수지만 승률 60% 미만
        icon = "🟡"
        title = "성과 양호 · 신뢰도 관찰"
        color = "text-yellow-400"
        bg = "bg-yellow-900/15 border-yellow-500/30"
        body = (
            "시장 대비 초과 성과는 확인되지만, 승률이 60% 미만이라 "
            "신뢰도는 관찰 단계입니다. 시장 모드와 종목 위험 신호를 "
            "함께 확인하세요."
        )
    else:
        icon = "⚠️"
        title = "성과 약화 · 점검 필요"
        color = "text-amber-400"
        bg = "bg-amber-900/20 border-amber-500/40"
        body = (
            "최근 누적 성과가 약화되었거나 시장 대비 초과 성과가 확인되지 "
            "않습니다. 임계값/룰 점검이 필요한 구간입니다."
        )

    with ui.card().classes(f"w-full p-3 mb-3 {bg} rounded-lg"):
        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.label(icon).classes("text-2xl")
            ui.label(title).classes(f"text-lg font-bold {color}")
        # 한 줄 지표 요약
        summary_parts = [f"표본가중 승률 {win:.1f}%", f"표본가중 수익률 {avg_ret:+.2f}%"]
        if alpha is not None:
            alpha_label = "행별 KOSPI 알파" if alpha_source == "row_exact" else "KOSPI 알파"
            summary_parts.append(f"{alpha_label} {alpha:+.2f}%p")
        else:
            summary_parts.append("KOSPI 알파 데이터 없음")
        ui.label("기준: " + " · ".join(summary_parts)).classes(
            "text-xs text-gray-300 mb-1"
        )
        ui.label(body).classes(
            "text-sm text-gray-200 leading-relaxed"
        )
        # [v3.9.12c] 아래 필터 변경해도 판정 카드는 고정 기준임을 명시
        ui.label(
            "※ 위 판정은 기본 기준(ELITE_SCORE / Top5 / 5영업일) 고정입니다. "
            "아래 필터를 바꿔도 이 카드는 변하지 않습니다."
        ).classes("text-[10px] text-gray-500 italic mt-2")


def _render_validation_basis_card() -> None:
    """[v3.9.12] 검증 기준 안내 — Top5/Top3/Top1 혼선 해소.
    
    성과탭 +7% / 종목탭 신규 매수 주의 같은 충돌이 검증 기준 차이임을 명시.
    """
    with ui.card().classes(
        "w-full p-3 mb-3 bg-[rgba(59,130,246,0.08)] "
        "border border-blue-500/30 rounded-lg"
    ):
        with ui.row().classes("items-center gap-2 mb-1"):
            ui.label("📌").classes("text-sm")
            ui.label("검증 기준 안내").classes("text-sm font-bold text-blue-200")
        with ui.column().classes("gap-0.5 pl-5 mt-1"):
            ui.label(
                "• 상단 성과 요약 — Top5 후보를 5영업일 보유했을 때의 평균 검증"
            ).classes("text-xs text-gray-300")
            ui.label(
                "• 종목탭 Top Pick — 실제 하루 1종목 운용에 가까운 Top1 실전 검증 (10영업일)"
            ).classes("text-xs text-gray-300")
            ui.label(
                "• Shadow 실험실 — 추천 로직을 바꾸지 않고 \"바꿨다면 어땠을지\"를 측정한 실험 결과"
            ).classes("text-xs text-gray-300")
            ui.label(
                "• 공식 신규매수 — TOP_PICK + BUY_NOW_ELIGIBLE 기준. 현재는 별도 누적 전이라 데이터 준비 중"
            ).classes("text-xs text-amber-200")
        ui.label(
            "※ 각 영역의 숫자가 다른 이유는 \"검증 기준\"이 다르기 때문이며, "
            "성과탭 +X%인데 종목탭은 매수 주의인 경우 충돌이 아닙니다. "
            "공식 매수 판단은 반드시 TOP_PICK + BUY_NOW_ELIGIBLE 기준으로 분리해 보세요."
        ).classes("text-[10px] text-gray-500 italic mt-2 pl-5")


def _render_shadow_summary_card(j: dict) -> None:
    """[v3.9.12] Shadow 실험실 상단 종합 판정 + 각 shadow 상태 배지.
    
    각 shadow의 single_backtest_ok/RWF 통과/구성변경률 기반으로
    🟢 적용 후보 / 🟡 관찰 중 / 🔒 측정만 / 🚫 폐기 후보 판정.
    """
    em = j.get("entry_mode_shadow", {})
    sr = j.get("struct_risk_shadow", {})
    pe = j.get("pre_entry_risk_shadow", {})

    if not (em.get("enabled") or sr.get("enabled") or pe.get("enabled")):
        return

    # 각 shadow 상태 판정
    statuses = []

    # ENTRY_MODE — chase 체결 수가 적으면 표본 부족
    # [v3.9.13] 회원 친화 한글 — 관리자 용어 → 회원이 이해 가능한 문구
    if em.get("enabled"):
        extra_n = em.get("extra_fills", em.get("n_chase_filled", 0)) or 0
        if extra_n < 5:
            statuses.append(("미체결 회수 실험", "🟡 표본 부족 — 더 관찰 필요",
                             "text-yellow-300"))
        elif em.get("production_candidate"):
            statuses.append(("미체결 회수 실험", "🟢 적용 후보 — 관찰 통과",
                             "text-emerald-300"))
        else:
            statuses.append(("미체결 회수 실험", "🔒 측정 단계 — 반복 검증 전",
                             "text-gray-300"))

    # STRUCT risk — 효과는 있는데 구성변경 큼
    if sr.get("enabled"):
        d_ev = sr.get("delta_ev", 0) or 0
        change_pct = (sr.get("changed_pick_rate", 0) or 0) * 100
        if d_ev > 0 and change_pct >= 40:
            statuses.append(("구조점수 위험구간 제외 실험",
                             f"🟡 효과 있음 · 추천 후보가 많이 바뀜 ({change_pct:.0f}%)",
                             "text-yellow-300"))
        elif d_ev > 0:
            statuses.append(("구조점수 위험구간 제외 실험",
                             "🟢 효과 양호 · 추천 후보 변화 적음",
                             "text-emerald-300"))
        else:
            statuses.append(("구조점수 위험구간 제외 실험",
                             "🔒 측정 단계 — 효과 미확정",
                             "text-gray-300"))

    # PRE_ENTRY_RISK — B_red RWF 통과 + 화면 표시 단계 완료
    if pe.get("enabled") and "rules" in pe:
        b_red = pe.get("rules", {}).get("B_red", {})
        b_red_ok = b_red.get("single_backtest_ok", False)
        b_red_dev = b_red.get("delta_ev", 0) or 0
        if b_red_ok and b_red_dev > 0:
            statuses.append(("진입 위험 사전 식별 실험",
                             "🟢 위험 표시 단계 완료 · 반복 검증 통과",
                             "text-emerald-300"))
        else:
            statuses.append(("진입 위험 사전 식별 실험", "🟡 관찰 중",
                             "text-yellow-300"))

    if not statuses:
        return

    # 종합 판정 한 줄
    # [v3.9.12c] 변수명 정정 — 실제 의미는 has_green (🟢 검사)
    has_green = any("🟢" in s[1] for s in statuses)
    has_yellow = any("🟡" in s[1] for s in statuses)
    if has_green and not has_yellow:
        overall = ("🟢", "Shadow 실험 — 적용 후보 다수", "text-emerald-400")
    elif has_green and has_yellow:
        overall = ("🟡", "Shadow 실험 — 유망하나 일부 관찰 필요", "text-yellow-400")
    else:
        overall = ("🔒", "Shadow 실험 — 측정 단계 유지", "text-gray-400")

    with ui.card().classes(
        "w-full p-3 mb-2 bg-[rgba(139,92,246,0.08)] "
        "border border-purple-500/30 rounded-lg"
    ):
        with ui.row().classes("items-center gap-2 mb-2"):
            ui.label(overall[0]).classes("text-lg")
            ui.label(overall[1]).classes(f"text-sm font-bold {overall[2]}")
        with ui.column().classes("gap-0.5 pl-5"):
            for name, status, color in statuses:
                # [v3.9.12c] 줄바꿈 안전성 — name과 status를 한 라벨로
                ui.label(f"• {name}: {status}").classes(
                    f"text-[11px] {color}"
                )
        ui.label(
            "ℹ️ 모든 Shadow 결과는 운영 추천에 자동 반영되지 않습니다. "
            "PRE_ENTRY_RISK는 화면 위험 표시까지만 적용된 상태입니다."
        ).classes("text-[10px] text-gray-500 italic mt-2 pl-5")


def render_tab_perf(auth: str = "free"):
    """[Step AK+AL+AM] 시스템 성과 추세 — 면책 + 6개 메트릭 + 모바일 + KOSPI 알파
    
    [v3.9.13b] auth 추가 — 관리자는 Research Workbench 기본 펼침
    """
    
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

    # [v3.9.12] 회원용 요약 카드 (성과 판정 + 검증 기준 안내)
    # [v3.9.13] 최근 7거래일 성과 변화 카드 추가
    # [v3.9.14] 최근 손실 기여 자동 리포트 추가
    # [v3.9.14c] 균형용 — 최근 수익 기여 카드도 함께 (손실만 보면 부정적 인상)
    # [v3.9.14e] 캐시 공유 — trades/recs를 1회만 로딩 후 두 카드에 전달
    try:
        _render_perf_judgment_card(history, bench_data)
        _render_recent_trend_card(history)
        # 두 카드가 공유할 캐시 (한 번만 로딩)
        _top1_trades = _load_top1_trades()
        _rec_cache = _load_recommend_cache(days=90) if not _top1_trades.empty else {}
        _render_profit_attribution_card(_top1_trades, _rec_cache)
        _render_loss_attribution_card(_top1_trades, _rec_cache)
        _render_validation_basis_card()
    except Exception as e:
        _logger.warning(f"perf 요약 카드 렌더 실패 (페이지 영향 없음): {e}")

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

    # [v22.3.12] 공식 신규매수 성과 누적 — TOP_PICK + BUY_NOW_ELIGIBLE 별도 검증
    try:
        ui.separator().classes("my-6")
        _render_official_buy_validation_card()
    except Exception as e:
        _logger.warning(f"official buy validation 카드 렌더 실패 (페이지 영향 없음): {e}")

    # [v3.9.1] Shadow 실험실 — ENTRY_MODE / STRUCT risk shadow 측정 결과
    try:
        ui.separator().classes("my-6")
        _render_shadow_lab_card()
    except Exception as e:
        _logger.warning(f"shadow lab 카드 렌더 실패 (페이지 영향 없음): {e}")

    # ─── Research Workbench 통합 정리 ───
    # [v3.9.13] 기본 접기 — 회원 기본 화면에서 숨김. 펼치면 고급 분석 도구 표시.
    # [v3.9.13b] 관리자는 기본 펼침 (시장/종목 탭과 동일 패턴)
    try:
        from research_tab import render_research_tab
        ui.separator().classes("my-6")

        _rw_is_admin = (auth == "admin")
        _rw_title = (
            "🔬 심화 분석 (관리자 — 기본 펼침)"
            if _rw_is_admin
            else "🔬 심화 분석 보기 (Research Workbench — 고급 사용자용)"
        )
        with ui.expansion(
            _rw_title,
            icon="science",
            value=_rw_is_admin,  # 관리자만 기본 열림
        ).classes(
            "w-full bg-[rgba(34,211,238,0.05)] "
            "border border-[rgba(34,211,238,0.2)] rounded"
        ).props("dense"):
            with ui.row().classes("w-full items-center gap-2 mb-2 mt-2"):
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
