"""
SwingPicker Research Workbench — 성과 분석 탭 (NiceGUI)

Phase 1-5: 추천 성과를 빠르게 확인하는 최소 대시보드.
기존 rank_validation_*.csv, reality_check_*.csv, recommend_*.csv 활용.

사용법 (main.py 내부에서):
    from research_tab import render_research_tab
    # Tab 내에서:
    render_research_tab(data_dir=DATA_DIR)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from glob import glob
from typing import Optional, List

import pandas as pd
import numpy as np

# NiceGUI (main.py에서 이미 import된 상태)
try:
    from nicegui import ui
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    NICEGUI_OK = True
except ImportError:
    NICEGUI_OK = False


# ══════════════════════════════════════════════════════════
#  데이터 로더 (순수 Python — 프레임워크 무관)
# ══════════════════════════════════════════════════════════

def load_rank_validation_summaries(data_dir: str = "data") -> Optional[pd.DataFrame]:
    """rank_validation_summary_*.csv 파일들을 통합 로드"""
    pattern = os.path.join(data_dir, "rank_validation_summary_*.csv")
    files = sorted(glob(pattern))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            basename = os.path.basename(f)
            date_part = basename.replace("rank_validation_summary_", "").replace(".csv", "")
            if date_part == "latest":
                df["eval_date"] = datetime.now().strftime("%Y%m%d")
            else:
                df["eval_date"] = date_part
            dfs.append(df)
        except Exception:
            continue

    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_reality_check(data_dir: str = "data") -> Optional[pd.DataFrame]:
    """reality_check_*.csv 파일들을 통합 로드"""
    pattern = os.path.join(data_dir, "reality_check_*.csv")
    files = sorted(glob(pattern))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            continue

    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_recommend_csvs(data_dir: str = "data", days: int = 30) -> Optional[pd.DataFrame]:
    """최근 N일 recommend_*.csv 파일들을 통합 로드"""
    pattern = os.path.join(data_dir, "recommend_*.csv")
    files = sorted(glob(pattern))
    if not files:
        return None

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    recent = []
    for f in files:
        basename = os.path.basename(f)
        date_part = basename.replace("recommend_", "").replace(".csv", "")
        if date_part in ("latest", "latest_cp949"):
            continue
        if date_part >= cutoff:
            recent.append(f)

    if not recent:
        recent = [f for f in files if "latest" not in os.path.basename(f)][-min(days, len(files)):]

    dfs = []
    for f in recent:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            continue

    return pd.concat(dfs, ignore_index=True) if dfs else None


# ══════════════════════════════════════════════════════════
#  성과 계산 (순수 Python — 프레임워크 무관)
# ══════════════════════════════════════════════════════════

def _find_return_col(df: pd.DataFrame) -> Optional[str]:
    """수익률 컬럼 자동 탐지"""
    candidates = [
        "realized_ret", "ret_5d_%", "ret_7d_%", "return_pct",
        "AVG_RET_%", "avg_ret", "RET_%",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_score_col(df: pd.DataFrame) -> Optional[str]:
    """점수 컬럼 자동 탐지"""
    candidates = ["FINAL_SCORE", "RANK_SCORE", "글로벌점수", "global_score"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def calc_topk_performance(df: pd.DataFrame, k_values: list = None) -> pd.DataFrame:
    """Top-K별 평균 수익률 / hit-rate 계산"""
    if k_values is None:
        k_values = [5, 10, 20]

    ret_col = _find_return_col(df)
    score_col = _find_score_col(df)

    if ret_col is None or score_col is None:
        return pd.DataFrame()

    results = []
    for k in k_values:
        topk = df.nlargest(k, score_col)
        rets = pd.to_numeric(topk[ret_col], errors="coerce").dropna()
        if rets.empty:
            continue

        results.append({
            "Top-K": k,
            "종목수": len(rets),
            "평균수익률(%)": round(rets.mean(), 2),
            "중앙값(%)": round(rets.median(), 2),
            "승률(%)": round((rets > 0).mean() * 100, 1),
            "최대손실(%)": round(rets.min(), 2),
            "최대수익(%)": round(rets.max(), 2),
        })

    return pd.DataFrame(results)


def calc_score_band_stats(df: pd.DataFrame) -> pd.DataFrame:
    """점수 구간별 승률 테이블"""
    ret_col = _find_return_col(df)
    score_col = _find_score_col(df)

    if ret_col is None or score_col is None:
        return pd.DataFrame()

    df = df.copy()
    bins = [0, 50, 60, 70, 80, 90, 100]
    labels = ["0~50", "50~60", "60~70", "70~80", "80~90", "90+"]

    df["_score_band"] = pd.cut(
        pd.to_numeric(df[score_col], errors="coerce"),
        bins=bins, labels=labels, right=False,
    )
    df["_ret"] = pd.to_numeric(df[ret_col], errors="coerce")

    grouped = df.groupby("_score_band", observed=True)["_ret"]
    stats = grouped.agg(
        종목수="count",
        평균수익률="mean",
        승률=lambda x: (x > 0).mean() * 100 if len(x) > 0 else 0,
        중앙값="median",
    ).round(2)

    stats.index.name = "점수구간"
    return stats.reset_index()


# ══════════════════════════════════════════════════════════
#  NiceGUI 렌더링
# ══════════════════════════════════════════════════════════

def render_research_tab(data_dir: str = "data"):
    """
    NiceGUI 대시보드 내 '📊 Research Workbench' 렌더링.

    main.py의 Tab 내에서 호출:
        with ui.tab_panel(tab_research):
            render_research_tab(DATA_DIR)
    """
    if not NICEGUI_OK:
        print("NiceGUI가 설치되지 않았습니다.")
        return

    # ── 1) 점수 구간별 승률 (reality_check 또는 recommend 기반) ──
    ui.label("📊 점수 구간별 승률 분석").classes("text-xl font-bold text-white mt-4 mb-2")

    rc_df = load_reality_check(data_dir)
    rec_df = load_recommend_csvs(data_dir, days=30)
    source_df = rc_df if rc_df is not None and not rc_df.empty else rec_df

    if source_df is not None and not source_df.empty:
        band_stats = calc_score_band_stats(source_df)
        if not band_stats.empty:
            columns = [
                {"name": col, "label": col, "field": col, "align": "center"}
                for col in band_stats.columns
            ]
            rows = band_stats.to_dict("records")
            ui.table(
                columns=columns, rows=rows, row_key="점수구간",
                pagination={"rowsPerPage": 10},
            ).classes("w-full").props("dense dark flat bordered")

            # 승률 바 차트
            colors = ["#EF5350", "#FF7043", "#FFA726", "#66BB6A", "#42A5F5", "#AB47BC"]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=band_stats["점수구간"],
                y=band_stats["승률"],
                marker_color=colors[:len(band_stats)],
                text=band_stats["승률"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto",
            ))
            fig.update_layout(
                title="점수 구간별 승률",
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                yaxis_title="승률(%)",
                yaxis_range=[0, 100],
            )
            ui.plotly(fig).classes("w-full")
        else:
            ui.label("점수/수익률 컬럼을 찾을 수 없습니다.").classes("text-gray-400")
    else:
        ui.label("reality_check 또는 recommend 데이터가 없습니다.").classes("text-gray-400")

    # ── 2) Top-K 성과 테이블 ──
    ui.label("🏆 Top-K 추천 성과").classes("text-xl font-bold text-white mt-6 mb-2")

    if source_df is not None and not source_df.empty:
        topk_stats = calc_topk_performance(source_df)
        if not topk_stats.empty:
            columns = [
                {"name": col, "label": col, "field": col, "align": "center"}
                for col in topk_stats.columns
            ]
            rows = topk_stats.to_dict("records")
            ui.table(
                columns=columns, rows=rows, row_key="Top-K",
            ).classes("w-full").props("dense dark flat bordered")
        else:
            ui.label("수익률 데이터가 부족합니다.").classes("text-gray-400")
    else:
        ui.label("데이터가 없습니다.").classes("text-gray-400")

    # ── 3) Reality Check 최근 데이터 ──
    ui.label("🔍 Reality Check 최근 데이터").classes("text-xl font-bold text-white mt-6 mb-2")

    if rc_df is not None and not rc_df.empty:
        ui.label(f"총 {len(rc_df)}건").classes("text-gray-300 mb-2")

        # 최근 20건, 표시 컬럼 제한
        display_cols = [c for c in rc_df.columns if not c.startswith("_")][:10]
        recent = rc_df.tail(20)[display_cols] if display_cols else rc_df.tail(20)

        columns = [
            {"name": col, "label": col, "field": col, "align": "center"}
            for col in recent.columns
        ]
        rows = recent.fillna("").to_dict("records")
        ui.table(
            columns=columns, rows=rows,
            pagination={"rowsPerPage": 10},
        ).classes("w-full").props("dense dark flat bordered")
    else:
        ui.label("reality_check 데이터가 없습니다.").classes("text-gray-400")
