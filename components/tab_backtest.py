# -*- coding: utf-8 -*-
"""
tab_backtest.py — 🧪 전략 샌드박스 (백테스트 시뮬레이터)
═══════════════════════════════════════════════════
유저가 슬라이더로 매매 조건을 세팅 → 과거 추천 데이터 기반 수익곡선/MDD/승률 시각화
Premium 전용 킬러 기능
"""
import os
import io
import logging
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nicegui import ui

_logger = logging.getLogger(__name__)
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ── 보유기간 → 수익률 컬럼 매핑 ──
_RET_MAP = [
    (3,   "ret_1d_%"),
    (7,   "ret_5d_%"),
    (15,  "ret_10d_%"),
    (40,  "ret_20d_%"),
    (90,  "ret_60d_%"),
    (999, "ret_120d_%"),
]


def _get_ret_col(hold_days: int) -> str:
    for threshold, col in _RET_MAP:
        if hold_days <= threshold:
            return col
    return "ret_20d_%"


def _load_recommend_files() -> pd.DataFrame:
    """data/ 내 모든 recommend_*.csv 로드 (날짜별 병합)"""
    pattern = os.path.join(_DATA_DIR, "recommend_*.csv")
    files = sorted(glob(pattern))
    dfs = []
    for f in files:
        basename = os.path.basename(f)
        if basename == "recommend_latest.csv":
            continue
        try:
            date_str = basename.replace("recommend_", "").replace(".csv", "")
            if not date_str.isdigit() or len(date_str) != 8:
                continue
            df = pd.read_csv(f, dtype={"종목코드": str})
            df["rec_date"] = date_str
            dfs.append(df)
        except Exception as e:
            _logger.debug(f"파일 로드 실패 {f}: {e}")

    # latest도 추가 (날짜 추출)
    latest_path = os.path.join(_DATA_DIR, "recommend_latest.csv")
    if os.path.exists(latest_path):
        try:
            df = pd.read_csv(latest_path, dtype={"종목코드": str})
            date_col = next((c for c in ["기준일", "trade_date", "DATA_DATE"] if c in df.columns), None)
            if date_col:
                date_str = str(df[date_col].iloc[0]).replace("-", "")[:8]
            else:
                date_str = datetime.now().strftime("%Y%m%d")
            # 중복 방지
            existing_dates = {d["rec_date"].iloc[0] for d in dfs} if dfs else set()
            if date_str not in existing_dates:
                df["rec_date"] = date_str
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)

    # ── 종목명 오염 복구 (코드==이름인 경우) ──
    if "종목코드" in merged.columns and "종목명" in merged.columns:
        merged["종목명"] = merged["종목명"].astype(str)
        mask = merged["종목명"].str.match(r'^\d+$')
        if mask.any():
            code_to_name = _load_code_to_name()
            if code_to_name:
                merged.loc[mask, "종목명"] = (
                    merged.loc[mask, "종목코드"].astype(str).str.zfill(6)
                    .map(code_to_name)
                    .fillna(merged.loc[mask, "종목명"])
                )

    return merged


def _load_code_to_name() -> dict:
    """종목코드→종목명 매핑 로드 (krx_names CSV → data_store KRX 캐시)"""
    # 1순위: krx_names_latest.csv
    names_path = os.path.join(_DATA_DIR, "krx_names_latest.csv")
    if os.path.exists(names_path):
        try:
            ndf = pd.read_csv(names_path, dtype=str)
            if "종목코드" in ndf.columns and "종목명" in ndf.columns:
                c2n = dict(zip(
                    ndf["종목코드"].astype(str).str.zfill(6),
                    ndf["종목명"]
                ))
                c2n = {c: n for c, n in c2n.items() if c != n and n and not n.isdigit()}
                if c2n:
                    return c2n
        except Exception:
            pass

    # 2순위: data_store의 KRX 캐시
    try:
        from services.data_store import _KRX_NAME_MAP
        if _KRX_NAME_MAP:
            return {v: k for k, v in _KRX_NAME_MAP.items()}
    except Exception:
        pass

    # 3순위: store.scored에서 추출
    try:
        from services.data_store import store
        if store.loaded:
            scored = store.scored
            if "종목코드" in scored.columns and "종목명" in scored.columns:
                valid = scored[~scored["종목명"].astype(str).str.match(r'^\d+$')]
                if not valid.empty:
                    return dict(zip(
                        valid["종목코드"].astype(str).str.zfill(6),
                        valid["종목명"]
                    ))
    except Exception:
        pass

    return {}


def _run_backtest(all_recs: pd.DataFrame, min_score: int, hold_days: int,
                  stop_pct: float, target_pct: float, top_k: int,
                  cost_pct: float) -> dict:
    """
    백테스트 실행.
    - recommend CSV의 ret_Xd_% 컬럼 활용 (실제 과거 수익률)
    - 스톱/타겟 적용, 비용 차감
    """
    ret_col = _get_ret_col(hold_days)

    # 점수 컬럼
    score_col = None
    for c in ["DISPLAY_SCORE", "FINAL_SCORE", "TOTAL_SCORE", "RANK_SCORE"]:
        if c in all_recs.columns:
            score_col = c
            break
    if score_col is None or ret_col not in all_recs.columns:
        return {"error": "필요한 데이터 컬럼 없음"}

    all_recs[score_col] = pd.to_numeric(all_recs[score_col], errors="coerce").fillna(0)
    all_recs[ret_col] = pd.to_numeric(all_recs[ret_col], errors="coerce").fillna(0)

    # 점수 필터
    filtered = all_recs[all_recs[score_col] >= min_score].copy()
    if filtered.empty:
        return {"error": f"{min_score}점 이상 종목이 없습니다"}

    # 날짜별 그룹 → Top-K 선별
    trades = []
    for date, grp in filtered.groupby("rec_date"):
        top = grp.nlargest(top_k, score_col)
        for _, row in top.iterrows():
            raw_ret = float(row[ret_col])

            # 스톱/타겟 적용
            if raw_ret <= -stop_pct:
                applied_ret = -stop_pct
                status = "STOP"
            elif raw_ret >= target_pct:
                applied_ret = target_pct
                status = "WIN"
            else:
                applied_ret = raw_ret
                status = "HOLD_EXIT"

            # 비용 차감
            net_ret = applied_ret - cost_pct

            trades.append({
                "rec_date": str(date),
                "code": str(row.get("종목코드", "")),
                "name": str(row.get("종목명", "")),
                "score": float(row[score_col]),
                "raw_ret": round(raw_ret, 2),
                "net_ret": round(net_ret, 2),
                "status": status,
            })

    if not trades:
        return {"error": "조건에 맞는 거래가 없습니다"}

    df = pd.DataFrame(trades).sort_values("rec_date")

    # ── 날짜별 포트폴리오 수익률 (균등 배분) ──
    daily_rets = df.groupby("rec_date")["net_ret"].mean()
    daily_rets = daily_rets.sort_index()

    # ── 누적 수익곡선 ──
    equity = (1 + daily_rets / 100).cumprod()
    equity_series = pd.DataFrame({"date": equity.index, "equity": equity.values})

    # ── MDD 계산 ──
    peak = equity.cummax()
    drawdown = ((equity - peak) / peak) * 100
    mdd = drawdown.min()
    dd_series = pd.DataFrame({"date": drawdown.index, "drawdown": drawdown.values})

    # ── 통계 ──
    total_trades = len(df)
    wins = (df["net_ret"] > 0).sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    avg_win = df.loc[df["net_ret"] > 0, "net_ret"].mean() if wins > 0 else 0
    avg_loss = df.loc[df["net_ret"] <= 0, "net_ret"].mean() if (total_trades - wins) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    total_return = (equity.iloc[-1] - 1) * 100 if len(equity) > 0 else 0
    trading_days = len(daily_rets)

    # 상태별 분포
    status_dist = df["status"].value_counts().to_dict()

    # 상위/하위 종목
    best_trades = df.nlargest(5, "net_ret")[["name", "score", "net_ret", "status"]].to_dict("records")
    worst_trades = df.nsmallest(5, "net_ret")[["name", "score", "net_ret", "status"]].to_dict("records")

    return {
        "total_return": round(total_return, 2),
        "mdd": round(mdd, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
        "trading_days": trading_days,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "status_dist": status_dist,
        "equity": equity_series,
        "drawdown": dd_series,
        "best_trades": best_trades,
        "worst_trades": worst_trades,
        "hold_col": ret_col,
    }


def _plotly_dark(fig, height=350):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="white",
        margin=dict(t=40, b=30, l=50, r=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def render_tab_backtest(df, auth):
    """Tab: 🧪 전략 샌드박스"""

    # ── 프리미엄 게이트 ──
    if auth not in ("admin", "prime"):
        with ui.card().classes("w-full p-8 bg-[#1a1a2e] border border-gray-700 rounded-xl text-center"):
            ui.label("🔒 전략 샌드박스").classes("text-2xl font-bold text-white mb-4")
            ui.label("Prime 구독자 전용 기능입니다").classes("text-gray-400 mb-2")
            ui.label("과거 추천 데이터를 기반으로 나만의 전략을 백테스트하고\n"
                     "수익곡선, 최대낙폭(MDD), 승률을 확인하세요.").classes("text-gray-500 text-sm")
            with ui.row().classes("justify-center mt-6 gap-4"):
                ui.html("""
                <div style="text-align:center; padding:16px; border:1px solid #374151; border-radius:12px; min-width:120px;">
                    <div style="font-size:24px;">📊</div>
                    <div style="color:#9CA3AF; font-size:13px; margin-top:4px;">수익곡선</div>
                </div>
                <div style="text-align:center; padding:16px; border:1px solid #374151; border-radius:12px; min-width:120px;">
                    <div style="font-size:24px;">📉</div>
                    <div style="color:#9CA3AF; font-size:13px; margin-top:4px;">MDD 분석</div>
                </div>
                <div style="text-align:center; padding:16px; border:1px solid #374151; border-radius:12px; min-width:120px;">
                    <div style="font-size:24px;">🎯</div>
                    <div style="color:#9CA3AF; font-size:13px; margin-top:4px;">승률/손익비</div>
                </div>
                """)
        return

    # ── Premium 유저: 전체 UI ──
    ui.label("🧪 전략 샌드박스").classes("text-2xl font-bold text-white mb-1")
    ui.label("과거 추천 데이터 기반 백테스트 시뮬레이터").classes("text-gray-400 text-sm mb-4")

    # ── 파라미터 패널 ──
    with ui.card().classes("w-full p-6 bg-[#1a1a2e] border border-gray-700 rounded-xl mb-4"):
        ui.label("⚙️ 전략 파라미터").classes("text-lg font-bold text-white mb-4")

        with ui.row().classes("w-full gap-6 flex-wrap"):
            # 진입 조건
            with ui.column().classes("flex-1 min-w-[250px]"):
                ui.label("📋 진입 조건").classes("text-sm font-bold text-blue-400 mb-2")
                sl_score = ui.slider(min=40, max=95, value=70, step=5).classes("w-full")
                ui.label("").bind_text_from(sl_score, "value", backward=lambda v: f"최소 점수: {v}점")

                sl_topk = ui.slider(min=3, max=30, value=10, step=1).classes("w-full")
                ui.label("").bind_text_from(sl_topk, "value", backward=lambda v: f"일일 편입 종목 수: {v}개")

            # 매매 규칙
            with ui.column().classes("flex-1 min-w-[250px]"):
                ui.label("💰 매매 규칙").classes("text-sm font-bold text-green-400 mb-2")
                sl_hold = ui.slider(min=1, max=60, value=10, step=1).classes("w-full")
                lbl_hold = ui.label("")
                lbl_hold.bind_text_from(sl_hold, "value",
                                        backward=lambda v: f"보유 기간: {v}일 → {_get_ret_col(v)}")

                sl_target = ui.slider(min=2, max=30, value=10, step=1).classes("w-full")
                ui.label("").bind_text_from(sl_target, "value", backward=lambda v: f"익절선: +{v}%")

            # 리스크 관리
            with ui.column().classes("flex-1 min-w-[250px]"):
                ui.label("🛡️ 리스크 관리").classes("text-sm font-bold text-red-400 mb-2")
                sl_stop = ui.slider(min=2, max=15, value=5, step=1).classes("w-full")
                ui.label("").bind_text_from(sl_stop, "value", backward=lambda v: f"손절선: -{v}%")

                sl_cost = ui.slider(min=0, max=1.0, value=0.4, step=0.05).classes("w-full")
                ui.label("").bind_text_from(sl_cost, "value", backward=lambda v: f"왕복 비용: {v:.2f}%")

    # ── 실행 버튼 + 결과 영역 ──
    result_container = ui.column().classes("w-full")

    async def run_simulation():
        result_container.clear()
        with result_container:
            spinner = ui.spinner("dots", size="lg", color="blue")

        from async_helpers import run_sync

        all_recs = await run_sync(_load_recommend_files)
        if all_recs.empty:
            result_container.clear()
            with result_container:
                ui.label("❌ data/ 폴더에 recommend_*.csv 파일이 없습니다").classes("text-red-400")
            return

        _min_score = int(sl_score.value)
        _hold_days = int(sl_hold.value)
        _stop_pct = float(sl_stop.value)
        _target_pct = float(sl_target.value)
        _top_k = int(sl_topk.value)
        _cost_pct = float(sl_cost.value)

        result = await run_sync(
            lambda: _run_backtest(all_recs, _min_score, _hold_days,
                                  _stop_pct, _target_pct, _top_k, _cost_pct)
        )

        result_container.clear()
        with result_container:
            if "error" in result:
                ui.label(f"⚠️ {result['error']}").classes("text-yellow-400 text-lg")
                return

            # ── 메트릭 카드 ──
            with ui.row().classes("w-full gap-3 flex-wrap mb-4"):
                _stat_card("📈 총 수익률", f"{result['total_return']:+.2f}%",
                           result["total_return"] >= 0)
                _stat_card("📉 최대낙폭", f"{result['mdd']:.2f}%", False)
                _stat_card("🎯 승률", f"{result['win_rate']:.1f}%",
                           result["win_rate"] >= 50)
                _stat_card("⚖️ 손익비", f"{result['profit_factor']:.2f}",
                           result["profit_factor"] >= 1.5)
                _stat_card("🔢 총 거래", f"{result['total_trades']}회", True)
                _stat_card("📅 분석일수", f"{result['trading_days']}일", True)

            # ── 평균 수익/손실 ──
            with ui.row().classes("w-full gap-3 mb-4"):
                _stat_card("💚 평균 수익", f"+{result['avg_win']:.2f}%", True)
                _stat_card("💔 평균 손실", f"{result['avg_loss']:.2f}%", False)

                # 상태 분포
                dist = result.get("status_dist", {})
                dist_text = " / ".join(f"{k}: {v}" for k, v in dist.items())
                with ui.card().classes("p-3 flex-1 bg-[#1a1a2e] border border-gray-700 rounded-xl"):
                    ui.label("🏷️ 청산 유형").classes("text-xs text-gray-400")
                    ui.label(dist_text).classes("text-sm text-white mt-1")

            # ── 수익곡선 차트 ──
            eq = result["equity"]
            if not eq.empty:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=eq["date"], y=eq["equity"],
                    mode="lines", fill="tozeroy",
                    line=dict(color="#3B82F6", width=2),
                    fillcolor="rgba(59,130,246,0.1)",
                    name="자산 가치",
                ))
                fig_eq.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                annotation_text="원금", annotation_font_color="gray")
                fig_eq.update_layout(title="📈 자산 성장 곡선 (복리, 균등배분)")
                _plotly_dark(fig_eq, 380)
                ui.plotly(fig_eq).classes("w-full")

            # ── Drawdown 차트 ──
            dd = result["drawdown"]
            if not dd.empty:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=dd["date"], y=dd["drawdown"],
                    mode="lines", fill="tozeroy",
                    line=dict(color="#EF4444", width=1.5),
                    fillcolor="rgba(239,68,68,0.15)",
                    name="낙폭",
                ))
                fig_dd.update_layout(title=f"📉 Drawdown (MDD: {result['mdd']:.2f}%)")
                _plotly_dark(fig_dd, 250)
                ui.plotly(fig_dd).classes("w-full")

            # ── Top/Bottom 종목 ──
            with ui.row().classes("w-full gap-4 mt-4 flex-wrap"):
                with ui.card().classes("flex-1 min-w-[300px] p-4 bg-[#1a1a2e] border border-gray-700 rounded-xl"):
                    ui.label("🏆 최고 수익 종목").classes("text-sm font-bold text-green-400 mb-2")
                    for t in result.get("best_trades", []):
                        with ui.row().classes("w-full justify-between items-center py-1 border-b border-gray-800"):
                            ui.label(f"{t['name']}").classes("text-white text-sm")
                            with ui.row().classes("gap-2"):
                                ui.badge(f"{t['score']:.0f}점").props("color=blue")
                                color = "green" if t["net_ret"] > 0 else "red"
                                ui.label(f"{t['net_ret']:+.1f}%").classes(f"text-{color}-400 text-sm font-bold")

                with ui.card().classes("flex-1 min-w-[300px] p-4 bg-[#1a1a2e] border border-gray-700 rounded-xl"):
                    ui.label("💀 최대 손실 종목").classes("text-sm font-bold text-red-400 mb-2")
                    for t in result.get("worst_trades", []):
                        with ui.row().classes("w-full justify-between items-center py-1 border-b border-gray-800"):
                            ui.label(f"{t['name']}").classes("text-white text-sm")
                            with ui.row().classes("gap-2"):
                                ui.badge(f"{t['score']:.0f}점").props("color=blue")
                                color = "green" if t["net_ret"] > 0 else "red"
                                ui.label(f"{t['net_ret']:+.1f}%").classes(f"text-{color}-400 text-sm font-bold")

            # ── 설정 요약 ──
            ui.label(
                f"⚙️ 설정: {int(sl_score.value)}점↑ / Top-{int(sl_topk.value)} / "
                f"보유 {int(sl_hold.value)}일({result['hold_col']}) / "
                f"익절 +{int(sl_target.value)}% / 손절 -{int(sl_stop.value)}% / "
                f"비용 {float(sl_cost.value):.2f}%"
            ).classes("text-xs text-gray-500 mt-4")

    ui.button("▶  시뮬레이션 실행", on_click=run_simulation).props(
        "color=primary size=lg"
    ).classes("w-full mt-2 mb-4")

    # ── 안내 ──
    ui.label(
        "💡 data/ 폴더의 recommend_*.csv 파일이 많을수록 백테스트 정확도가 높아집니다. "
        "현재는 시스템 운영 기간의 추천 데이터를 기반으로 시뮬레이션합니다."
    ).classes("text-xs text-gray-600 mt-2")


def _stat_card(title, value, positive=True):
    color = "text-green-400" if positive else "text-red-400"
    with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-xl"):
        ui.label(title).classes("text-xs text-gray-400")
        ui.label(str(value)).classes(f"text-lg font-bold {color} mt-1")
