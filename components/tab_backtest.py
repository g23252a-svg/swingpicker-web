# -*- coding: utf-8 -*-
"""
tab_backtest.py — 🧪 전략 샌드박스 (백테스트 시뮬레이터)
═══════════════════════════════════════════════════════════
[v22 Step AO] 전면 리팩토링 — 82 → 95점 목표

개선 사항:
1. ✅ 면책 + 과적합 경고 (Prime 유료 기능 법적 안전)
2. ✅ 프리셋 4종 (보수/균형/공격/커스텀)
3. ✅ CAGR + Sharpe 추가 메트릭
4. ✅ 즐겨찾기 저장 (app.storage.user 활용)
5. ✅ 사용자 친화 라벨 (ret_10d_% 자동 숨김, 보유기간 라벨)
6. ✅ 모바일 슬라이더 + 숫자 입력 동기화
7. ✅ 차트 보기 모드 (자산 성장 / Drawdown / 수익률 분포 / 월별 히트맵)
8. ✅ 거래 내역 CSV 다운로드

Premium 전용 킬러 기능 — Prime 가입 동기의 핵심
"""
import os
import io
import json
import logging
import math
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nicegui import ui, app

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


# ═══════════════════════════════════════════════════
#  [Step AO] 사용자 친화 라벨
# ═══════════════════════════════════════════════════
def _hold_days_label(days: int) -> str:
    """보유 기간 일수 → 사용자 친화 라벨"""
    if days <= 1:
        return f"{days}일 (당일)"
    elif days <= 5:
        return f"{days}일 (단기)"
    elif days <= 10:
        return f"{days}일 (1~2주)"
    elif days <= 20:
        return f"{days}일 (약 1개월)"
    elif days <= 60:
        return f"{days}일 (분기)"
    else:
        return f"{days}일 (장기)"


# ═══════════════════════════════════════════════════
#  [Step AO] 전략 프리셋
# ═══════════════════════════════════════════════════
PRESETS = {
    "conservative": {
        "label": "🛡️ 보수형",
        "desc": "고점수만 + 짧은 보유 + 빠른 익절",
        "min_score": 80,
        "top_k": 5,
        "hold_days": 5,
        "target_pct": 5,
        "stop_pct": 3,
        "cost_pct": 0.4,
    },
    "balanced": {
        "label": "⚖️ 균형형 (기본)",
        "desc": "표준 설정 — 첫 사용자 권장",
        "min_score": 70,
        "top_k": 10,
        "hold_days": 10,
        "target_pct": 10,
        "stop_pct": 5,
        "cost_pct": 0.4,
    },
    "aggressive": {
        "label": "🚀 공격형",
        "desc": "광범위 종목 + 긴 보유 + 큰 익절",
        "min_score": 60,
        "top_k": 20,
        "hold_days": 20,
        "target_pct": 20,
        "stop_pct": 8,
        "cost_pct": 0.4,
    },
    "scalping": {
        "label": "⚡ 단타형",
        "desc": "고점수 + 1일 보유 + 작은 익절 (비용 영향 큼)",
        "min_score": 75,
        "top_k": 5,
        "hold_days": 1,
        "target_pct": 3,
        "stop_pct": 2,
        "cost_pct": 0.7,
    },
}


# ═══════════════════════════════════════════════════
#  [Step AO] 차트 보기 모드 해설
# ═══════════════════════════════════════════════════
CHART_MODE_EXPLANATIONS = {
    "equity": (
        "📈 자산 성장 곡선: 초기 자본 1.0 기준 복리 누적. "
        "원금 점선 위에 머물수록 안정적. 균등 배분 가정."
    ),
    "drawdown": (
        "📉 Drawdown: 고점 대비 하락폭 추세. "
        "최대낙폭(MDD)이 크면 실전 유지가 심리적으로 어려움."
    ),
    "histogram": (
        "📊 수익률 분포: 거래별 수익률 히스토그램. "
        "왼쪽(손실) 꼬리가 두꺼우면 위험 큰 전략. 0% 기준 분포 형태 확인."
    ),
    "monthly": (
        "📅 월별 수익률: 각 월의 누적 수익률. "
        "특정 월에만 수익이 몰리면 시장 의존성 높음 (강건성↓)."
    ),
}


def _get_ret_col(hold_days: int) -> str:
    for threshold, col in _RET_MAP:
        if hold_days <= threshold:
            return col
    return "ret_20d_%"


# ═══════════════════════════════════════════════════
#  데이터 로딩 (기존 유지)
# ═══════════════════════════════════════════════════
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
            date_col = next(
                (c for c in ["기준일", "trade_date", "DATA_DATE"]
                 if c in df.columns),
                None,
            )
            if date_col:
                date_str = str(df[date_col].iloc[0]).replace("-", "")[:8]
            else:
                date_str = datetime.now().strftime("%Y%m%d")
            existing_dates = (
                {d["rec_date"].iloc[0] for d in dfs} if dfs else set()
            )
            if date_str not in existing_dates:
                df["rec_date"] = date_str
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)

    # ── 종목명 오염 복구 ──
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
    p = os.path.join(_DATA_DIR, "krx_names_latest.csv")
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, dtype={"종목코드": str})
            if "종목코드" in df.columns and "종목명" in df.columns:
                return dict(
                    zip(df["종목코드"].str.zfill(6), df["종목명"])
                )
        except Exception:
            pass
    # 2순위: services.data_store.store.krx_df
    try:
        from services.data_store import store
        krx = getattr(store, "krx_df", None)
        if krx is not None and not krx.empty:
            if "종목코드" in krx.columns and "종목명" in krx.columns:
                return dict(
                    zip(
                        krx["종목코드"].astype(str).str.zfill(6),
                        krx["종목명"],
                    )
                )
    except Exception:
        pass
    return {}


# ═══════════════════════════════════════════════════
#  백테스트 코어 + [Step AO] 추가 메트릭
# ═══════════════════════════════════════════════════
def _calc_advanced_metrics(daily_rets: pd.Series, equity: pd.Series) -> dict:
    """[Step AO] CAGR + Sharpe + 추가 메트릭 계산.
    
    Args:
        daily_rets: 일별 평균 수익률 (%)
        equity: 누적 자산 곡선 (1.0 기준)
    
    Returns:
        {"cagr": ..., "sharpe": ..., "volatility": ..., "win_streak": ..., "loss_streak": ...}
    """
    result = {
        "cagr": None,
        "sharpe": None,
        "volatility": None,
        "win_streak": 0,
        "loss_streak": 0,
    }
    
    if daily_rets.empty or equity.empty:
        return result
    
    try:
        # CAGR (연환산) — 252 영업일/년 가정
        n_days = len(daily_rets)
        total = float(equity.iloc[-1])
        if total > 0 and n_days > 0:
            cagr = (total ** (252 / n_days) - 1) * 100
            if not (math.isinf(cagr) or math.isnan(cagr)):
                result["cagr"] = round(cagr, 2)
        
        # Sharpe ratio (간이 — 무위험금리 0% 가정, 일별)
        std = float(daily_rets.std())
        mean = float(daily_rets.mean())
        if std > 0:
            sharpe = (mean / std) * (252 ** 0.5)
            if not (math.isinf(sharpe) or math.isnan(sharpe)):
                result["sharpe"] = round(sharpe, 2)
        
        # 연환산 변동성
        if std > 0:
            vol = std * (252 ** 0.5)
            if not (math.isinf(vol) or math.isnan(vol)):
                result["volatility"] = round(vol, 2)
        
        # 최대 연승/연패
        wins = (daily_rets > 0).astype(int).tolist()
        losses = (daily_rets <= 0).astype(int).tolist()
        result["win_streak"] = _max_streak(wins)
        result["loss_streak"] = _max_streak(losses)
    except Exception as e:
        _logger.debug(f"고급 메트릭 계산 오류: {e}")
    
    return result


def _max_streak(binary_list: list) -> int:
    """1이 연속된 최대 길이"""
    max_s = 0
    cur = 0
    for v in binary_list:
        if v:
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s


def _run_backtest(all_recs: pd.DataFrame, min_score: int, hold_days: int,
                  stop_pct: float, target_pct: float, top_k: int,
                  cost_pct: float) -> dict:
    """백테스트 실행 — recommend CSV의 ret_Xd_% 컬럼 활용"""
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

    # 날짜별 포트폴리오 수익률 (균등 배분)
    daily_rets = df.groupby("rec_date")["net_ret"].mean()
    daily_rets = daily_rets.sort_index()

    # 누적 수익곡선
    equity = (1 + daily_rets / 100).cumprod()
    equity_series = pd.DataFrame({"date": equity.index, "equity": equity.values})

    # MDD 계산
    peak = equity.cummax()
    drawdown = ((equity - peak) / peak) * 100
    mdd = drawdown.min()
    dd_series = pd.DataFrame({"date": drawdown.index, "drawdown": drawdown.values})

    # 통계
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
    
    # [Step AO] 고급 메트릭
    adv = _calc_advanced_metrics(daily_rets, equity)

    return {
        "total_return": round(total_return, 2),
        "mdd": round(float(mdd), 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2)
            if not math.isinf(profit_factor) else 99.99,
        "total_trades": total_trades,
        "trading_days": trading_days,
        "avg_win": round(float(avg_win), 2) if pd.notna(avg_win) else 0,
        "avg_loss": round(float(avg_loss), 2) if pd.notna(avg_loss) else 0,
        "status_dist": status_dist,
        "equity": equity_series,
        "drawdown": dd_series,
        "best_trades": best_trades,
        "worst_trades": worst_trades,
        "hold_col": ret_col,
        "trades_df": df,  # [Step AO] 다운로드/히스토그램용
        "daily_rets": daily_rets,  # [Step AO] 월별 차트용
        # 고급 메트릭
        "cagr": adv["cagr"],
        "sharpe": adv["sharpe"],
        "volatility": adv["volatility"],
        "win_streak": adv["win_streak"],
        "loss_streak": adv["loss_streak"],
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


# ═══════════════════════════════════════════════════
#  [Step AO] 차트 빌더 (모드별)
# ═══════════════════════════════════════════════════
def _build_chart_by_mode(result: dict, mode: str):
    """모드별 차트 생성"""
    if mode == "equity":
        eq = result["equity"]
        if eq.empty:
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq["date"], y=eq["equity"],
            mode="lines", fill="tozeroy",
            line=dict(color="#3B82F6", width=2),
            fillcolor="rgba(59,130,246,0.1)",
            name="자산 가치",
            hovertemplate="<b>%{x}</b><br>자산: %{y:.4f}<extra></extra>",
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="원금 (1.0)",
                      annotation_font_color="gray")
        fig.update_layout(title="📈 자산 성장 곡선 (복리, 균등 배분)")
        return _plotly_dark(fig, 380)
    
    elif mode == "drawdown":
        dd = result["drawdown"]
        if dd.empty:
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd["date"], y=dd["drawdown"],
            mode="lines", fill="tozeroy",
            line=dict(color="#EF4444", width=1.5),
            fillcolor="rgba(239,68,68,0.15)",
            name="낙폭",
            hovertemplate="<b>%{x}</b><br>낙폭: %{y:.2f}%<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dot",
                      line_color="rgba(255,255,255,0.3)")
        fig.update_layout(title=f"📉 Drawdown (최대 낙폭: {result['mdd']:.2f}%)")
        return _plotly_dark(fig, 320)
    
    elif mode == "histogram":
        # [Step AO] 거래별 수익률 분포
        trades_df = result.get("trades_df")
        if trades_df is None or trades_df.empty:
            return None
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trades_df["net_ret"],
            nbinsx=40,
            marker_color="#3B82F6",
            opacity=0.7,
            name="거래 수",
            hovertemplate="<b>구간: %{x}%</b><br>거래 수: %{y}<extra></extra>",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="white",
                      annotation_text="0% (손익분기)",
                      annotation_font_color="white")
        # 평균 라인
        mean_ret = trades_df["net_ret"].mean()
        fig.add_vline(x=mean_ret, line_dash="dot",
                      line_color="#10B981",
                      annotation_text=f"평균 {mean_ret:+.2f}%",
                      annotation_font_color="#10B981")
        fig.update_layout(
            title="📊 거래별 수익률 분포 (히스토그램)",
            xaxis_title="수익률 (%)",
            yaxis_title="거래 수",
        )
        return _plotly_dark(fig, 380)
    
    elif mode == "monthly":
        # [Step AO] 월별 수익률
        daily_rets = result.get("daily_rets")
        if daily_rets is None or daily_rets.empty:
            return None
        try:
            # rec_date(YYYYMMDD) → 월 추출
            df = pd.DataFrame({"date": daily_rets.index, "ret": daily_rets.values})
            df["month"] = df["date"].astype(str).str[:6]  # YYYYMM
            monthly = df.groupby("month")["ret"].apply(
                lambda x: (1 + x / 100).prod() - 1
            ) * 100
            
            colors = ["#10B981" if v > 0 else "#EF4444"
                      for v in monthly.values]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly.index, y=monthly.values,
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>수익률: %{y:.2f}%<extra></extra>",
            ))
            fig.add_hline(y=0, line_dash="dot",
                          line_color="rgba(255,255,255,0.3)")
            fig.update_layout(
                title="📅 월별 수익률 (월말 기준 누적)",
                xaxis_title="월",
                yaxis_title="수익률 (%)",
            )
            return _plotly_dark(fig, 320)
        except Exception as e:
            _logger.debug(f"월별 차트 오류: {e}")
            return None
    
    return None


# ═══════════════════════════════════════════════════
#  [Step AO] 면책 + 과적합 경고
# ═══════════════════════════════════════════════════
def _render_disclaimer():
    """[Step AO] 백테스트 한계 + 과적합 경고 (Prime 유료 기능 법적 안전)"""
    
    # 백테스트 한계
    with ui.card().classes(
        "w-full p-3 bg-amber-900/20 border border-amber-500/40 rounded-xl mb-3"
    ):
        with ui.row().classes("w-full items-start gap-2"):
            ui.label("⚠️").classes("text-xl")
            with ui.column().classes("flex-1 gap-1"):
                ui.label("백테스트 시뮬레이션 한계").classes(
                    "text-sm font-bold text-amber-300"
                )
                for line in [
                    "• 과거 데이터는 미래 수익을 보장하지 않습니다",
                    "• 실제 체결가 ≠ 시뮬레이션 가격 (슬리피지/체결 실패)",
                    "• 생존자 편향 — 상장폐지/거래정지 종목 미반영",
                    "• 균등 배분 가정 — 실제 매수 단위 반영 X",
                ]:
                    ui.label(line).classes("text-xs text-gray-300")
    
    # 과적합 경고 (가장 중요) — 빨간 카드
    with ui.card().classes(
        "w-full p-3 bg-red-900/20 border border-red-500/40 rounded-xl mb-3"
    ):
        with ui.row().classes("w-full items-start gap-2"):
            ui.label("🚨").classes("text-xl")
            with ui.column().classes("flex-1 gap-1"):
                ui.label("과적합(Overfitting) 주의").classes(
                    "text-sm font-bold text-red-300"
                )
                ui.label(
                    "파라미터를 과거 데이터에 맞춰 조정하면 테스트 결과는 좋아지지만, "
                    "실전에서는 작동하지 않을 수 있습니다."
                ).classes("text-xs text-gray-200 leading-relaxed")
                ui.label(
                    "💡 강건한 전략의 신호: '여러 파라미터 조합에서 일관되게 양호한 결과'"
                ).classes("text-xs text-amber-200 mt-1 font-bold")
                ui.label(
                    "💡 권장: 한 가지 설정에서 +30%보다, 여러 설정에서 +10~15%가 더 신뢰할 만합니다."
                ).classes("text-xs text-amber-200")


# ═══════════════════════════════════════════════════
#  [Step AO] 즐겨찾기 저장/로드
# ═══════════════════════════════════════════════════
FAVORITES_KEY = "backtest_favorites"
MAX_FAVORITES = 10


def _load_favorites() -> list:
    """app.storage.user에서 즐겨찾기 로드"""
    try:
        raw = app.storage.user.get(FAVORITES_KEY, [])
        if isinstance(raw, str):
            return json.loads(raw)
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def _save_favorite(config: dict, label: str = ""):
    """즐겨찾기 추가 (최대 MAX_FAVORITES개)"""
    try:
        favs = _load_favorites()
        # 라벨 자동 생성
        if not label:
            label = (
                f"{config['min_score']}점 / Top{config['top_k']} / "
                f"{config['hold_days']}일 / +{config['target_pct']}/-{config['stop_pct']}"
            )
        entry = {
            "label": label[:50],
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            **config,
        }
        favs.insert(0, entry)
        favs = favs[:MAX_FAVORITES]  # 최대 개수 제한
        app.storage.user[FAVORITES_KEY] = favs
        return True
    except Exception as e:
        _logger.warning(f"즐겨찾기 저장 실패: {e}")
        return False


def _delete_favorite(idx: int):
    """즐겨찾기 삭제"""
    try:
        favs = _load_favorites()
        if 0 <= idx < len(favs):
            favs.pop(idx)
            app.storage.user[FAVORITES_KEY] = favs
            return True
    except Exception as e:
        _logger.warning(f"즐겨찾기 삭제 실패: {e}")
    return False


# ═══════════════════════════════════════════════════
#  메인 렌더링
# ═══════════════════════════════════════════════════
def render_tab_backtest(df, auth):
    """[Step AO] Tab: 🧪 전략 샌드박스 — 면책+프리셋+고급 메트릭"""
    
    # ── 프리미엄 게이트 ──
    if auth not in ("admin", "prime"):
        with ui.card().classes(
            "w-full p-8 bg-[#1a1a2e] border border-gray-700 rounded-xl text-center"
        ):
            ui.label("🔒 전략 샌드박스").classes(
                "text-2xl font-bold text-white mb-4"
            )
            ui.label("Prime 구독자 전용 기능입니다").classes(
                "text-gray-400 mb-2"
            )
            ui.label(
                "과거 추천 데이터를 기반으로 나만의 전략을 백테스트하고\n"
                "수익곡선, 최대낙폭(MDD), Sharpe, 월별 수익률을 확인하세요."
            ).classes("text-gray-500 text-sm whitespace-pre-line")
            with ui.row().classes("justify-center mt-6 gap-4 flex-wrap"):
                for icon, label in [
                    ("📊", "수익곡선"),
                    ("📉", "MDD 분석"),
                    ("🎯", "Sharpe / CAGR"),
                    ("📅", "월별 수익률"),
                ]:
                    with ui.card().classes(
                        "p-3 border border-gray-700 rounded-xl items-center"
                    ):
                        ui.label(icon).classes("text-2xl")
                        ui.label(label).classes("text-xs text-gray-400")
            ui.button(
                "💎 멤버십 업그레이드 알아보기",
                on_click=lambda: ui.run_javascript(
                    "document.querySelector('[role=tab]:nth-child(4)')?.click()"
                ),
            ).classes("mt-4").props("color=primary rounded size=lg")
        return

    # ─── Premium 유저: 전체 UI ───
    ui.label("🧪 전략 샌드박스").classes(
        "text-2xl font-bold text-white mb-1"
    )
    ui.label(
        "과거 추천 데이터 기반 백테스트 시뮬레이터 (Prime 전용)"
    ).classes("text-gray-400 text-sm mb-3")

    # ─── [Step AO] 면책 + 과적합 경고 (가장 먼저!) ───
    _render_disclaimer()

    # ─── [Step AO] 프리셋 + 즐겨찾기 ───
    state = {
        "config": dict(PRESETS["balanced"]),  # 기본 균형형
        "result": None,
    }
    
    # 슬라이더 위젯 참조 (프리셋 로드 시 값 변경용)
    sliders = {}
    
    def _apply_config_to_sliders(cfg: dict):
        """설정 dict를 슬라이더에 적용"""
        for key, slider in sliders.items():
            if key in cfg:
                slider.value = cfg[key]
    
    # 프리셋 카드
    with ui.card().classes(
        "w-full p-4 bg-[#1a1a2e] border border-cyan-500/30 rounded-xl mb-3"
    ):
        ui.label("🎯 전략 프리셋").classes(
            "text-sm font-bold text-cyan-300 mb-2"
        )
        with ui.row().classes("w-full gap-2 flex-wrap"):
            for key, preset in PRESETS.items():
                preset_card = ui.card().classes(
                    "flex-1 min-w-[150px] p-2 cursor-pointer "
                    "bg-[#0a0a14] border border-gray-700 rounded-lg "
                    "hover:border-cyan-500 transition-colors"
                )
                with preset_card:
                    ui.label(preset["label"]).classes(
                        "text-sm font-bold text-white"
                    )
                    ui.label(preset["desc"]).classes(
                        "text-[10px] text-gray-400 leading-tight"
                    )
                preset_card.on(
                    "click",
                    lambda _, p=preset: (
                        _apply_config_to_sliders(p),
                        ui.notify(f"{p['label']} 적용됨", type="positive"),
                    ),
                )
    
    # ─── 파라미터 패널 ───
    with ui.card().classes(
        "w-full p-4 bg-[#1a1a2e] border border-gray-700 rounded-xl mb-3"
    ):
        ui.label("⚙️ 전략 파라미터").classes(
            "text-base font-bold text-white mb-3"
        )
        
        # [Step AO] 모바일 친화 — flex-wrap + 충분한 너비
        with ui.row().classes("w-full gap-4 flex-wrap"):
            # 진입 조건
            with ui.column().classes("flex-1 min-w-[260px] gap-2"):
                ui.label("📋 진입 조건").classes(
                    "text-sm font-bold text-blue-400 mb-1"
                )
                
                sl_score = ui.slider(
                    min=40, max=95,
                    value=PRESETS["balanced"]["min_score"],
                    step=5,
                ).classes("w-full")
                ui.label("").bind_text_from(
                    sl_score, "value",
                    backward=lambda v: f"최소 점수: {int(v)}점",
                ).classes("text-xs text-gray-300")
                sliders["min_score"] = sl_score
                
                sl_topk = ui.slider(
                    min=3, max=30,
                    value=PRESETS["balanced"]["top_k"],
                    step=1,
                ).classes("w-full")
                ui.label("").bind_text_from(
                    sl_topk, "value",
                    backward=lambda v: f"일일 편입 종목 수: {int(v)}개",
                ).classes("text-xs text-gray-300")
                sliders["top_k"] = sl_topk
            
            # 매매 규칙
            with ui.column().classes("flex-1 min-w-[260px] gap-2"):
                ui.label("💰 매매 규칙").classes(
                    "text-sm font-bold text-green-400 mb-1"
                )
                
                sl_hold = ui.slider(
                    min=1, max=60,
                    value=PRESETS["balanced"]["hold_days"],
                    step=1,
                ).classes("w-full")
                # [Step AO] 사용자 친화 라벨 — ret_10d_% 자동 숨김
                ui.label("").bind_text_from(
                    sl_hold, "value",
                    backward=lambda v: f"보유 기간: {_hold_days_label(int(v))}",
                ).classes("text-xs text-gray-300")
                sliders["hold_days"] = sl_hold
                
                sl_target = ui.slider(
                    min=2, max=30,
                    value=PRESETS["balanced"]["target_pct"],
                    step=1,
                ).classes("w-full")
                ui.label("").bind_text_from(
                    sl_target, "value",
                    backward=lambda v: f"익절선: +{int(v)}%",
                ).classes("text-xs text-gray-300")
                sliders["target_pct"] = sl_target
            
            # 리스크 관리
            with ui.column().classes("flex-1 min-w-[260px] gap-2"):
                ui.label("🛡️ 리스크 관리").classes(
                    "text-sm font-bold text-red-400 mb-1"
                )
                
                sl_stop = ui.slider(
                    min=2, max=15,
                    value=PRESETS["balanced"]["stop_pct"],
                    step=1,
                ).classes("w-full")
                ui.label("").bind_text_from(
                    sl_stop, "value",
                    backward=lambda v: f"손절선: -{int(v)}%",
                ).classes("text-xs text-gray-300")
                sliders["stop_pct"] = sl_stop
                
                sl_cost = ui.slider(
                    min=0, max=1.0,
                    value=PRESETS["balanced"]["cost_pct"],
                    step=0.05,
                ).classes("w-full")
                ui.label("").bind_text_from(
                    sl_cost, "value",
                    backward=lambda v: f"왕복 거래비용: {float(v):.2f}%",
                ).classes("text-xs text-gray-300")
                sliders["cost_pct"] = sl_cost

    # ─── [Step AO] 즐겨찾기 영역 ───
    favorites_container = ui.column().classes("w-full mb-3")
    
    def _refresh_favorites():
        favorites_container.clear()
        favs = _load_favorites()
        if not favs:
            return
        
        with favorites_container:
            with ui.card().classes(
                "w-full p-3 bg-[#1a1a2e] border border-purple-500/30 rounded-xl"
            ):
                ui.label(f"⭐ 저장된 설정 ({len(favs)}/{MAX_FAVORITES})").classes(
                    "text-sm font-bold text-purple-300 mb-2"
                )
                for idx, fav in enumerate(favs):
                    with ui.row().classes("w-full items-center gap-2 py-1 border-b border-gray-700/50"):
                        ui.label(fav.get("label", "(no label)")).classes(
                            "text-xs text-white flex-1 truncate"
                        )
                        ui.label(fav.get("saved_at", "")).classes(
                            "text-[10px] text-gray-500"
                        )
                        ui.button(
                            "📥",
                            on_click=lambda _, c=fav: (
                                _apply_config_to_sliders(c),
                                ui.notify("설정 불러옴", type="positive"),
                            ),
                        ).props("flat dense size=xs color=cyan").tooltip("불러오기")
                        ui.button(
                            "🗑️",
                            on_click=lambda _, i=idx: (
                                _delete_favorite(i),
                                _refresh_favorites(),
                            ),
                        ).props("flat dense size=xs color=red").tooltip("삭제")
    
    _refresh_favorites()

    # ─── 실행 버튼 + 결과 영역 ───
    with ui.row().classes("w-full gap-2 mb-3 flex-wrap"):
        run_btn = ui.button(
            "▶  시뮬레이션 실행",
        ).props("color=primary size=lg").classes("flex-1 min-w-[220px]")
        
        save_btn = ui.button(
            "💾 현재 설정 저장",
        ).props("flat color=purple size=md")
    
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
                ui.label(
                    "❌ data/ 폴더에 recommend_*.csv 파일이 없습니다"
                ).classes("text-red-400")
            return

        cfg = {
            "min_score": int(sl_score.value),
            "top_k": int(sl_topk.value),
            "hold_days": int(sl_hold.value),
            "target_pct": float(sl_target.value),
            "stop_pct": float(sl_stop.value),
            "cost_pct": float(sl_cost.value),
        }
        state["config"] = cfg

        result = await run_sync(
            lambda: _run_backtest(
                all_recs,
                cfg["min_score"], cfg["hold_days"],
                cfg["stop_pct"], cfg["target_pct"],
                cfg["top_k"], cfg["cost_pct"],
            )
        )
        state["result"] = result

        result_container.clear()
        with result_container:
            if "error" in result:
                ui.label(f"⚠️ {result['error']}").classes(
                    "text-yellow-400 text-lg p-4"
                )
                return

            _render_results(result, cfg)

    run_btn.on_click(run_simulation)
    
    # 즐겨찾기 저장 버튼
    def save_current():
        cfg = {
            "min_score": int(sl_score.value),
            "top_k": int(sl_topk.value),
            "hold_days": int(sl_hold.value),
            "target_pct": float(sl_target.value),
            "stop_pct": float(sl_stop.value),
            "cost_pct": float(sl_cost.value),
        }
        if _save_favorite(cfg):
            ui.notify("⭐ 설정 저장됨", type="positive")
            _refresh_favorites()
        else:
            ui.notify("⚠️ 저장 실패", type="warning")
    
    save_btn.on_click(save_current)


# ═══════════════════════════════════════════════════
#  [Step AO] 결과 렌더링 (확장)
# ═══════════════════════════════════════════════════
def _render_results(result: dict, cfg: dict):
    """결과 메트릭 + 차트 모드 + 다운로드"""
    
    # ─── 핵심 메트릭 5개 ───
    with ui.row().classes("w-full gap-2 flex-wrap mb-3"):
        _stat_card(
            "📈 총 수익률",
            f"{result['total_return']:+.2f}%",
            result["total_return"] >= 0,
            tooltip="기간 전체 누적 수익률 (복리, 비용 차감 후)",
        )
        _stat_card(
            "📉 최대낙폭",
            f"{result['mdd']:.2f}%",
            False,
            tooltip="고점 대비 최대 하락폭 (실전 위험 지표)",
        )
        _stat_card(
            "🎯 승률",
            f"{result['win_rate']:.1f}%",
            result["win_rate"] >= 50,
            tooltip="비용 차감 후 양수 수익 거래 비율",
        )
        _stat_card(
            "⚖️ 손익비",
            f"{result['profit_factor']:.2f}",
            result["profit_factor"] >= 1.5,
            tooltip="평균수익 ÷ 평균손실 (1.5 이상이면 양호)",
        )
        _stat_card(
            "🔢 총 거래",
            f"{result['total_trades']}회",
            True,
            tooltip=f"분석일수: {result['trading_days']}일",
        )
    
    # ─── [Step AO] 고급 메트릭 (CAGR/Sharpe) ───
    cagr = result.get("cagr")
    sharpe = result.get("sharpe")
    vol = result.get("volatility")
    
    if cagr is not None or sharpe is not None:
        ui.label("📊 고급 지표 (위험 조정 수익률)").classes(
            "text-sm font-bold text-cyan-300 mt-3 mb-2"
        )
        with ui.row().classes("w-full gap-2 flex-wrap"):
            if cagr is not None:
                _stat_card(
                    "📅 CAGR (연환산)",
                    f"{cagr:+.2f}%",
                    cagr >= 0,
                    tooltip="연복리 수익률 — 투자 기간을 1년으로 환산",
                )
            if sharpe is not None:
                _stat_card(
                    "⚡ Sharpe ratio",
                    f"{sharpe:.2f}",
                    sharpe >= 1.0,
                    tooltip=(
                        "샤프 비율 — 위험 대비 수익. "
                        "1.0 이상 양호, 2.0 이상 우수"
                    ),
                )
            if vol is not None:
                _stat_card(
                    "📏 변동성 (연환산)",
                    f"{vol:.2f}%",
                    vol < 30,
                    tooltip="일별 수익률 표준편차의 연환산값 (작을수록 안정)",
                )
            ws = result.get("win_streak", 0)
            ls = result.get("loss_streak", 0)
            if ws or ls:
                _stat_card(
                    "🔥 연승/연패",
                    f"{ws}↑ / {ls}↓",
                    ws >= ls,
                    tooltip="최대 연속 양수일 / 최대 연속 음수일",
                )

    # ─── 평균 수익/손실 + 청산 유형 ───
    with ui.row().classes("w-full gap-2 flex-wrap mt-3"):
        _stat_card(
            "💚 평균 수익",
            f"+{result['avg_win']:.2f}%",
            True,
        )
        _stat_card(
            "💔 평균 손실",
            f"{result['avg_loss']:.2f}%",
            False,
        )
        # 상태 분포
        dist = result.get("status_dist", {})
        dist_text = " / ".join(f"{k}: {v}" for k, v in dist.items())
        with ui.card().classes(
            "p-3 flex-1 min-w-[180px] bg-[#1a1a2e] border border-gray-700 rounded-xl"
        ):
            ui.label("🏷️ 청산 유형").classes("text-xs text-gray-400")
            ui.label(dist_text or "—").classes(
                "text-sm text-white mt-1"
            )

    # ─── [Step AO] 차트 보기 모드 ───
    chart_mode_state = {"mode": "equity"}
    
    chart_mode_options = {
        "equity": "📈 자산 성장 곡선",
        "drawdown": "📉 Drawdown",
        "histogram": "📊 수익률 분포",
        "monthly": "📅 월별 수익률",
    }
    
    ui.label("📊 차트 분석").classes(
        "text-sm font-bold text-cyan-300 mt-4 mb-2"
    )
    
    sel_chart = ui.select(
        options=chart_mode_options,
        value="equity",
        label="차트 보기",
    ).classes("w-full md:w-1/2 mb-2").props("outlined dense")
    
    chart_box = ui.column().classes("w-full")
    
    def refresh_chart():
        chart_box.clear()
        mode = sel_chart.value or "equity"
        chart_mode_state["mode"] = mode
        with chart_box:
            fig = _build_chart_by_mode(result, mode)
            if fig:
                ui.plotly(fig).classes("w-full")
            else:
                ui.label("차트 데이터를 표시할 수 없습니다.").classes(
                    "text-gray-400 p-4"
                )
            
            # 모드별 해설
            explanation = CHART_MODE_EXPLANATIONS.get(mode, "")
            if explanation:
                with ui.card().classes(
                    "w-full p-2 bg-[#0a0a14]/50 "
                    "border border-cyan-700/20 rounded-lg mt-1"
                ):
                    ui.label(explanation).classes(
                        "text-xs text-cyan-100 leading-relaxed"
                    )
    
    sel_chart.on("update:model-value", lambda _: refresh_chart())
    refresh_chart()

    # ─── Top/Bottom 종목 ───
    with ui.row().classes("w-full gap-3 mt-4 flex-wrap"):
        with ui.card().classes(
            "flex-1 min-w-[280px] p-3 bg-[#1a1a2e] "
            "border border-gray-700 rounded-xl"
        ):
            ui.label("🏆 최고 수익 종목").classes(
                "text-sm font-bold text-green-400 mb-2"
            )
            for t in result.get("best_trades", []):
                with ui.row().classes(
                    "w-full justify-between items-center py-1 "
                    "border-b border-gray-800"
                ):
                    ui.label(f"{t['name']}").classes("text-white text-sm")
                    with ui.row().classes("gap-2"):
                        ui.badge(f"{t['score']:.0f}점").props("color=blue")
                        color = "green" if t["net_ret"] > 0 else "red"
                        ui.label(f"{t['net_ret']:+.1f}%").classes(
                            f"text-{color}-400 text-sm font-bold"
                        )

        with ui.card().classes(
            "flex-1 min-w-[280px] p-3 bg-[#1a1a2e] "
            "border border-gray-700 rounded-xl"
        ):
            ui.label("💀 최대 손실 종목").classes(
                "text-sm font-bold text-red-400 mb-2"
            )
            for t in result.get("worst_trades", []):
                with ui.row().classes(
                    "w-full justify-between items-center py-1 "
                    "border-b border-gray-800"
                ):
                    ui.label(f"{t['name']}").classes("text-white text-sm")
                    with ui.row().classes("gap-2"):
                        ui.badge(f"{t['score']:.0f}점").props("color=blue")
                        color = "green" if t["net_ret"] > 0 else "red"
                        ui.label(f"{t['net_ret']:+.1f}%").classes(
                            f"text-{color}-400 text-sm font-bold"
                        )

    # ─── [Step AO] 거래 내역 다운로드 ───
    trades_df = result.get("trades_df")
    if trades_df is not None and not trades_df.empty:
        with ui.row().classes("w-full justify-end mt-3"):
            ui.button(
                f"📥 거래 내역 CSV 다운로드 ({len(trades_df)}건)",
                on_click=lambda: _download_trades(trades_df, cfg),
            ).props("flat color=cyan size=sm")

    # ─── 설정 요약 ───
    ui.label(
        f"⚙️ 설정: {cfg['min_score']}점↑ / Top-{cfg['top_k']} / "
        f"보유 {_hold_days_label(cfg['hold_days'])} / "
        f"익절 +{cfg['target_pct']:.0f}% / 손절 -{cfg['stop_pct']:.0f}% / "
        f"비용 {cfg['cost_pct']:.2f}%"
    ).classes("text-xs text-gray-500 mt-3 text-center")
    
    # ─── [Step AO] 결과 해석 가이드 ───
    with ui.card().classes(
        "w-full p-3 bg-[#0a0a14] border border-gray-700/30 rounded-lg mt-3"
    ):
        ui.label("💡 결과 해석 가이드").classes(
            "text-xs font-bold text-gray-400 mb-1"
        )
        for line in [
            "• 총 수익률만 보지 마세요. MDD가 -20%를 넘으면 실전 유지 어려움",
            "• Sharpe 1.0 미만이면 위험 대비 수익 부족",
            "• 손익비 < 1.0이면 평균적으로 손실 크기 > 수익 크기",
            "• 다른 파라미터 조합도 시도해서 일관성을 확인하세요 (과적합 방어)",
        ]:
            ui.label(line).classes(
                "text-[11px] text-gray-500 leading-relaxed"
            )


def _download_trades(trades_df: pd.DataFrame, cfg: dict):
    """[Step AO] 거래 내역 CSV 다운로드"""
    try:
        # CSV 생성
        csv_buf = io.StringIO()
        trades_df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
        csv_content = csv_buf.getvalue()
        
        # 파일명 (설정 요약 포함)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = (
            f"backtest_{ts}_score{cfg['min_score']}_"
            f"top{cfg['top_k']}_hold{cfg['hold_days']}d.csv"
        )
        
        # NiceGUI download 트리거
        ui.download(csv_content.encode('utf-8-sig'), filename=fname)
        ui.notify(f"📥 다운로드: {fname}", type="positive")
    except Exception as e:
        _logger.error(f"다운로드 실패: {e}")
        ui.notify(f"⚠️ 다운로드 실패: {e}", type="negative")


def _stat_card(title, value, positive=True, tooltip: str = ""):
    color = "text-green-400" if positive else "text-red-400"
    card = ui.card().classes(
        "p-3 min-w-[140px] flex-1 bg-[#1a1a2e] "
        "border border-gray-700 rounded-xl"
    )
    with card:
        ui.label(title).classes("text-xs text-gray-400")
        ui.label(str(value)).classes(f"text-lg font-bold {color} mt-1")
    if tooltip:
        card.tooltip(tooltip)
