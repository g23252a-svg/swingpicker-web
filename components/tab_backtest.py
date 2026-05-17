# -*- coding: utf-8 -*-
"""
tab_backtest.py — 🧪 전략 샌드박스 (간이 백테스트 시뮬레이터)
═══════════════════════════════════════════════════════════
[v22 Step AO+AP] 전면 리팩토링 — 82 → 95점 목표

⚠️ 이것은 간이 백테스트입니다:
- recommend CSV의 사후 수익률 컬럼(ret_Xd_%)을 사용
- 손절/익절 도달 순서, 장중 고가/저가, 실제 동시 보유 자금 제약 미반영
- 더 정밀한 백테스트는 OHLCV 기반 별도 백엔드 필요

개선 사항 (Step AO):
1. ✅ 면책 + 과적합 경고 (Prime 유료 기능 법적 안전)
2. ✅ 프리셋 4종 (보수/균형/공격/단타)
3. ✅ CAGR + Sharpe 추가 메트릭
4. ✅ 즐겨찾기 저장 (app.storage.user 활용)
5. ✅ 사용자 친화 라벨 (ret_10d_% 자동 숨김)
6. ✅ 차트 보기 모드 (자산 성장 / Drawdown / 수익률 분포 / 월별 히트맵)
7. ✅ 거래 내역 CSV 다운로드

추가 개선 (Step AP):
8. ✅ '간이 백테스트' 명시 + 자금/체결 제약 미반영 고지
9. ✅ 슬라이더 + 숫자 입력 동기화 (실제 구현)
10. ✅ 신뢰도 배지 (LOW/MEDIUM/HIGH — 거래 수 기반)

향후 작업 (백엔드 필요):
- OHLCV 기반 target/stop 도달 순서 계산
- 최대 동시 보유 수 제한
- position sizing / Kelly 비중
- walk-forward / out-of-sample 분리
- 즐겨찾기 계정별 저장 (Gist 통합)

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


# ═══════════════════════════════════════════════════════════════
# [v3.9.15e + 10] anomaly 정책 — services.backtest_policy SSOT 사용
# ═══════════════════════════════════════════════════════════════
# 이전엔 tab_backtest 내부 상수였지만, 정책값은 UI보다 전략 검증 영역이라
# services로 분리. v3.9.16 프리셋 비교표 / v3.9.17 강건성 테스트 /
# v3.9.18 Train/Test 분할이 모두 같은 임계값 공유.
#
# Backward compat: 기존 ANOMALY_* / TP_SAT_* 이름을 re-export 형태로 유지.
# 외부 모듈/테스트가 tab_backtest의 이 상수를 직접 import하던 경우에도 동작.
from services.backtest_policy import (
    ANOMALY_TOTAL_RET_ABS,
    ANOMALY_SHARPE_MAX,
    ANOMALY_CAGR_MAX,
    ANOMALY_SHORT_DAYS_RET,
    ANOMALY_SHORT_RET,
    ANOMALY_SHORT_DAYS_CAGR,
    TP_SAT_THRESH_LOW_TARGET,
    TP_SAT_THRESH_MID_TARGET,
    TP_SAT_THRESH_HIGH_TARGET,
    tp_saturation_threshold as _tp_saturation_threshold,
    detect_anomaly_flags as _detect_anomaly_flags,
)

# [v3.9.17] Backward compat re-export — verdict / preset_compare 함수 분리 후
# 기존 import 경로 보존. 외부 모듈/테스트가
#   from components.tab_backtest import _calc_kospi_alpha,
#                                       _derive_strategy_verdict,
#                                       _run_preset_comparison,
#                                       _render_preset_comparison_table
# 등을 사용하던 경우에도 동작.
#
# 순환 import 안전성 (검증됨):
# - backtest_verdict 모듈 top-level: tab_backtest import 없음 (services/* 만 의존)
# - backtest_preset_compare 모듈 top-level: tab_backtest import 없음
#   (PRESETS/_run_backtest는 함수 내부 lazy import)
# 따라서 tab_backtest top-level이 위 두 모듈을 import해도 순환 발생 안 함.
from components.backtest_verdict import (
    _calc_kospi_alpha,
    _derive_strategy_verdict,
)
from components.backtest_preset_compare import (
    _run_preset_comparison,
    _render_preset_comparison_table,
)


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


# ═══════════════════════════════════════════════════
#  [Step AP] 신뢰도 배지 (표본 수 기반)
# ═══════════════════════════════════════════════════
def _get_confidence_level(total_trades: int, trading_days: int) -> dict:
    """[Step AP] 거래 수와 기간 기반 결과 신뢰도 평가.
    
    거래 수 기준:
    - LOW: < 30건 (또는 < 7일)
    - MEDIUM: 30~99건 (또는 7~14일)
    - HIGH: 100건 이상 (또는 15일 이상)
    
    Returns:
        {"level": "LOW/MEDIUM/HIGH", "label", "color", "icon", "message"}
    """
    if total_trades < 30 or trading_days < 7:
        return {
            "level": "LOW",
            "label": "낮음",
            "color": "red",
            "icon": "🚨",
            "message": (
                f"표본 부족 — 거래 {total_trades}건 / {trading_days}일. "
                "결과 과신 금지, 과적합 위험 매우 높음."
            ),
        }
    elif total_trades < 100 or trading_days < 15:
        return {
            "level": "MEDIUM",
            "label": "보통",
            "color": "amber",
            "icon": "⚠️",
            "message": (
                f"표본 보통 — 거래 {total_trades}건 / {trading_days}일. "
                "참고용으로만 사용하고 다른 파라미터 조합도 시도하세요."
            ),
        }
    else:
        return {
            "level": "HIGH",
            "label": "양호",
            "color": "green",
            "icon": "✅",
            "message": (
                f"표본 충분 — 거래 {total_trades}건 / {trading_days}일. "
                "통계적 의미 있는 표본이지만, 여전히 과거 데이터의 한계는 존재."
            ),
        }


def _render_confidence_badge(result: dict):
    """[Step AP] 신뢰도 배지 카드 표시"""
    conf = _get_confidence_level(
        result.get("total_trades", 0),
        result.get("trading_days", 0),
    )
    
    color_map = {
        "red": ("bg-red-900/20", "border-red-500/40", "text-red-300"),
        "amber": ("bg-amber-900/20", "border-amber-500/40", "text-amber-300"),
        "green": ("bg-emerald-900/20", "border-emerald-500/40", "text-emerald-300"),
    }
    bg, border, text_color = color_map.get(conf["color"], color_map["amber"])
    
    with ui.card().classes(
        f"w-full p-3 {bg} border {border} rounded-xl mt-3 mb-1"
    ):
        with ui.row().classes("w-full items-start gap-2"):
            ui.label(conf["icon"]).classes("text-xl")
            with ui.column().classes("flex-1 gap-1"):
                ui.label(
                    f"📊 결과 신뢰도: {conf['label']} ({conf['level']})"
                ).classes(f"text-sm font-bold {text_color}")
                ui.label(conf["message"]).classes(
                    "text-xs text-gray-200 leading-relaxed"
                )


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

# ═══════════════════════════════════════════════════════════════════
# [v3.9.16] 프리셋 비교 — 4프리셋 동시 실행 + 비교표 + 하이라이트
# ═══════════════════════════════════════════════════════════════════
# 평가에서 1순위로 제안한 기능. 사용자가 보수/균형/공격/단타를
# 각각 눌러보는 대신 한 번에 비교 가능.
#
# 설계 원칙:
# - 새 산식/배지 만들지 않음 (scope creep 방지 — v3.9.16b로 분리)
# - 기존 services.backtest_policy SSOT (detect_anomaly_flags,
#   tp_saturation_threshold) 그대로 재사용
# - 기존 _calc_kospi_alpha / _run_backtest 그대로 재사용
# - 추천/매수가/Top3 baseline 변경 0
# ═══════════════════════════════════════════════════════════════════


def _render_disclaimer():
    """[Step AO+AP] 백테스트 한계 + 과적합 경고 (Prime 유료 기능 법적 안전)"""
    
    # [Step AP] 간이 백테스트 명시 (가장 중요 — 오해 방지)
    with ui.card().classes(
        "w-full p-3 bg-blue-900/20 border border-blue-500/40 rounded-xl mb-3"
    ):
        with ui.row().classes("w-full items-start gap-2"):
            ui.label("ℹ️").classes("text-xl")
            with ui.column().classes("flex-1 gap-1"):
                ui.label("간이 백테스트 (Simplified Backtest)").classes(
                    "text-sm font-bold text-blue-300"
                )
                ui.label(
                    "이 샌드박스는 recommend CSV의 사후 수익률 컬럼"
                    "(ret_1d/5d/10d/20d/60d/120d_%)을 사용한 간이 시뮬레이션입니다."
                ).classes("text-xs text-gray-200 leading-relaxed")
                ui.label(
                    "다음 항목은 단순화 또는 미반영되어 있습니다:"
                ).classes("text-xs text-gray-300 mt-1 font-bold")
                for line in [
                    "• 손절/익절 도달 순서 (장중 고가→저가 순서 X)",
                    "• 장중 고가/저가 (시작가 대비 종가 기반 평균 수익률)",
                    "• 실제 동시 보유 자금 제약 (Top-K 균등 배분 가정)",
                    "• 포지션 사이징 (Kelly 등 자본 비중 계산 X)",
                    "• 슬리피지 / 부분 체결 / 거래정지",
                ]:
                    ui.label(line).classes("text-xs text-gray-300")
                ui.label(
                    "💡 더 정밀한 결과는 OHLCV 기반 별도 백테스트 도구가 필요합니다."
                ).classes("text-xs text-blue-200 mt-1")
    
    # 백테스트 한계 (기존 amber)
    with ui.card().classes(
        "w-full p-3 bg-amber-900/20 border border-amber-500/40 rounded-xl mb-3"
    ):
        with ui.row().classes("w-full items-start gap-2"):
            ui.label("⚠️").classes("text-xl")
            with ui.column().classes("flex-1 gap-1"):
                ui.label("백테스트 일반 한계").classes(
                    "text-sm font-bold text-amber-300"
                )
                for line in [
                    "• 과거 데이터는 미래 수익을 보장하지 않습니다",
                    "• 실제 체결가 ≠ 시뮬레이션 가격",
                    "• 생존자 편향 — 상장폐지/거래정지 종목 미반영",
                    "• 수수료/세금/슬리피지는 cost_pct로 단순화",
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
                
                # [Step AP] 슬라이더 + 숫자 입력 동기화
                sl_score = ui.slider(
                    min=40, max=95,
                    value=PRESETS["balanced"]["min_score"],
                    step=5,
                ).classes("w-full")
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("").bind_text_from(
                        sl_score, "value",
                        backward=lambda v: f"최소 점수: {int(v)}점",
                    ).classes("text-xs text-gray-300 flex-1")
                    ui.number(
                        value=PRESETS["balanced"]["min_score"],
                        min=40, max=95, step=5,
                        format="%.0f",
                    ).bind_value(sl_score, "value").classes("w-20").props(
                        "outlined dense"
                    )
                sliders["min_score"] = sl_score
                
                sl_topk = ui.slider(
                    min=3, max=30,
                    value=PRESETS["balanced"]["top_k"],
                    step=1,
                ).classes("w-full")
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("").bind_text_from(
                        sl_topk, "value",
                        backward=lambda v: f"일일 편입 종목 수: {int(v)}개",
                    ).classes("text-xs text-gray-300 flex-1")
                    ui.number(
                        value=PRESETS["balanced"]["top_k"],
                        min=3, max=30, step=1,
                        format="%.0f",
                    ).bind_value(sl_topk, "value").classes("w-20").props(
                        "outlined dense"
                    )
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
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("").bind_text_from(
                        sl_hold, "value",
                        backward=lambda v: f"보유 기간: {_hold_days_label(int(v))}",
                    ).classes("text-xs text-gray-300 flex-1")
                    ui.number(
                        value=PRESETS["balanced"]["hold_days"],
                        min=1, max=60, step=1,
                        format="%.0f",
                    ).bind_value(sl_hold, "value").classes("w-20").props(
                        "outlined dense"
                    )
                sliders["hold_days"] = sl_hold
                
                sl_target = ui.slider(
                    min=2, max=30,
                    value=PRESETS["balanced"]["target_pct"],
                    step=1,
                ).classes("w-full")
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("").bind_text_from(
                        sl_target, "value",
                        backward=lambda v: f"익절선: +{int(v)}%",
                    ).classes("text-xs text-gray-300 flex-1")
                    ui.number(
                        value=PRESETS["balanced"]["target_pct"],
                        min=2, max=30, step=1,
                        format="%.0f",
                    ).bind_value(sl_target, "value").classes("w-20").props(
                        "outlined dense"
                    )
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
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("").bind_text_from(
                        sl_stop, "value",
                        backward=lambda v: f"손절선: -{int(v)}%",
                    ).classes("text-xs text-gray-300 flex-1")
                    ui.number(
                        value=PRESETS["balanced"]["stop_pct"],
                        min=2, max=15, step=1,
                        format="%.0f",
                    ).bind_value(sl_stop, "value").classes("w-20").props(
                        "outlined dense"
                    )
                sliders["stop_pct"] = sl_stop
                
                sl_cost = ui.slider(
                    min=0, max=1.0,
                    value=PRESETS["balanced"]["cost_pct"],
                    step=0.05,
                ).classes("w-full")
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("").bind_text_from(
                        sl_cost, "value",
                        backward=lambda v: f"왕복 거래비용: {float(v):.2f}%",
                    ).classes("text-xs text-gray-300 flex-1")
                    ui.number(
                        value=PRESETS["balanced"]["cost_pct"],
                        min=0, max=1.0, step=0.05,
                        format="%.2f",
                    ).bind_value(sl_cost, "value").classes("w-20").props(
                        "outlined dense"
                    )
                sliders["cost_pct"] = sl_cost
        
        # [Step AP] 자금 제약 안내 (파라미터 패널 하단)
        ui.label(
            "ℹ️ 결과는 날짜별 Top-K 균등 평균이며, 실제 동시 보유 슬롯/현금 제약은 단순화되어 있습니다."
        ).classes("text-[11px] text-gray-500 italic mt-3 leading-relaxed")

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

        # [v3.9.16] 프리셋 비교 — 4개 동시 실행 + 비교표
        compare_btn = ui.button(
            "📊 4프리셋 비교",
        ).props("color=cyan-7 size=lg").classes("flex-1 min-w-[180px]")
        compare_btn.tooltip(
            "보수형/균형형/공격형/단타형 4개를 한 번에 돌려 비교표로 표시. "
            "현재 슬라이더 값은 무시되고 각 프리셋 기본값이 사용됩니다."
        )

        # [v3.9.17b] 강건성 테스트 — 4프리셋 드롭다운 (기준 ±5)
        # 평가 v3.9.17 지적 3: 균형형 고정 → 4프리셋 선택 가능
        with ui.button(
            "🧱 강건성 27조합 ▾",
        ).props("color=indigo-7 size=lg").classes(
            "flex-1 min-w-[200px]"
        ) as robust_btn:
            with ui.menu() as robust_menu:
                ui.menu_item(
                    "🛡️ 보수형 강건성",
                    on_click=lambda: run_robustness("conservative"),
                )
                ui.menu_item(
                    "⚖️ 균형형 강건성 (기본)",
                    on_click=lambda: run_robustness("balanced"),
                )
                ui.menu_item(
                    "🚀 공격형 강건성",
                    on_click=lambda: run_robustness("aggressive"),
                )
                ui.menu_item(
                    "⚡ 단타형 강건성",
                    on_click=lambda: run_robustness("scalping"),
                )
        robust_btn.tooltip(
            "선택한 프리셋 주변 27조합 백테스트 (min_score/top_k/hold_days 각 ±5). "
            "과최적화/lookahead 의심 검증. (10~30초)"
        )

        # [v3.9.18] Train/Test 분할 검증 — 4프리셋 드롭다운
        # 평가 v3.9.17 "lookahead bias 결정타"
        with ui.button(
            "🔬 Train/Test 분할 ▾",
        ).props("color=teal-7 size=lg").classes(
            "flex-1 min-w-[200px]"
        ) as tt_btn:
            with ui.menu():
                ui.menu_item(
                    "🛡️ 보수형 Train/Test",
                    on_click=lambda: run_train_test("conservative"),
                )
                ui.menu_item(
                    "⚖️ 균형형 Train/Test (기본)",
                    on_click=lambda: run_train_test("balanced"),
                )
                ui.menu_item(
                    "🚀 공격형 Train/Test",
                    on_click=lambda: run_train_test("aggressive"),
                )
                ui.menu_item(
                    "⚡ 단타형 Train/Test",
                    on_click=lambda: run_train_test("scalping"),
                )
        tt_btn.tooltip(
            "rec_date 기준 70/30 분할 — Train(과거 70%)으로만 좋은 전략인지, "
            "Test(최근 30%)에서도 살아남는지 검증. (5~10초)"
        )

        # [v3.9.19] 시장 국면별 성과 — 4프리셋 드롭다운
        # 평가 로드맵: Train/Test가 시간 분할이면 국면별은 조건 분할
        with ui.button(
            "🌡️ 시장 국면별 성과 ▾",
        ).props("color=purple-7 size=lg").classes(
            "flex-1 min-w-[200px]"
        ) as regime_btn:
            with ui.menu():
                ui.menu_item(
                    "🛡️ 보수형 국면별",
                    on_click=lambda: run_regime("conservative"),
                )
                ui.menu_item(
                    "⚖️ 균형형 국면별 (기본)",
                    on_click=lambda: run_regime("balanced"),
                )
                ui.menu_item(
                    "🚀 공격형 국면별",
                    on_click=lambda: run_regime("aggressive"),
                )
                ui.menu_item(
                    "⚡ 단타형 국면별",
                    on_click=lambda: run_regime("scalping"),
                )
        regime_btn.tooltip(
            "run_health의 macro_risk(NORMAL/CAUTION/CRITICAL) 기준 — "
            "활황/주의/위험 시장에서도 살아남는지 검증. (5~10초)"
        )

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

    # ─── [v3.9.16] 4프리셋 비교 실행 ───
    async def run_comparison():
        result_container.clear()
        with result_container:
            ui.spinner("dots", size="lg", color="cyan")
            ui.label(
                "4개 프리셋 동시 실행 중... (5~15초)"
            ).classes("text-xs text-gray-400 mt-2")

        from async_helpers import run_sync

        all_recs = await run_sync(_load_recommend_files)
        if all_recs.empty:
            result_container.clear()
            with result_container:
                ui.label(
                    "❌ data/ 폴더에 recommend_*.csv 파일이 없습니다"
                ).classes("text-red-400")
            return

        # 4프리셋 동시 실행 — [v3.9.17] lazy import (순환 방지)
        try:
            from components.backtest_preset_compare import (
                _run_preset_comparison,
                _render_preset_comparison_table,
            )
            results_by_preset = await run_sync(
                lambda: _run_preset_comparison(all_recs)
            )
        except Exception as e:
            _logger.warning(f"[프리셋 비교] 실행 실패: {e}", exc_info=True)
            result_container.clear()
            with result_container:
                ui.label(f"⚠️ 프리셋 비교 실행 실패: {e}").classes(
                    "text-red-400 p-3"
                )
            return

        result_container.clear()
        with result_container:
            _render_preset_comparison_table(results_by_preset)

            # 비교 후 면책 한 줄 (단일 결과와 동일 면책 footer)
            ui.label(
                "※ 백테스트 결과는 과거 데이터 기반입니다. "
                "미래 수익을 보장하지 않습니다. anomaly 표시된 프리셋은 "
                "OHLCV 기반 정밀 검증 필요."
            ).classes("text-[10px] text-gray-500 italic mt-3")

    compare_btn.on_click(run_comparison)

    # ─── [v3.9.17b] 강건성 테스트 실행 — preset key 매개변수 ───
    async def run_robustness(preset_key: str = "balanced"):
        """[v3.9.17b] 평가 지적 3 해결: 4프리셋 선택 가능.

        Args:
            preset_key: "conservative" / "balanced" / "aggressive" / "scalping"
        """
        preset_label_map = {
            "conservative": "🛡️ 보수형",
            "balanced": "⚖️ 균형형",
            "aggressive": "🚀 공격형",
            "scalping": "⚡ 단타형",
        }
        label = preset_label_map.get(preset_key, "균형형")

        result_container.clear()
        with result_container:
            ui.spinner("dots", size="lg", color="indigo")
            ui.label(
                f"{label} 27조합 백테스트 실행 중... (10~30초)"
            ).classes("text-xs text-gray-400 mt-2")

        from async_helpers import run_sync

        all_recs = await run_sync(_load_recommend_files)
        if all_recs.empty:
            result_container.clear()
            with result_container:
                ui.label(
                    "❌ data/ 폴더에 recommend_*.csv 파일이 없습니다"
                ).classes("text-red-400")
            return

        # 강건성 테스트 — [v3.9.17b] services 로직 + components UI 분리
        try:
            from components.backtest_robustness import (
                _run_robustness_test,
                _render_robustness_table,
            )
            robustness_data = await run_sync(
                lambda: _run_robustness_test(all_recs, preset_key)
            )
        except Exception as e:
            _logger.warning(f"[강건성] 실행 실패: {e}", exc_info=True)
            result_container.clear()
            with result_container:
                ui.label(f"⚠️ 강건성 테스트 실행 실패: {e}").classes(
                    "text-red-400 p-3"
                )
            return

        result_container.clear()
        with result_container:
            _render_robustness_table(robustness_data)

            ui.label(
                "※ 강건성 테스트는 과거 데이터 기반입니다. "
                "27조합 모두 좋다고 실전 성과를 보장하지 않습니다. "
                "다음 단계로 v3.9.18 Train/Test 분할 검증 권장."
            ).classes("text-[10px] text-gray-500 italic mt-3")

    # [v3.9.17b] 드롭다운 메뉴가 직접 run_robustness(preset_key)를 호출하므로
    # 별도 on_click 불필요. ui.menu_item의 on_click 람다에서 호출.

    # ─── [v3.9.18] Train/Test 분할 검증 ───
    async def run_train_test(preset_key: str = "balanced"):
        """[v3.9.18] rec_date 70/30 분할 검증.

        Args:
            preset_key: "conservative" / "balanced" / "aggressive" / "scalping"
        """
        preset_label_map = {
            "conservative": "🛡️ 보수형",
            "balanced": "⚖️ 균형형",
            "aggressive": "🚀 공격형",
            "scalping": "⚡ 단타형",
        }
        label = preset_label_map.get(preset_key, "균형형")

        result_container.clear()
        with result_container:
            ui.spinner("dots", size="lg", color="teal")
            ui.label(
                f"{label} Train/Test 분할 검증 중... (5~10초)"
            ).classes("text-xs text-gray-400 mt-2")

        from async_helpers import run_sync

        all_recs = await run_sync(_load_recommend_files)
        if all_recs.empty:
            result_container.clear()
            with result_container:
                ui.label(
                    "❌ data/ 폴더에 recommend_*.csv 파일이 없습니다"
                ).classes("text-red-400")
            return

        try:
            from components.backtest_train_test import (
                _run_train_test_split,
                _render_train_test_result,
            )
            tt_data = await run_sync(
                lambda: _run_train_test_split(all_recs, preset_key)
            )
        except Exception as e:
            _logger.warning(f"[Train/Test] 실행 실패: {e}", exc_info=True)
            result_container.clear()
            with result_container:
                ui.label(f"⚠️ Train/Test 검증 실행 실패: {e}").classes(
                    "text-red-400 p-3"
                )
            return

        result_container.clear()
        with result_container:
            _render_train_test_result(tt_data)

            ui.label(
                "※ Train/Test 분할은 시간 순서를 기준으로 합니다. "
                "Test가 최근 30% 구간에서도 살아남으면 일반화 양호. "
                "다음 단계 v3.9.19 — 시장 국면별 성과 분석."
            ).classes("text-[10px] text-gray-500 italic mt-3")

    # ─── [v3.9.19] 시장 국면별 성과 ───
    async def run_regime(preset_key: str = "balanced"):
        """[v3.9.19] macro_risk 기준 NORMAL/CAUTION/CRITICAL 백테스트.

        Args:
            preset_key: "conservative" / "balanced" / "aggressive" / "scalping"
        """
        preset_label_map = {
            "conservative": "🛡️ 보수형",
            "balanced": "⚖️ 균형형",
            "aggressive": "🚀 공격형",
            "scalping": "⚡ 단타형",
        }
        label = preset_label_map.get(preset_key, "균형형")

        result_container.clear()
        with result_container:
            ui.spinner("dots", size="lg", color="purple")
            ui.label(
                f"{label} 시장 국면별 검증 중... (5~10초)"
            ).classes("text-xs text-gray-400 mt-2")

        from async_helpers import run_sync

        all_recs = await run_sync(_load_recommend_files)
        if all_recs.empty:
            result_container.clear()
            with result_container:
                ui.label(
                    "❌ data/ 폴더에 recommend_*.csv 파일이 없습니다"
                ).classes("text-red-400")
            return

        try:
            from components.backtest_regime import (
                _run_regime_split,
                _render_regime_table,
            )
            regime_data = await run_sync(
                lambda: _run_regime_split(all_recs, preset_key)
            )
        except Exception as e:
            _logger.warning(f"[국면별] 실행 실패: {e}", exc_info=True)
            result_container.clear()
            with result_container:
                ui.label(f"⚠️ 시장 국면별 검증 실행 실패: {e}").classes(
                    "text-red-400 p-3"
                )
            return

        result_container.clear()
        with result_container:
            _render_regime_table(regime_data)

            ui.label(
                "※ 국면 분류는 run_health JSON의 macro_risk 기준입니다. "
                "표본이 부족한 국면은 자동으로 ⚪ 표시됩니다."
            ).classes("text-[10px] text-gray-500 italic mt-3")

    
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
def _render_strategy_verdict_card(result: dict, cfg: dict):
    """[v3.9.15] 전략 판정 카드 — 결과 최상단에 등급 표시.
    
    [v3.9.15b 보정 5건] KOSPI 알파 + 🟢 강화 + logger + cfg + 검증 부족 표현
    [v3.9.15c 보정 4건] 간이 알파 명시 / 벤치 None 차단 / logger / services SSOT
    [v3.9.15d 보정 3건]:
        1. hold_days 슬라이더 ↔ bench key 매핑 (12일 → KOSPI 10일)
        2. services lru_cache (프리셋 비교 시 4회 호출 → 1회 read)
        3. 진짜 일자별 알파 시도 (kospi_daily.csv 있을 때) → 없으면 간이 fallback
    [v3.9.15e 보정 1건 — critical]:
        1. _calc_kospi_alpha의 trades_df 컬럼명 정합화
           "date"→"rec_date", "net_pct"→"net_ret".
           v3.9.15d의 real alpha 경로는 실제 컬럼명과 안 맞아서 항상
           매칭 0건 → simple로만 떨어졌음. 이번 수정으로 daily CSV 추가 시
           실제로 real 경로 활성화.
    [v3.9.15e + 7 보정 3건 — 과대추정 방어]:
        1. 비현실 수익률 anomaly 검사 (total_ret>300 / sharpe>5 / cagr>300)
           → 🟢이라도 자동 🟡 다운그레이드. 간이 백테스트 구조상 lookahead /
             동시 보유 자금 제약 미반영 / TP·SL 도달 순서 미반영으로 인한
             과대평가를 화면에 명시적으로 경고.
        2. TP 포화율 표시 (WIN 비율) — 70%+ 시 경고. recommend CSV의
           ret_NNd_% 사후 수익률을 사용하므로 익절가 도달 추정이 과한 비율이면
           장중 도달 순서 검증 필요.
        3. 자금제약 경고 강화 — 본문에 직접 표시 (5일 보유 Top-5 = 동시 최대
           ~25 슬롯이지만 화면은 매일 Top-K 평균을 독립 복리 누적).
    
    4단계 판정:
      🟢 실전 후보   : 수익≥5% AND 승률≥55 AND MDD≥-15 AND 거래≥100 AND
                       알파≥0 (None 허용 안 함) AND Sharpe≥0.8 AND anomaly 없음
      🟡 관찰 후보   : 수익+ but 위 중 일부 미달 OR 벤치 데이터 없음 OR anomaly 검출
      🟠 검증 부족   : 수익+ but MDD<-20 OR 거래<50 OR Sharpe<0.5
      🔴 실전 부적합 : 수익- OR MDD<-25 OR 알파<0 (시장 열위)
    """
    # ──────────────────────────────────────────────────────────────
    # [v3.9.16b → v3.9.17] 판정 SSOT — backtest_verdict 모듈로 분리됨
    # 비교표와 같은 함수 사용 → 갈라질 가능성 0
    # ──────────────────────────────────────────────────────────────
    from components.backtest_verdict import _derive_strategy_verdict
    verdict = _derive_strategy_verdict(result, cfg)

    # 호환 변수 풀기 (UI 렌더 코드 baseline 그대로 사용 위해)
    icon = verdict["icon"]
    title = verdict["title"]
    color = verdict["color_class"]
    bg = verdict["bg_class"]
    body = verdict["body"]
    is_anomaly = verdict["is_anomaly"]
    anomaly_flags = verdict["anomaly_flags"]
    alpha = verdict["alpha"]
    alpha_mode = verdict["alpha_mode"]
    tp_saturation = verdict["tp_saturation"]
    tp_threshold = verdict["tp_threshold"]
    tp_saturation_warn = verdict["tp_saturation_warn"]

    # 렌더용 보조 변수
    total_ret = float(result.get("total_return", 0) or 0)
    win_rate = float(result.get("win_rate", 0) or 0)
    mdd = float(result.get("mdd", 0) or 0)
    n_trades = int(result.get("total_trades", 0) or 0)
    sharpe = result.get("sharpe")
    sharpe_val = (
        float(sharpe) if sharpe is not None and not pd.isna(sharpe) else None
    )
    # n_win/n_stop/n_hold — anomaly 경고 박스에서 카운트 표시용
    status_dist = result.get("status_dist", {}) or {}
    n_win = int(status_dist.get("WIN", 0) or 0)
    n_stop = int(status_dist.get("STOP", 0) or 0)
    n_hold = int(status_dist.get("HOLD_EXIT", 0) or 0)
    n_total_status = n_win + n_stop + n_hold
    target_pct = float(cfg.get("target_pct", 5) or 5)

    with ui.card().classes(f"w-full p-3 mb-3 {bg} rounded-lg"):
        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.label(icon).classes("text-2xl")
            ui.label(f"전략 판정: {title}").classes(f"text-lg font-bold {color}")
        summary_parts = [
            f"수익률 {total_ret:+.2f}%",
            f"승률 {win_rate:.1f}%",
            f"MDD {mdd:.1f}%",
            f"거래 {n_trades}건",
        ]
        if sharpe_val is not None:
            # [v3.9.15e + 8/9] summary에도 Sharpe cap (상수 사용)
            if sharpe_val > ANOMALY_SHARPE_MAX:
                summary_parts.append(f"Sharpe 비정상 ({ANOMALY_SHARPE_MAX} 초과)")
            else:
                summary_parts.append(f"Sharpe {sharpe_val:.2f}")
        if alpha is not None:
            mode_label = "정확 알파" if alpha_mode == "real" else "간이 알파"
            summary_parts.append(f"{mode_label} {alpha:+.2f}%p")
        else:
            summary_parts.append("KOSPI 알파 데이터 없음")
        # [v3.9.15e + 7 보정 2] TP 포화율 summary에 추가
        if n_total_status > 0:
            summary_parts.append(f"TP 포화율 {tp_saturation:.1f}%")
        ui.label("결과: " + " · ".join(summary_parts)).classes(
            "text-xs text-gray-300 mb-1"
        )
        ui.label(body).classes("text-sm text-gray-200 leading-relaxed")

        # [v3.9.15e + 7 보정 1] anomaly 검출 시 별도 경고 박스
        if is_anomaly:
            with ui.card().classes(
                "w-full p-2 mt-2 bg-orange-900/25 border border-orange-500/50 rounded"
            ):
                ui.label("🚨 결과 과열 경고 — 간이 백테스트 과대추정 가능성").classes(
                    "text-sm font-bold text-orange-300"
                )
                ui.label(
                    "수익률/CAGR/Sharpe 중 하나 이상이 비정상 수준입니다. "
                    "간이 백테스트는 다음을 단순화/미반영합니다:"
                ).classes("text-[11px] text-orange-100 mt-1")
                ui.label(
                    "  · 보유기간 중 동시 보유 슬롯 제약 (5일 보유 Top-5 = 최대 ~25 슬롯)"
                ).classes("text-[11px] text-orange-100")
                ui.label(
                    "  · 매일 Top-K 평균을 독립 복리로 누적 → 실제 자금 제약 시 크게 낮아짐"
                ).classes("text-[11px] text-orange-100")
                ui.label(
                    "  · TP/SL 장중 도달 순서 (시작가 대비 종가 기반 ret_NNd_% 사용)"
                ).classes("text-[11px] text-orange-100")
                ui.label(
                    "→ 실전 수치는 화면 표시값보다 현저히 낮을 가능성이 높습니다. "
                    "OHLCV 기반 정밀 백테스트 + Train/Test 분할 검증을 권장합니다."
                ).classes("text-[11px] text-orange-200 mt-1")

        # [v3.9.15e + 7/9] TP 포화율 임계 초과 시 경고 박스
        if tp_saturation_warn:
            with ui.card().classes(
                "w-full p-2 mt-2 bg-amber-900/20 border border-amber-500/40 rounded"
            ):
                ui.label(
                    f"⚠️ TP 포화율 {tp_saturation:.1f}% — 대부분 익절 캡(+{target_pct:.0f}%)에 도달 "
                    f"(임계 {tp_threshold}%)"
                ).classes("text-xs font-bold text-amber-300")
                tier_note = (
                    f"익절선 +{target_pct:.0f}% tier 임계 {tp_threshold}% 초과."
                    if target_pct >= 10
                    else f"균형 tier 임계 {tp_threshold}% 초과."
                    if target_pct >= 6
                    else f"단타 tier 임계 {tp_threshold}% 초과 — 작은 익절도 80%+면 의심."
                )
                ui.label(
                    f"WIN {n_win}건 / HOLD_EXIT {n_hold}건 / STOP {n_stop}건. {tier_note} "
                    "ret_NNd_% 사후수익률 기반 시뮬이라 실제 장중 도달 순서 검증 필요. "
                    "고가 도달 후 손절가도 같은 날 찍었을 가능성이 시뮬엔 반영 안 됨."
                ).classes("text-[11px] text-amber-100 mt-1")

        good_pts = []
        bad_pts = []
        if total_ret >= 5:
            good_pts.append("수익 양호")
        elif total_ret > 0:
            good_pts.append("수익 양수")
        else:
            bad_pts.append("수익 음수")
        if win_rate >= 55:
            good_pts.append("승률 양호")
        elif win_rate < 50:
            bad_pts.append("승률 낮음")
        if mdd >= -15:
            good_pts.append("낙폭 안정")
        elif mdd < -20:
            bad_pts.append("낙폭 큼")
        if n_trades >= 100:
            good_pts.append("표본 충분")
        elif n_trades < 50:
            bad_pts.append("표본 부족")
        if alpha is not None:
            if alpha >= 0:
                # [v3.9.15e + 7 보정 1] anomaly 검출 시 "시장 초과"는 신뢰 못 함
                if not is_anomaly:
                    good_pts.append("시장 초과")
            else:
                bad_pts.append("시장 열위")
        # [v3.9.15e + 7 보정 2] TP 포화율 매우 높으면 bad_pts에도
        if tp_saturation_warn:
            bad_pts.append(f"TP 포화 {tp_saturation:.0f}%")

        with ui.row().classes("gap-3 mt-2 flex-wrap"):
            if good_pts:
                ui.label(f"✅ 장점: {' · '.join(good_pts)}").classes(
                    "text-[11px] text-emerald-300"
                )
            if bad_pts:
                ui.label(f"⚠️ 주의: {' · '.join(bad_pts)}").classes(
                    "text-[11px] text-amber-300"
                )

        try:
            cfg_summary = (
                f"조건: {cfg.get('min_score', '?')}점↑ / "
                f"Top-{cfg.get('top_k', '?')} / "
                f"보유 {cfg.get('hold_days', '?')}일 / "
                f"+{cfg.get('target_pct', '?')}/{cfg.get('stop_pct', '?')} / "
                f"비용 {cfg.get('cost_pct', '?')}%"
            )
            ui.label(cfg_summary).classes(
                "text-[10px] text-gray-400 mt-2 font-mono"
            )
        except Exception:
            pass

        # [v3.9.15c → v3.9.15d 보정 1] 알파 모드별 동적 footnote
        if alpha_mode == "real":
            note_alpha = (
                "정확 알파: 각 거래 추천일에서 hold_days 동안 KOSPI 수익률을 "
                "차감한 진짜 일자별 알파."
            )
        elif alpha_mode == "simple":
            note_alpha = (
                "간이 알파는 KOSPI 단일 시점 기준 — 거래일별 정확 알파는 "
                "daily KOSPI 데이터 추가 후 제공."
            )
        else:
            note_alpha = "KOSPI 데이터 없음 — 알파 미산정."
        ui.label(
            f"※ 백테스트 결과는 과거 데이터 기반입니다. 미래 수익을 보장하지 않습니다. {note_alpha}"
        ).classes("text-[10px] text-gray-500 italic mt-1")


def _render_results(result: dict, cfg: dict):
    """결과 메트릭 + 차트 모드 + 다운로드"""
    
    # [v3.9.15] 전략 판정 카드 — 결과 최상단에 등급 표시
    # [v3.9.15b 보정 3] 무음 except → logger.warning
    try:
        _render_strategy_verdict_card(result, cfg)
    except Exception as _e:
        _logger.warning(f"전략 판정 카드 렌더 실패: {_e}", exc_info=True)
    
    # [Step AP] 신뢰도 배지 (가장 먼저 — 표본 부족 시 즉시 인지)
    _render_confidence_badge(result)

    
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
                # [v3.9.15e + 8/9] CAGR 표시값 cap — 상수 ANOMALY_CAGR_MAX 사용
                if cagr > ANOMALY_CAGR_MAX:
                    cagr_display = f"비정상 과열 ({ANOMALY_CAGR_MAX}% 초과)"
                    cagr_tooltip = (
                        f"연복리 수익률 — 실제 값 {cagr:+.2f}% "
                        f"(비현실적 — 간이 백테스트 과대추정 가능성)"
                    )
                    cagr_is_positive = False
                else:
                    cagr_display = f"{cagr:+.2f}%"
                    cagr_tooltip = "연복리 수익률 — 투자 기간을 1년으로 환산"
                    cagr_is_positive = cagr >= 0
                _stat_card(
                    "📅 CAGR (연환산)",
                    cagr_display,
                    cagr_is_positive,
                    tooltip=cagr_tooltip,
                )
            if sharpe is not None:
                # [v3.9.15e + 8/9] Sharpe cap — 상수 ANOMALY_SHARPE_MAX 사용
                if sharpe > ANOMALY_SHARPE_MAX:
                    sharpe_display = f"비정상 ({ANOMALY_SHARPE_MAX} 초과)"
                    sharpe_tooltip = (
                        f"샤프 비율 — 실제 값 {sharpe:.2f} "
                        f"(비현실적 — lookahead 의심)"
                    )
                    sharpe_is_good = False
                else:
                    sharpe_display = f"{sharpe:.2f}"
                    sharpe_tooltip = (
                        "샤프 비율 — 위험 대비 수익. "
                        "1.0 이상 양호, 2.0 이상 우수"
                    )
                    sharpe_is_good = sharpe >= 1.0
                _stat_card(
                    "⚡ Sharpe ratio",
                    sharpe_display,
                    sharpe_is_good,
                    tooltip=sharpe_tooltip,
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
