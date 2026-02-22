# -*- coding: utf-8 -*-
"""
auto_backtest.py — 자동 백테스트 피드백 루프
═══════════════════════════════════════════════════
[v14] P3 #13: 추천 → 실현수익률 → 점수대별 승률 테이블 → 켈리 반영

핵심 원칙 (6개 안전장치):
  1. 성과 확정 조건: rec_date <= today - horizon_bdays (미확정 제외)
  2. 진입/청산 규칙 고정: 다음날 시가 진입, N일 후 종가 청산(or SL/TP)
  3. binning 고정: FINAL_SCORE 10점 단위 구간
  4. min_n + 스무딩: min_n=30, 라플라스(wins+1)/(n+2)
  5. 켈리 제한: fractional(0.25) + cap(0.10) + 표본 부족→0
  6. 비용 반영: 수수료+세금 편도 0.33% (왕복 0.66%)

사용법:
  collector.main() 끝에 auto_calibrate() 1줄 추가.
"""
import os
import json
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from glob import glob

import numpy as np
import pandas as pd

from collector_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  1. Config (collector_config 연동)
# ═══════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """백테스트 피드백 루프 설정"""
    # ── 성과 측정 ──
    horizon_bdays: int = 5           # 성과 확정 기간 (영업일)
    entry_rule: str = "next_open"    # 진입: 추천 다음날 시가
    exit_rule: str = "horizon_close" # 청산: N일 후 종가 (or SL/TP hit)

    # ── 비용 ──
    fee_oneway_pct: float = 0.015    # 수수료 편도 0.015%
    tax_sell_pct: float = 0.18       # 매도세 0.18% (코스피) — 2026 기준
    slippage_pct: float = 0.10       # 슬리피지 추정 0.10%

    @property
    def round_trip_cost_pct(self) -> float:
        """왕복 비용 (%)"""
        return (self.fee_oneway_pct * 2) + self.tax_sell_pct + (self.slippage_pct * 2)

    # ── 구간(binning) ──
    score_bins: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100.01),
    ])

    # ── 표본 안전 ──
    min_n: int = 30                   # 최소 raw 표본
    min_effective_n: float = 10.0     # 최소 가중 유효 표본
    laplace_alpha: float = 1.0        # 라플라스 스무딩 α (wins+α)/(n+2α)

    # ── 켈리 제한 ──
    kelly_fraction_mult: float = 0.25  # Quarter-Kelly
    kelly_cap: float = 0.10            # 단일 종목 최대 비중 10%
    kelly_floor_n: int = 30            # 이 미만 표본이면 kelly=0

    # ── lookback ──
    lookback_days: int = 180           # 최근 N 영업일 데이터만 사용

    # ── 시간 가중 ──
    half_life_days: int = 90

    # ── 기업행위 필터 ──
    corporate_action_threshold_pct: float = 30.0  # |수익률| > 30% → 분할/합병 의심 제외


DEFAULT_BT_CONFIG = BacktestConfig()


# ═══════════════════════════════════════════════════
#  2. 성과 확정 (look-ahead 차단)
# ═══════════════════════════════════════════════════

def _get_trade_days(out_dir: str) -> List[str]:
    """price_snapshot 날짜 목록 (정렬)"""
    pattern = os.path.join(out_dir, "recommend_*.csv")
    days = []
    for f in sorted(glob(pattern)):
        base = os.path.basename(f)
        ymd = base.replace("recommend_", "").replace(".csv", "")
        if ymd not in ("latest", "latest_cp949") and len(ymd) == 8 and ymd.isdigit():
            days.append(ymd)
    return sorted(days)


def _offset_bday(trade_days: List[str], ymd: str, offset: int) -> Optional[str]:
    """trade_days에서 ymd 기준 offset 영업일 후 날짜"""
    try:
        idx = trade_days.index(ymd)
        target_idx = idx + offset
        if 0 <= target_idx < len(trade_days):
            return trade_days[target_idx]
    except ValueError:
        pass
    return None


def _load_price_snapshot(out_dir: str, ymd: str) -> Dict[str, Dict[str, float]]:
    """price_snapshot → {code: {open, high, low, close}}"""
    path = os.path.join(out_dir, f"price_snapshot_{ymd}.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype={"종목코드": str})
        df["종목코드"] = df["종목코드"].str.zfill(6)
        result = {}
        for _, row in df.iterrows():
            result[row["종목코드"]] = {
                "open": float(row.get("시가", 0) or 0),
                "high": float(row.get("고가", 0) or 0),
                "low": float(row.get("저가", 0) or 0),
                "close": float(row.get("종가", 0) or 0),
            }
        return result
    except Exception:
        return {}


def compute_realized_returns(
    out_dir: str,
    as_of_ymd: str,
    config: BacktestConfig = DEFAULT_BT_CONFIG,
) -> pd.DataFrame:
    """
    확정된 추천의 실현 수익률 계산.

    안전장치:
    1. rec_date <= as_of_ymd - horizon_bdays (미확정 제외)
    2. 진입가 = 다음날 시가, 청산가 = N일 후 종가
    3. 비용 차감 (왕복)

    Returns: DataFrame [rec_date, code, score, entry_price, exit_price,
                        ret_gross_pct, ret_net_pct, win]
    """
    trade_days = _get_trade_days(out_dir)
    if not trade_days:
        return pd.DataFrame()

    # ✅ 안전장치 1: 성과 확정 가능한 날짜만 (look-ahead 차단)
    cutoff = _offset_bday(trade_days, as_of_ymd, -config.horizon_bdays)
    if cutoff is None:
        # as_of_ymd가 trade_days에 없으면 보수적으로 처리
        try:
            as_of_idx = next(i for i, d in enumerate(trade_days) if d >= as_of_ymd)
            cutoff_idx = as_of_idx - config.horizon_bdays
            cutoff = trade_days[max(0, cutoff_idx)] if cutoff_idx >= 0 else None
        except StopIteration:
            cutoff = None

    if cutoff is None:
        return pd.DataFrame()

    # lookback 제한
    lookback_start = _offset_bday(trade_days, as_of_ymd, -config.lookback_days)

    results = []
    for rec_ymd in trade_days:
        if rec_ymd > cutoff:
            break
        if lookback_start and rec_ymd < lookback_start:
            continue

        # 추천 파일 로드
        rec_path = os.path.join(out_dir, f"recommend_{rec_ymd}.csv")
        if not os.path.exists(rec_path):
            continue

        try:
            rec_df = pd.read_csv(rec_path, dtype={"종목코드": str})
        except Exception:
            continue

        if rec_df.empty or "종목코드" not in rec_df.columns:
            continue
        rec_df["종목코드"] = rec_df["종목코드"].str.zfill(6)

        # ✅ 안전장치 2: 진입일 = 다음 영업일, 청산일 = 진입일 + horizon
        entry_ymd = _offset_bday(trade_days, rec_ymd, 1)
        exit_ymd = _offset_bday(trade_days, rec_ymd, 1 + config.horizon_bdays)
        if not entry_ymd or not exit_ymd:
            continue

        entry_prices = _load_price_snapshot(out_dir, entry_ymd)
        exit_prices = _load_price_snapshot(out_dir, exit_ymd)
        if not entry_prices or not exit_prices:
            continue

        # 점수 컬럼 결정
        score_col = None
        for c in ["DISPLAY_SCORE", "FINAL_SCORE", "RANK_SCORE", "TOTAL_SCORE"]:
            if c in rec_df.columns:
                score_col = c
                break
        if score_col is None:
            continue

        for _, row in rec_df.iterrows():
            code = row["종목코드"]
            if code not in entry_prices or code not in exit_prices:
                continue

            entry_p = entry_prices[code]["open"]
            exit_p = exit_prices[code]["close"]
            if entry_p <= 0 or exit_p <= 0:
                continue

            # ✅ 안전장치 7: 기업행위(액면분할/거래정지/상폐) 필터
            # 비정상 가격 변동 감지: |수익률| > 30% → 분할/합병 가능성 → 제외
            raw_ret = abs(exit_p / entry_p - 1) * 100
            if raw_ret > config.corporate_action_threshold_pct:
                logger.debug(f"기업행위 의심 제외: {code} ret={raw_ret:.1f}%")
                continue
            # 거래정지: 시가=종가=고가=저가=0 or 전부 동일 + 거래량 0
            ep = entry_prices[code]
            if ep["open"] == ep["close"] == ep["high"] == ep["low"]:
                logger.debug(f"거래정지 의심 제외: {code}")
                continue

            # ── 중간 SL/TP 체크 (간소화) ──
            stop_price = float(row.get("손절가", 0) or 0)
            actual_exit = exit_p

            # ✅ 안전장치 3: 비용 차감
            ret_gross = (actual_exit / entry_p - 1) * 100
            ret_net = ret_gross - config.round_trip_cost_pct

            results.append({
                "rec_date": rec_ymd,
                "code": code,
                "score": float(row.get(score_col, 0)),
                "entry_price": entry_p,
                "exit_price": actual_exit,
                "ret_gross_pct": round(ret_gross, 4),
                "ret_net_pct": round(ret_net, 4),
                "win": 1 if ret_net > 0 else 0,
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════
#  3. 점수대별 승률 테이블 빌드
# ═══════════════════════════════════════════════════

def build_winrate_table(
    returns_df: pd.DataFrame,
    config: BacktestConfig = DEFAULT_BT_CONFIG,
    half_life_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    실현 수익률 → 점수 구간별 승률 테이블.

    안전장치:
    3. binning 고정 (config.score_bins)
    4. min_n + 라플라스 스무딩
    """
    if returns_df.empty:
        return pd.DataFrame()

    hl = half_life_days or config.half_life_days

    # 시간 가중
    try:
        rec_dates = pd.to_datetime(returns_df["rec_date"].astype(str), format="%Y%m%d")
        now = rec_dates.max()
        age_days = (now - rec_dates).dt.total_seconds() / 86400.0
        lam = np.log(2) / hl
        weights = np.exp(-lam * age_days.values)
    except Exception:
        weights = np.ones(len(returns_df))

    rows = []
    for lo, hi in config.score_bins:
        mask = (returns_df["score"] >= lo) & (returns_df["score"] < hi)
        sub = returns_df[mask]
        w_sub = weights[mask.values]

        n_raw = len(sub)
        n_eff = float(np.sum(w_sub)) if n_raw > 0 else 0.0

        # ✅ 안전장치 4: 표본 부족 시 건너뜀
        if n_raw < config.min_n or n_eff < config.min_effective_n:
            # 라플라스만 적용한 보수적 fallback
            alpha = config.laplace_alpha
            wins_raw = int(sub["win"].sum()) if n_raw > 0 else 0
            p_laplace = (wins_raw + alpha) / (n_raw + 2 * alpha) if n_raw > 0 else 0.45
            rows.append({
                "score_lo": lo,
                "score_hi": hi,
                "p_win": round(p_laplace, 4),
                "n_raw": n_raw,
                "n_effective": round(n_eff, 1),
                "avg_ret_net_pct": round(float(sub["ret_net_pct"].mean()), 4) if n_raw > 0 else 0.0,
                "sufficient": False,
            })
            continue

        # 가중 승률 + 라플라스
        alpha = config.laplace_alpha
        w_wins = float(np.sum(w_sub * sub["win"].values))
        p_weighted = (w_wins + alpha) / (n_eff + 2 * alpha)

        rows.append({
            "score_lo": lo,
            "score_hi": hi,
            "p_win": round(p_weighted, 4),
            "n_raw": n_raw,
            "n_effective": round(n_eff, 1),
            "avg_ret_net_pct": round(float(sub["ret_net_pct"].mean()), 4),
            "sufficient": True,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════
#  4. 켈리 매핑 (안전장치 5)
# ═══════════════════════════════════════════════════

def kelly_from_table(
    score: float,
    winrate_table: pd.DataFrame,
    avg_b_ratio: float = 2.0,
    config: BacktestConfig = DEFAULT_BT_CONFIG,
) -> float:
    """
    승률 테이블에서 score에 해당하는 켈리 비중 산출.

    안전장치 5:
    - fractional kelly (quarter)
    - 단일 종목 cap
    - 표본 부족(sufficient=False) → kelly=0
    """
    if winrate_table.empty:
        return 0.0

    # 구간 매칭
    for _, row in winrate_table.iterrows():
        if row["score_lo"] <= score < row["score_hi"]:
            # ✅ 표본 부족 → 0
            if not row.get("sufficient", False):
                if row["n_raw"] < config.kelly_floor_n:
                    return 0.0

            p = row["p_win"]
            q = 1.0 - p
            if avg_b_ratio <= 0 or p <= 0:
                return 0.0

            f_full = p - (q / avg_b_ratio)
            if f_full <= 0:
                return 0.0

            # ✅ 안전장치 5: fractional + cap
            f_safe = f_full * config.kelly_fraction_mult
            return min(f_safe, config.kelly_cap)

    return 0.0


# ═══════════════════════════════════════════════════
#  5. 자동 캘리브레이션 (main() 끝에 호출)
# ═══════════════════════════════════════════════════

def auto_calibrate(
    out_dir: str,
    as_of_ymd: str,
    config: BacktestConfig = DEFAULT_BT_CONFIG,
) -> Dict:
    """
    collector.main() 끝에서 호출.
    1. 실현 수익률 계산
    2. 승률 테이블 빌드
    3. JSON 저장
    4. 요약 리턴

    Returns: {
        "n_trades": int,
        "n_bins_sufficient": int,
        "overall_winrate": float,
        "overall_avg_ret_net": float,
        "table_path": str,
    }
    """
    logger.info(f"🔄 자동 백테스트 캘리브레이션 시작 (as_of={as_of_ymd})")

    # 1. 실현 수익률
    returns_df = compute_realized_returns(out_dir, as_of_ymd, config)
    if returns_df.empty:
        logger.info("📊 확정된 추천 없음 → 캘리브레이션 스킵")
        return {"n_trades": 0, "n_bins_sufficient": 0}

    # 2. 승률 테이블
    table = build_winrate_table(returns_df, config)

    # 3. 저장 (버전 태깅 포함)
    meta = {
        "version": "v14.0",
        "as_of_ymd": as_of_ymd,
        "horizon_bdays": config.horizon_bdays,
        "entry_rule": config.entry_rule,
        "exit_rule": config.exit_rule,
        "min_n": config.min_n,
        "half_life_days": config.half_life_days,
        "round_trip_cost_pct": config.round_trip_cost_pct,
        "kelly_fraction_mult": config.kelly_fraction_mult,
        "kelly_cap": config.kelly_cap,
        "corporate_action_threshold_pct": config.corporate_action_threshold_pct,
        "n_trades": len(returns_df),
    }

    # 테이블 + 메타를 하나의 JSON으로
    save_obj = {
        "meta": meta,
        "table": json.loads(table.to_json(orient="records")) if not table.empty else [],
    }
    table_path = os.path.join(out_dir, f"winrate_table_{as_of_ymd}.json")
    try:
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(save_obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"승률 테이블 저장 실패: {e}")
        table_path = ""

    # latest 심볼릭
    latest_path = os.path.join(out_dir, "winrate_table_latest.json")
    try:
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(save_obj, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # 4. 요약
    n_sufficient = int(table["sufficient"].sum()) if "sufficient" in table.columns else 0
    overall_wr = float(returns_df["win"].mean()) if len(returns_df) > 0 else 0.0
    overall_ret = float(returns_df["ret_net_pct"].mean()) if len(returns_df) > 0 else 0.0

    summary = {
        "n_trades": len(returns_df),
        "n_bins_sufficient": n_sufficient,
        "overall_winrate": round(overall_wr, 4),
        "overall_avg_ret_net": round(overall_ret, 4),
        "table_path": table_path,
    }

    logger.info(f"📊 캘리브레이션 완료: {summary['n_trades']}건, "
                f"승률={summary['overall_winrate']:.1%}, "
                f"평균수익={summary['overall_avg_ret_net']:+.2f}%, "
                f"충분구간={n_sufficient}/{len(table)}")

    return summary
