
# -*- coding: utf-8 -*-
"""No-Buy Breaker Backtest v3.9.26b.

Evidence-Gated No-Buy Breaker의 검증 커버리지를 보강한다.

v3.9.26 문제:
- No-Buy Breaker 후보는 잡히지만 기존 backtest_top1/top3_trades에 없으면
  REALIZED_RET_5D가 전부 NaN이 되어 N=0 처리된다.

v3.9.26b 보강:
- 우선 기존 backtest_top1/top3_trades와 조인한다.
- 조인 실패 시 recommend_YYYYMMDD.csv의 미래 종가 스냅샷으로 5거래일 직접 성과를 계산한다.
- 미래 5거래일 데이터가 부족하면 PENDING_OUTCOME으로 남기고 production N에는 포함하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

MIN_N = 20
MIN_WIN_RATE_5D = 55.0
MIN_AVG_RET_5D = 0.0
MIN_AVG_ALPHA_5D = 0.0
MAX_STOP_RISK_RATE = 35.0
MIN_MAX_LOSS_AVG = -7.0
RECENT_WINDOW = 20
HORIZON_DAYS = 5
_DATE_RE = re.compile(r"(20\d{6})")


@dataclass(frozen=True)
class BreakerRule:
    rule_id: str
    description: str


RULES: List[BreakerRule] = [
    BreakerRule(
        "RULE_A_STRUCT90_TIMING60",
        "ROUTE active + BUY_NOW_PASS + EBS + STRUCT>=90 + TIMING>=60 + FINAL>=75 + RR>=1.10",
    ),
    BreakerRule(
        "RULE_B_FINAL80_ENTRY_CLEAN",
        "ROUTE active + BUY_NOW_PASS + FINAL>=80 + entry gap/VWAP/POC clean",
    ),
    BreakerRule(
        "RULE_C_HIGH_STRUCT_LOW_TIMING_RECOVERY",
        "High STRUCT recovery: STRUCT>=95 + TIMING>=55 + AI>=70 + clean risk",
    ),
    BreakerRule(
        "RULE_D_ROUTE_ARMED_CLEAN_ENTRY",
        "Active route clean-entry fallback with MFI/short-term loss guard",
    ),
]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def _date_from_path(path: Path) -> str:
    m = _DATE_RE.search(path.name)
    return m.group(1) if m else path.stem


def _norm_code(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    return digits.zfill(6) if digits else ""


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, dtype={"종목코드": str, "code": str}, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def _first_existing(df: pd.DataFrame, cols: List[str], default: float = float("nan")) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype="float64")
    for col in cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out = out.where(out.notna(), vals)
    return out


def official_buy_count(df: pd.DataFrame) -> int:
    top = _num(df, "TOP_PICK", 0).astype(int)
    eligible = _num(df, "BUY_NOW_ELIGIBLE", 0).astype(int)
    return int(((top == 1) & (eligible == 1)).sum())


def rule_mask(df: pd.DataFrame, rule_id: str) -> pd.Series:
    route = _str(df, "ROUTE").str.upper().str.strip()
    risk = _str(df, "ENTRY_RISK_LEVEL", "GREEN").str.upper().str.strip()
    buy_pass = _num(df, "BUY_NOW_PASS", 0)
    pass_ebs = _num(df, "PASS_EBS", 0)
    volume = _num(df, "거래대금(억원)", 0)
    final = _num(df, "FINAL_SCORE", 0)
    struct = _num(df, "STRUCT_SCORE", 0)
    timing = _num(df, "TIMING_SCORE", 0)
    ai = _num(df, "AI_SCORE", 0)
    rr = _num(df, "RR_NOW_TP1", 0)
    gap = _num(df, "ENTRY_GAP_PCT", 99)
    if "ENTRY_GAP_PCT" not in df.columns and "GAP_PCT" in df.columns:
        gap = _num(df, "GAP_PCT", 99)
    vwap = _num(df, "VWAP_GAP", 0)
    poc = _num(df, "POC_GAP", 0)
    mfi = _num(df, "MFI14", 50)
    ret_1d = _num(df, "ret_1d_%", 0)
    ret_5d = _num(df, "ret_5d_%", 0)

    base_clean = (
        route.isin(["ARMED", "ATTACK"])
        & (buy_pass == 1)
        & (pass_ebs == 1)
        & (volume >= 50)
        & (gap <= 3)
        & (rr >= 1.10)
        & (~risk.isin(["RED", "ORANGE"]))
    )

    rule_id = str(rule_id or "").upper().strip()
    if rule_id == "RULE_A_STRUCT90_TIMING60":
        return base_clean & (struct >= 90) & (timing >= 60) & (final >= 75)
    if rule_id == "RULE_B_FINAL80_ENTRY_CLEAN":
        return base_clean & (final >= 80) & (vwap <= 12) & (poc <= 40)
    if rule_id == "RULE_C_HIGH_STRUCT_LOW_TIMING_RECOVERY":
        return base_clean & (struct >= 95) & (timing >= 55) & (ai >= 70) & (vwap <= 12)
    if rule_id == "RULE_D_ROUTE_ARMED_CLEAN_ENTRY":
        return base_clean & (vwap <= 12) & (poc <= 40) & (mfi <= 80) & (ret_5d <= 20) & (ret_1d > -5)
    return pd.Series(False, index=df.index, dtype=bool)


def select_top_candidate(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    cand = df.loc[mask].copy()
    if cand.empty:
        return cand
    cand["_SORT_FINAL"] = _num(cand, "FINAL_SCORE", 0)
    cand["_SORT_ELITE"] = _num(cand, "ELITE_SCORE", 0)
    cand["_SORT_RR"] = _num(cand, "RR_NOW_TP1", 0)
    cand["_SORT_GAP"] = _num(cand, "ENTRY_GAP_PCT", 99)
    cand = cand.sort_values(
        ["_SORT_FINAL", "_SORT_ELITE", "_SORT_RR", "_SORT_GAP"],
        ascending=[False, False, False, True],
    )
    return cand.head(1)


def load_realized_trades(data_dir: str | Path) -> pd.DataFrame:
    data_path = Path(data_dir)
    frames: List[pd.DataFrame] = []
    for pat in ("backtest_top1_trades_*.csv", "backtest_top3_trades_*.csv"):
        for path in sorted(data_path.glob(pat)):
            if "latest" in path.name.lower():
                continue
            df = _read_csv(path)
            if df is None or df.empty or "date" not in df.columns or "code" not in df.columns:
                continue
            tmp = df.copy()
            tmp["SNAPSHOT_DATE"] = tmp["date"].astype(str).str.replace("-", "", regex=False).str[:8]
            tmp["code_norm"] = tmp["code"].map(_norm_code)
            tmp["TRADE_SOURCE"] = path.name
            ret = pd.to_numeric(tmp.get("net_pct", tmp.get("ret_pct")), errors="coerce")
            tmp["REALIZED_RET_5D"] = ret
            if "ALPHA_%" in tmp.columns:
                tmp["REALIZED_ALPHA_5D"] = pd.to_numeric(tmp["ALPHA_%"], errors="coerce")
            else:
                tmp["REALIZED_ALPHA_5D"] = tmp["REALIZED_RET_5D"]
            if "stop_hit" in tmp.columns:
                tmp["STOP_HIT_REALIZED"] = tmp["stop_hit"].astype(str).str.lower().isin(["1", "true", "yes"])
            else:
                tmp["STOP_HIT_REALIZED"] = tmp["REALIZED_RET_5D"] <= -6.0
            keep = ["SNAPSHOT_DATE", "code_norm", "REALIZED_RET_5D", "REALIZED_ALPHA_5D", "STOP_HIT_REALIZED", "TRADE_SOURCE"]
            frames.append(tmp[keep])
    if not frames:
        return pd.DataFrame(columns=["SNAPSHOT_DATE", "code_norm", "REALIZED_RET_5D", "REALIZED_ALPHA_5D", "STOP_HIT_REALIZED", "TRADE_SOURCE"])
    out = pd.concat(frames, ignore_index=True)
    out["_HAS_RESULT"] = out["REALIZED_RET_5D"].notna().astype(int)
    out = out.sort_values(["SNAPSHOT_DATE", "code_norm", "_HAS_RESULT"], ascending=[True, True, False])
    out = out.drop_duplicates(["SNAPSHOT_DATE", "code_norm"], keep="first")
    return out.drop(columns=["_HAS_RESULT"])


def load_recommend_snapshot_prices(paths: Iterable[Path]) -> Tuple[pd.DataFrame, List[str]]:
    """recommend snapshots에서 날짜/종목별 종가를 직접 성과계산용으로 적재한다."""
    frames: List[pd.DataFrame] = []
    dates: List[str] = []
    for path in sorted(paths):
        day = _date_from_path(path)
        df = _read_csv(path)
        if df is None or df.empty:
            continue
        code_col = "종목코드" if "종목코드" in df.columns else "code" if "code" in df.columns else None
        if code_col is None:
            continue
        close_col = "종가" if "종가" in df.columns else "Close" if "Close" in df.columns else "close" if "close" in df.columns else None
        if close_col is None:
            continue
        tmp = pd.DataFrame({
            "SNAPSHOT_DATE": day,
            "code_norm": df[code_col].map(_norm_code),
            "SNAP_CLOSE": pd.to_numeric(df[close_col], errors="coerce"),
        })
        tmp = tmp[tmp["code_norm"].astype(str).str.len() > 0]
        frames.append(tmp)
        dates.append(day)
    if not frames:
        return pd.DataFrame(columns=["SNAPSHOT_DATE", "code_norm", "SNAP_CLOSE"]), []
    px = pd.concat(frames, ignore_index=True).dropna(subset=["SNAP_CLOSE"])
    px = px.drop_duplicates(["SNAPSHOT_DATE", "code_norm"], keep="last")
    return px, sorted(set(dates))


def build_direct_outcomes(candidates: pd.DataFrame, snapshot_prices: pd.DataFrame, trading_dates: List[str], horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    """기존 backtest trade 조인 실패 후보에 대해 recommend 미래 종가로 직접 5D 성과를 계산한다."""
    cols = [
        "SNAPSHOT_DATE", "code_norm", "DIRECT_REALIZED_RET_5D", "DIRECT_REALIZED_ALPHA_5D",
        "DIRECT_STOP_HIT_REALIZED", "DIRECT_TP1_HIT_REALIZED", "DIRECT_MAX_DRAWDOWN_REALIZED",
        "DIRECT_OUTCOME_DAYS", "DIRECT_TRADE_SOURCE",
    ]
    if candidates is None or candidates.empty or snapshot_prices is None or snapshot_prices.empty:
        return pd.DataFrame(columns=cols)

    date_to_pos = {d: i for i, d in enumerate(trading_dates)}
    price_map = snapshot_prices.set_index(["SNAPSHOT_DATE", "code_norm"])["SNAP_CLOSE"].to_dict()
    out_rows: List[Dict[str, object]] = []

    # 후보 행은 rule별로 중복될 수 있으므로 날짜+종목+entry/stop/tp 기준으로 중복 제거
    cand = candidates.copy()
    cand["_ENTRY_PRICE"] = _first_existing(cand, ["추천매수가", "ENTRY_PRICE", "종가", "Close", "close"])
    cand["_STOP_PRICE"] = _first_existing(cand, ["손절가", "STOP_PRICE"])
    cand["_TP1_PRICE"] = _first_existing(cand, ["추천매도가1", "TP1", "TP1_PRICE"])
    cand = cand.drop_duplicates(["SNAPSHOT_DATE", "code_norm", "_ENTRY_PRICE", "_STOP_PRICE", "_TP1_PRICE"], keep="first")

    for _, row in cand.iterrows():
        day = str(row.get("SNAPSHOT_DATE", ""))[:8]
        code = str(row.get("code_norm", ""))
        entry = pd.to_numeric(row.get("_ENTRY_PRICE"), errors="coerce")
        stop_px = pd.to_numeric(row.get("_STOP_PRICE"), errors="coerce")
        tp1_px = pd.to_numeric(row.get("_TP1_PRICE"), errors="coerce")
        pos = date_to_pos.get(day)
        base = {"SNAPSHOT_DATE": day, "code_norm": code}
        if pos is None or pd.isna(entry) or float(entry) <= 0:
            out_rows.append({**base, "DIRECT_TRADE_SOURCE": "NO_ENTRY_PRICE", "DIRECT_OUTCOME_DAYS": 0})
            continue
        future_dates = trading_dates[pos + 1 : pos + 1 + horizon]
        closes = []
        used_dates = []
        for fd in future_dates:
            val = price_map.get((fd, code))
            if val is not None and not pd.isna(val):
                closes.append(float(val))
                used_dates.append(fd)
        if len(closes) < horizon:
            out_rows.append({
                **base,
                "DIRECT_TRADE_SOURCE": f"PENDING_DIRECT_RECOMMEND_CLOSE_{len(closes)}D",
                "DIRECT_OUTCOME_DAYS": len(closes),
            })
            continue
        exit_close = closes[horizon - 1]
        ret = (exit_close / float(entry) - 1.0) * 100.0
        drawdowns = [(c / float(entry) - 1.0) * 100.0 for c in closes]
        max_dd = min(drawdowns) if drawdowns else 0.0
        stop_hit = False
        if not pd.isna(stop_px) and float(stop_px) > 0:
            stop_hit = min(closes) <= float(stop_px)
        tp1_hit = False
        if not pd.isna(tp1_px) and float(tp1_px) > 0:
            tp1_hit = max(closes) >= float(tp1_px)
        out_rows.append({
            **base,
            "DIRECT_REALIZED_RET_5D": round(ret, 4),
            # KOSPI future return이 없을 때는 alpha=ret로 보수적 대체. 추후 벤치 스냅샷 연결 가능.
            "DIRECT_REALIZED_ALPHA_5D": round(ret, 4),
            "DIRECT_STOP_HIT_REALIZED": bool(stop_hit),
            "DIRECT_TP1_HIT_REALIZED": bool(tp1_hit),
            "DIRECT_MAX_DRAWDOWN_REALIZED": round(max_dd, 4),
            "DIRECT_OUTCOME_DAYS": len(closes),
            "DIRECT_TRADE_SOURCE": "DIRECT_RECOMMEND_CLOSE_5D",
            "DIRECT_EXIT_DATE": used_dates[horizon - 1] if len(used_dates) >= horizon else "",
        })
    return pd.DataFrame(out_rows, columns=list(dict.fromkeys(cols + ["DIRECT_EXIT_DATE"])))


def collect_rule_candidates(paths: Iterable[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidates: List[pd.DataFrame] = []
    day_rows: List[Dict[str, object]] = []
    for path in sorted(paths):
        df = _read_csv(path)
        if df is None or df.empty:
            continue
        day = _date_from_path(path)
        off_n = official_buy_count(df)
        no_buy = off_n == 0
        day_rows.append({"SNAPSHOT_DATE": day, "FILE": path.name, "OFFICIAL_BUY_COUNT": off_n, "NO_BUY_DAY": int(no_buy)})
        if not no_buy:
            continue
        code_col = "종목코드" if "종목코드" in df.columns else "code" if "code" in df.columns else None
        if code_col is None:
            continue
        for rule in RULES:
            mask = rule_mask(df, rule.rule_id)
            top = select_top_candidate(df, mask)
            if top.empty:
                continue
            row = top.copy()
            row["SNAPSHOT_DATE"] = day
            row["SOURCE_FILE"] = path.name
            row["NO_BUY_BREAKER_RULE_ID"] = rule.rule_id
            row["RULE_ID"] = rule.rule_id
            row["RULE_DESC"] = rule.description
            row["code_norm"] = row[code_col].map(_norm_code)
            candidates.append(row)
    cand_df = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    day_df = pd.DataFrame(day_rows)
    return cand_df, day_df


def merge_realized_coverage(candidates: pd.DataFrame, trades: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    merged = candidates.merge(trades, on=["SNAPSHOT_DATE", "code_norm"], how="left")
    if direct is not None and not direct.empty:
        merged = merged.merge(direct, on=["SNAPSHOT_DATE", "code_norm"], how="left")
    else:
        merged["DIRECT_TRADE_SOURCE"] = pd.NA

    # 기존 trade 파일 결과가 최우선, 없으면 직접 성과계산 사용.
    if "DIRECT_REALIZED_RET_5D" in merged.columns:
        merged["REALIZED_RET_5D"] = merged["REALIZED_RET_5D"].where(merged["REALIZED_RET_5D"].notna(), merged["DIRECT_REALIZED_RET_5D"])
        merged["REALIZED_ALPHA_5D"] = merged["REALIZED_ALPHA_5D"].where(merged["REALIZED_ALPHA_5D"].notna(), merged["DIRECT_REALIZED_ALPHA_5D"])
        merged["STOP_HIT_REALIZED"] = merged["STOP_HIT_REALIZED"].where(merged["STOP_HIT_REALIZED"].notna(), merged["DIRECT_STOP_HIT_REALIZED"])
        merged["TRADE_SOURCE"] = merged["TRADE_SOURCE"].where(merged["TRADE_SOURCE"].notna(), merged["DIRECT_TRADE_SOURCE"])
    merged["OUTCOME_STATUS"] = "NO_OUTCOME"
    merged.loc[merged["REALIZED_RET_5D"].notna(), "OUTCOME_STATUS"] = "REALIZED"
    if "TRADE_SOURCE" in merged.columns:
        pending = merged["TRADE_SOURCE"].astype(str).str.startswith("PENDING_DIRECT")
        merged.loc[pending & merged["REALIZED_RET_5D"].isna(), "OUTCOME_STATUS"] = "PENDING"
    return merged


def summarize_rules(merged: pd.DataFrame, candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    candidates = candidates if candidates is not None else pd.DataFrame()
    for rule in RULES:
        cand_sub = candidates[candidates.get("RULE_ID", pd.Series(dtype=str)) == rule.rule_id].copy() if not candidates.empty else pd.DataFrame()
        raw_sub = merged[merged.get("RULE_ID", pd.Series(dtype=str)) == rule.rule_id].copy() if not merged.empty else pd.DataFrame()
        realized_sub = raw_sub[raw_sub["REALIZED_RET_5D"].notna()].copy() if not raw_sub.empty and "REALIZED_RET_5D" in raw_sub.columns else pd.DataFrame()
        candidate_n = int(len(cand_sub))
        realized_n = int(len(realized_sub))
        pending_n = int((raw_sub.get("OUTCOME_STATUS", pd.Series(dtype=str)) == "PENDING").sum()) if not raw_sub.empty else 0
        coverage = round((realized_n / candidate_n) * 100.0, 2) if candidate_n else 0.0
        if realized_n == 0:
            reason = "NO_CANDIDATE" if candidate_n == 0 else "NO_REALIZED_OUTCOME"
            if pending_n > 0:
                reason = "PENDING_OUTCOME"
            rows.append({
                "RULE_ID": rule.rule_id,
                "RULE_DESC": rule.description,
                "CANDIDATE_N": candidate_n,
                "REALIZED_N": 0,
                "PENDING_N": pending_n,
                "VALIDATION_COVERAGE": coverage,
                "N": 0,
                "WIN_RATE_5D": 0.0,
                "AVG_RET_5D": 0.0,
                "MEDIAN_RET_5D": 0.0,
                "AVG_ALPHA_5D": 0.0,
                "STOP_RISK_RATE": 0.0,
                "MAX_LOSS_AVG": 0.0,
                "RECENT_N": 0,
                "RECENT_AVG_RET_5D": 0.0,
                "DECISION": "REJECT_INSUFFICIENT_SAMPLE",
                "REJECT_REASON": reason,
            })
            continue
        ret = pd.to_numeric(realized_sub["REALIZED_RET_5D"], errors="coerce").dropna()
        alpha = pd.to_numeric(realized_sub.get("REALIZED_ALPHA_5D", realized_sub["REALIZED_RET_5D"]), errors="coerce").dropna()
        stop = realized_sub.get("STOP_HIT_REALIZED", pd.Series(False, index=realized_sub.index)).fillna(False).astype(bool)
        recent = realized_sub.sort_values("SNAPSHOT_DATE").tail(RECENT_WINDOW)
        recent_ret = pd.to_numeric(recent["REALIZED_RET_5D"], errors="coerce").dropna()
        stop_risk = float(stop.mean() * 100.0) if len(stop) else 0.0
        loss_only = ret[ret < 0]
        max_loss_avg = float(loss_only.mean()) if len(loss_only) else 0.0
        win_rate = float((ret > 0).mean() * 100.0) if len(ret) else 0.0
        avg_ret = float(ret.mean()) if len(ret) else 0.0
        med_ret = float(ret.median()) if len(ret) else 0.0
        avg_alpha = float(alpha.mean()) if len(alpha) else avg_ret
        recent_avg = float(recent_ret.mean()) if len(recent_ret) else 0.0

        decision = "PASS_PRODUCTION_GATE"
        reject_reason = "PASS"
        if realized_n < MIN_N:
            decision, reject_reason = "REJECT_INSUFFICIENT_SAMPLE", "INSUFFICIENT_REALIZED_N"
        elif win_rate < MIN_WIN_RATE_5D:
            decision, reject_reason = "REJECT_LOW_WIN_RATE", "LOW_WIN_RATE"
        elif avg_ret <= MIN_AVG_RET_5D:
            decision, reject_reason = "REJECT_NEGATIVE_EV", "NEGATIVE_EV"
        elif avg_alpha <= MIN_AVG_ALPHA_5D:
            decision, reject_reason = "REJECT_NEGATIVE_ALPHA", "NEGATIVE_ALPHA"
        elif stop_risk > MAX_STOP_RISK_RATE:
            decision, reject_reason = "REJECT_HIGH_STOP_RISK", "HIGH_STOP_RISK"
        elif max_loss_avg <= MIN_MAX_LOSS_AVG:
            decision, reject_reason = "REJECT_HIGH_DRAWDOWN", "HIGH_DRAWDOWN"
        elif len(recent_ret) >= 5 and recent_avg <= 0:
            decision, reject_reason = "REJECT_RECENT_WEAKNESS", "RECENT_WEAKNESS"

        rows.append({
            "RULE_ID": rule.rule_id,
            "RULE_DESC": rule.description,
            "CANDIDATE_N": candidate_n,
            "REALIZED_N": realized_n,
            "PENDING_N": pending_n,
            "VALIDATION_COVERAGE": coverage,
            "N": realized_n,
            "WIN_RATE_5D": round(win_rate, 2),
            "AVG_RET_5D": round(avg_ret, 2),
            "MEDIAN_RET_5D": round(med_ret, 2),
            "AVG_ALPHA_5D": round(avg_alpha, 2),
            "STOP_RISK_RATE": round(stop_risk, 2),
            "MAX_LOSS_AVG": round(max_loss_avg, 2),
            "RECENT_N": int(len(recent_ret)),
            "RECENT_AVG_RET_5D": round(recent_avg, 2),
            "DECISION": decision,
            "REJECT_REASON": reject_reason,
        })
    return pd.DataFrame(rows)


def run_backtest(data_dir: str = "data", out_dir: str = "data") -> Dict[str, object]:
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in data_path.glob("recommend_*.csv") if "latest" not in p.name and _DATE_RE.search(p.name))
    candidates, days = collect_rule_candidates(paths)
    trades = load_realized_trades(data_path)
    prices, trading_dates = load_recommend_snapshot_prices(paths)
    direct = build_direct_outcomes(candidates, prices, trading_dates, horizon=HORIZON_DAYS)
    merged = merge_realized_coverage(candidates, trades, direct) if not candidates.empty else candidates.copy()
    summary = summarize_rules(merged, candidates)

    candidates_path = out_path / "no_buy_breaker_candidates_latest.csv"
    trades_path = out_path / "no_buy_breaker_trades_latest.csv"
    rules_path = out_path / "no_buy_breaker_rules_latest.csv"
    days_path = out_path / "no_buy_breaker_days_latest.csv"
    direct_path = out_path / "no_buy_breaker_direct_outcomes_latest.csv"
    json_path = out_path / "no_buy_breaker_backtest_latest.json"

    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    merged.to_csv(trades_path, index=False, encoding="utf-8-sig")
    summary.to_csv(rules_path, index=False, encoding="utf-8-sig")
    days.to_csv(days_path, index=False, encoding="utf-8-sig")
    direct.to_csv(direct_path, index=False, encoding="utf-8-sig")

    realized_rows = int(merged["REALIZED_RET_5D"].notna().sum()) if not merged.empty and "REALIZED_RET_5D" in merged.columns else 0
    pending_rows = int((merged.get("OUTCOME_STATUS", pd.Series(dtype=str)) == "PENDING").sum()) if not merged.empty else 0
    payload = {
        "version": "3.9.26b",
        "data_dir": str(data_path),
        "files": len(paths),
        "no_buy_days": int(days["NO_BUY_DAY"].sum()) if not days.empty and "NO_BUY_DAY" in days.columns else 0,
        "candidate_rows": int(len(candidates)),
        "realized_rows": realized_rows,
        "pending_rows": pending_rows,
        "rules": summary.to_dict(orient="records"),
        "outputs": {
            "candidates": str(candidates_path),
            "trades": str(trades_path),
            "rules": str(rules_path),
            "days": str(days_path),
            "direct_outcomes": str(direct_path),
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="No-Buy Breaker backtest v3.9.26b")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    payload = run_backtest(args.data_dir, args.out_dir)
    passed = [r for r in payload.get("rules", []) if r.get("DECISION") == "PASS_PRODUCTION_GATE"]
    print(
        f"[v3.9.26b] No-Buy Breaker backtest complete: "
        f"files={payload['files']} no_buy_days={payload['no_buy_days']} "
        f"candidates={payload['candidate_rows']} realized={payload['realized_rows']} "
        f"pending={payload['pending_rows']} pass_rules={len(passed)}"
    )


if __name__ == "__main__":
    main()
