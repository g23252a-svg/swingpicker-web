# -*- coding: utf-8 -*-
"""
no_buy_breaker_backtest_v3926.py — v25.0 FILL-AWARE 룰 인증 백테스트 (공식 신호의 열쇠공정)
═══════════════════════════════════════════════════════════════════════════════════════
공식 신규매수(TOP_PICK ∧ BUY_NOW_ELIGIBLE)는 v3.9.26 설계상 이 스크립트가 산출하는
PASS_PRODUCTION_GATE 룰이 있어야만 켜진다(pipeline_finalize.apply_evidence_gated_no_buy_breaker).
92일간 발화 0회의 원인이 바로 이 인증 부재였다.

■ v25.0 엔진 교체 이유 (구엔진의 두 가지 유령)
  구버전은 ① per_trade_log 조인(유령체결 30~36% 포함, v24.8에서 적출된 그 편향)과
  ② direct 경로(추천매수가 '도달 여부 미검증' 진입) 로 성과를 계산했다.
  → 인증 게이트가 오염된 저울 위에 있었다. v25.0은 데이터 소스를 통째로 교체한다:
     OHLCV 실경로 · 지정가 체결검증 · 체결일 기준 보유 · KOSPI 동일창 알파.

■ 체결/성과 규칙 (v24.8 Fill-Aware · v24.7 honest 재백테스트와 동일 기준)
  entry   = 추천매수가 (없으면 추천일 종가)
  fill    = 추천일 이후 FILL_WINDOW_DAYS(3)거래일 내 저가 ≤ entry → 최초 도달일 체결
  UNFILLED= 창 완결 & 미도달 (성과 계산 금지 — 유령 차단의 핵심)
  PENDING = 창 미완결 or 보유기간 미완결 (미래 데이터 부족 → 판정 유보, N 제외)
  RET_5D  = 체결일 포함 5거래일째 종가 / entry − 1
  MDD_5D  = 보유창 내 최저 저가 / entry − 1
  ALPHA_5D= RET_5D − KOSPI 동일 5거래일 수익(체결일 기준, kospi_daily.csv)

■ PASS_PRODUCTION_GATE 사다리 (전부 통과해야 인증)
  N≥20 → 승률≥55 → 평균>0 → 알파>0 → STOP_RISK(MDD≤-6% 비율)≤35 → 평균MDD≥-7
  → 최근20표본 평균>0 → [v25.0 신설] OOS(후반기) N≥8 & 승률≥50
  하나라도 미달 시 사유와 함께 REJECT — 억지 통과 없음. 오늘 전룰 REJECT라면
  그것이 정직한 답이며, 매일 밤 자동 재심사로 표본이 쌓여 좋은 레짐에서 스스로 열린다.

■ 호환성 (건드리지 않은 것)
  파일명/CLI(--data-dir/--out-dir)/6개 산출 파일/CSV 선두 17컬럼 스키마/JSON payload 키.
  진단 컬럼(N_UNFILLED, FILL_RATE_PCT, OOS_*, ENGINE 등)은 뒤에 추가 — 로더는
  DECISION/N/WIN_RATE_5D/AVG_RET_5D/AVG_ALPHA_5D만 읽으므로 무해.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── 인증 기준 (구버전과 동일 + OOS 안전조건 신설) ──
MIN_N = 20
MIN_WIN_RATE_5D = 55.0
MIN_AVG_RET_5D = 0.0
MIN_AVG_ALPHA_5D = 0.0
MAX_STOP_RISK_RATE = 35.0
MIN_MAX_LOSS_AVG = -7.0
RECENT_WINDOW = 20
HORIZON_DAYS = 5
# ── v25.0 엔진 파라미터 ──
FILL_WINDOW_DAYS = 3            # 실전 카드 'fill 3일' · v24.8 백필과 동일
STOP_RISK_MDD_PCT = -6.0        # STOP_RISK_RATE = MDD가 이 값 이하인 비율
OOS_MIN_N = 8                   # 후반기 표본 최소
OOS_MIN_WIN_RATE = 50.0         # 후반기 승률 최소
ENGINE_TAG = "FILL_AWARE_OHLCV_V25"

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

RULE_SCHEMA_COLS = [
    "RULE_ID", "RULE_DESC", "CANDIDATE_N", "REALIZED_N", "PENDING_N",
    "VALIDATION_COVERAGE", "N", "WIN_RATE_5D", "AVG_RET_5D", "MEDIAN_RET_5D",
    "AVG_ALPHA_5D", "STOP_RISK_RATE", "MAX_LOSS_AVG", "RECENT_N",
    "RECENT_AVG_RET_5D", "DECISION", "REJECT_REASON",
    # ── v25.0 진단 (뒤에 추가 — 로더 미사용, UI/감사용) ──
    "N_UNFILLED", "FILL_RATE_PCT", "OOS_N", "OOS_WIN_RATE_5D", "OOS_AVG_RET_5D",
    "ALPHA_N", "WINDOW_START", "WINDOW_END", "ENGINE",
]


# ═══════════════ 공용 헬퍼 (구버전 유지) ═══════════════
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


def official_buy_count(df: pd.DataFrame) -> int:
    top = _num(df, "TOP_PICK", 0).astype(int)
    eligible = _num(df, "BUY_NOW_ELIGIBLE", 0).astype(int)
    return int(((top == 1) & (eligible == 1)).sum())


def rule_mask(df: pd.DataFrame, rule_id: str) -> pd.Series:
    """v3.9.26 룰 문법 그대로 — pipeline_finalize.get_no_buy_breaker_rule_mask 와 동치."""
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


# ═══════════════ v25.0 OHLCV 체결검증 엔진 ═══════════════
def load_ohlcv_lut(data_dir: Path) -> Dict[str, np.ndarray]:
    """최신 ohlcv_cache parquet(전기간) → {code: [[ymd, open, high, low, close], ...]}."""
    files = sorted(glob.glob(str(data_dir / "ohlcv_cache_*.parquet")))
    if not files:
        return {}
    o = pd.read_parquet(files[-1]).reset_index()
    date_col = "Date" if "Date" in o.columns else o.columns[0]
    o["code"] = o["종목코드"].astype(str).str.zfill(6)
    o["d"] = pd.to_datetime(o[date_col]).dt.strftime("%Y%m%d").astype(int)
    o = o.sort_values(["code", "d"])
    cols = ["d", "시가", "고가", "저가", "종가"]
    return {c: g[cols].to_numpy(dtype=float) for c, g in o.groupby("code")}


def load_kospi_fwd(data_dir: Path, horizon: int = HORIZON_DAYS) -> Dict[int, float]:
    """kospi_daily.csv → {ymd: 동일 horizon 거래일 fwd 수익 %} (미완결일은 NaN 제외)."""
    p = data_dir / "kospi_daily.csv"
    if not p.exists():
        return {}
    try:
        k = pd.read_csv(p)
        k = k.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        dates = k["date"].astype(int).to_numpy()
        close = pd.to_numeric(k["close"], errors="coerce").to_numpy(dtype=float)
        out: Dict[int, float] = {}
        for i in range(len(k)):
            j = i + horizon - 1  # 체결일 포함 5거래일째 (종가 대비 종가와 정합)
            if j < len(k) and np.isfinite(close[i]) and close[i] > 0 and np.isfinite(close[j]):
                out[int(dates[i])] = (close[j] / close[i] - 1.0) * 100.0
        return out
    except Exception:
        return {}


def evaluate_candidate(lut: Dict[str, np.ndarray], code: str, rec_date: int,
                       entry: float, kospi_fwd: Dict[int, float]) -> Dict[str, object]:
    """단일 후보의 체결검증 성과. FILL_STATUS ∈ FILLED/UNFILLED/PENDING/DATA_MISSING."""
    res = dict(FILL_STATUS="DATA_MISSING", FILL_DAY=0, FILL_DATE=0,
               ENTRY_USED=np.nan, RET_5D=np.nan, MDD_5D=np.nan, ALPHA_5D=np.nan)
    a = lut.get(code)
    if a is None or len(a) == 0:
        return res
    idx = int(np.searchsorted(a[:, 0], rec_date))
    if idx >= len(a) or int(a[idx, 0]) != int(rec_date):
        return res
    e = float(entry) if np.isfinite(entry) and entry > 0 else float(a[idx, 4])
    if not np.isfinite(e) or e <= 0:
        return res
    res["ENTRY_USED"] = round(e, 2)

    seg = a[idx + 1: idx + 1 + FILL_WINDOW_DAYS]
    hit = np.where(seg[:, 3] <= e)[0] if len(seg) else np.array([], dtype=int)
    if len(hit) == 0:
        res["FILL_STATUS"] = "UNFILLED" if len(seg) >= FILL_WINDOW_DAYS else "PENDING"
        return res

    fj = idx + 1 + int(hit[0])
    res["FILL_DAY"] = int(hit[0]) + 1
    res["FILL_DATE"] = int(a[fj, 0])
    xj = fj + HORIZON_DAYS - 1          # 체결일 포함 5거래일째 종가
    if xj >= len(a):
        res["FILL_STATUS"] = "PENDING"
        return res
    res["FILL_STATUS"] = "FILLED"
    res["RET_5D"] = round((a[xj, 4] / e - 1.0) * 100.0, 2)
    res["MDD_5D"] = round((np.nanmin(a[fj: xj + 1, 3]) / e - 1.0) * 100.0, 2)
    kf = kospi_fwd.get(int(a[fj, 0]))
    if kf is not None and np.isfinite(kf):
        res["ALPHA_5D"] = round(res["RET_5D"] - kf, 2)
    return res


def collect_and_evaluate(paths: List[Path], lut: Dict[str, np.ndarray],
                         kospi_fwd: Dict[int, float]):
    """no-buy day의 룰별 후보 수집 + OHLCV 체결검증 평가 → (candidates_df, days_df)."""
    cand_rows: List[Dict] = []
    day_rows: List[Dict] = []
    for path in paths:
        df = _read_csv(path)
        if df is None or df.empty or "종목코드" not in df.columns:
            continue
        ymd = _date_from_path(path)
        try:
            rec_date = int(ymd)
        except ValueError:
            continue
        df["종목코드"] = df["종목코드"].map(_norm_code)
        official = official_buy_count(df)
        day = {"REC_DATE": rec_date, "NO_BUY_DAY": int(official == 0),
               "OFFICIAL_BUY_N": official}
        entry_s = pd.to_numeric(
            df["추천매수가"] if "추천매수가" in df.columns
            else pd.Series(np.nan, index=df.index), errors="coerce")
        name_s = _str(df, "종목명")
        for rule in RULES:
            mask = rule_mask(df, rule.rule_id)
            day[rule.rule_id] = int(mask.sum())
            if official > 0:            # 설계 원칙 1: 공식이 있으면 breaker 미개입 → 후보 제외
                continue
            for i in df.index[mask]:
                code = df.at[i, "종목코드"]
                if not code:
                    continue
                ev = evaluate_candidate(lut, code, rec_date,
                                        float(entry_s.get(i, np.nan)), kospi_fwd)
                cand_rows.append({
                    "RULE_ID": rule.rule_id, "REC_DATE": rec_date,
                    "종목코드": code, "종목명": name_s.get(i, ""), **ev,
                })
        day_rows.append(day)
    cand = pd.DataFrame(cand_rows)
    days = pd.DataFrame(day_rows).sort_values("REC_DATE") if day_rows else pd.DataFrame(
        columns=["REC_DATE", "NO_BUY_DAY", "OFFICIAL_BUY_N"] + [r.rule_id for r in RULES])
    return cand, days


def summarize_rules(cand: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for rule in RULES:
        sub = cand[cand["RULE_ID"] == rule.rule_id].copy() if not cand.empty else pd.DataFrame()
        candidate_n = int(len(sub))
        if candidate_n:
            realized = sub[(sub["FILL_STATUS"] == "FILLED") & sub["RET_5D"].notna()].sort_values("REC_DATE")
            pending_n = int((sub["FILL_STATUS"] == "PENDING").sum())
            unfilled_n = int((sub["FILL_STATUS"] == "UNFILLED").sum())
        else:
            realized = pd.DataFrame(columns=["REC_DATE", "RET_5D", "MDD_5D", "ALPHA_5D"])
            pending_n = unfilled_n = 0
        n = int(len(realized))
        ret = pd.to_numeric(realized.get("RET_5D", pd.Series(dtype=float)), errors="coerce")
        mdd = pd.to_numeric(realized.get("MDD_5D", pd.Series(dtype=float)), errors="coerce")
        alpha = pd.to_numeric(realized.get("ALPHA_5D", pd.Series(dtype=float)), errors="coerce").dropna()
        win = float((ret > 0).mean() * 100) if n else 0.0
        avg = float(ret.mean()) if n else 0.0
        med = float(ret.median()) if n else 0.0
        alpha_avg = float(alpha.mean()) if len(alpha) else float("nan")
        stop_risk = float((mdd <= STOP_RISK_MDD_PCT).mean() * 100) if n else 0.0
        loss_avg = float(mdd.mean()) if n else 0.0
        recent = ret.tail(RECENT_WINDOW)
        recent_avg = float(recent.mean()) if len(recent) else 0.0
        # OOS = 실현표본 후반부 (rec_date 순 절반)
        if n >= 2:
            oos = ret.iloc[n // 2:]
            oos_n = int(len(oos)); oos_win = float((oos > 0).mean() * 100); oos_avg = float(oos.mean())
        else:
            oos_n, oos_win, oos_avg = 0, 0.0, 0.0
        realized_frac = (n / candidate_n) if candidate_n else 0.0
        denom = n + unfilled_n
        fill_rate = (n / denom * 100.0) if denom else 0.0

        decision, reason = "PASS_PRODUCTION_GATE", ""
        if n < MIN_N:
            decision, reason = "REJECT_INSUFFICIENT_SAMPLE", "INSUFFICIENT_SAMPLE"
        elif win < MIN_WIN_RATE_5D:
            decision, reason = "REJECT_LOW_WIN_RATE", "LOW_WIN_RATE"
        elif avg <= MIN_AVG_RET_5D:
            decision, reason = "REJECT_NEGATIVE_RETURN", "NEGATIVE_RETURN"
        elif not np.isfinite(alpha_avg):
            decision, reason = "REJECT_NO_ALPHA_DATA", "NO_ALPHA_DATA"
        elif alpha_avg <= MIN_AVG_ALPHA_5D:
            decision, reason = "REJECT_NEGATIVE_ALPHA", "NEGATIVE_ALPHA"
        elif stop_risk > MAX_STOP_RISK_RATE:
            decision, reason = "REJECT_HIGH_STOP_RISK", "HIGH_STOP_RISK"
        elif loss_avg < MIN_MAX_LOSS_AVG:
            decision, reason = "REJECT_HIGH_DRAWDOWN", "HIGH_DRAWDOWN"
        elif len(recent) >= 5 and recent_avg <= 0:
            decision, reason = "REJECT_RECENT_WEAKNESS", "RECENT_WEAKNESS"
        elif oos_n < OOS_MIN_N:
            decision, reason = "REJECT_OOS_SAMPLE", "OOS_SAMPLE_SHORT"
        elif oos_win < OOS_MIN_WIN_RATE:
            decision, reason = "REJECT_OOS_WEAK", "OOS_WIN_RATE_LOW"

        rows.append({
            "RULE_ID": rule.rule_id,
            "RULE_DESC": rule.description,
            "CANDIDATE_N": candidate_n,
            "REALIZED_N": n,
            "PENDING_N": pending_n,
            "VALIDATION_COVERAGE": round(realized_frac, 3),
            "N": n,
            "WIN_RATE_5D": round(win, 2),
            "AVG_RET_5D": round(avg, 2),
            "MEDIAN_RET_5D": round(med, 2),
            "AVG_ALPHA_5D": round(alpha_avg, 2) if np.isfinite(alpha_avg) else 0.0,
            "STOP_RISK_RATE": round(stop_risk, 2),
            "MAX_LOSS_AVG": round(loss_avg, 2),
            "RECENT_N": int(len(recent)),
            "RECENT_AVG_RET_5D": round(recent_avg, 2),
            "DECISION": decision,
            "REJECT_REASON": reason,
            "N_UNFILLED": unfilled_n,
            "FILL_RATE_PCT": round(fill_rate, 1),
            "OOS_N": oos_n,
            "OOS_WIN_RATE_5D": round(oos_win, 1),
            "OOS_AVG_RET_5D": round(oos_avg, 2),
            "ALPHA_N": int(len(alpha)),
            "WINDOW_START": int(realized["REC_DATE"].min()) if n else 0,
            "WINDOW_END": int(realized["REC_DATE"].max()) if n else 0,
            "ENGINE": ENGINE_TAG,
        })
    return pd.DataFrame(rows, columns=RULE_SCHEMA_COLS)


def run_backtest(data_dir: str = "data", out_dir: str = "data") -> Dict[str, object]:
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in data_path.glob("recommend_*.csv")
                   if "latest" not in p.name and _DATE_RE.search(p.name))

    lut = load_ohlcv_lut(data_path)
    kospi_fwd = load_kospi_fwd(data_path)
    candidates, days = collect_and_evaluate(paths, lut, kospi_fwd)
    summary = summarize_rules(candidates)

    realized_mask = (candidates.get("FILL_STATUS", pd.Series(dtype=str)) == "FILLED") \
        if not candidates.empty else pd.Series(dtype=bool)
    realized_df = candidates[realized_mask & candidates["RET_5D"].notna()] \
        if not candidates.empty else candidates

    candidates_path = out_path / "no_buy_breaker_candidates_latest.csv"
    trades_path = out_path / "no_buy_breaker_trades_latest.csv"
    rules_path = out_path / "no_buy_breaker_rules_latest.csv"
    days_path = out_path / "no_buy_breaker_days_latest.csv"
    direct_path = out_path / "no_buy_breaker_direct_outcomes_latest.csv"
    json_path = out_path / "no_buy_breaker_backtest_latest.json"

    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    realized_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    summary.to_csv(rules_path, index=False, encoding="utf-8-sig")
    days.to_csv(days_path, index=False, encoding="utf-8-sig")
    realized_df.to_csv(direct_path, index=False, encoding="utf-8-sig")

    pending_rows = int((candidates.get("FILL_STATUS", pd.Series(dtype=str)) == "PENDING").sum()) \
        if not candidates.empty else 0
    payload = {
        "version": "25.0.0",
        "engine": ENGINE_TAG,
        "data_dir": str(data_path),
        "files": len(paths),
        "no_buy_days": int(days["NO_BUY_DAY"].sum()) if not days.empty and "NO_BUY_DAY" in days.columns else 0,
        "candidate_rows": int(len(candidates)),
        "realized_rows": int(len(realized_df)),
        "pending_rows": pending_rows,
        "unfilled_rows": int((candidates.get("FILL_STATUS", pd.Series(dtype=str)) == "UNFILLED").sum()) if not candidates.empty else 0,
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
    parser = argparse.ArgumentParser(description="No-Buy Breaker fill-aware certifier v25.0")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    payload = run_backtest(args.data_dir, args.out_dir)
    passed = [r for r in payload.get("rules", []) if r.get("DECISION") == "PASS_PRODUCTION_GATE"]
    print(
        f"[v25.0 {ENGINE_TAG}] cert complete: files={payload['files']} "
        f"no_buy_days={payload['no_buy_days']} candidates={payload['candidate_rows']} "
        f"realized={payload['realized_rows']} unfilled={payload['unfilled_rows']} "
        f"pending={payload['pending_rows']} pass_rules={len(passed)}"
    )


if __name__ == "__main__":
    main()
