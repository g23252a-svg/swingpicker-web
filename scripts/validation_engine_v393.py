# -*- coding: utf-8 -*-
"""Validation Engine v3.9.3 — No-Buy Decision & Shadow Candidate Validation.

Measurement-only validation layer for the recommendation system v3.9.24 columns.

Goals:
- Verify whether official no-buy days were defensive successes or overly conservative.
- Validate candidate triage buckets such as ENTRY_CLEAN_OBSERVE / HIGH_SCORE_OBSERVE.
- Validate shadow relaxation flags without promoting them to official buy signals.

Safety:
- Does not change TOP_PICK, BUY_NOW_ELIGIBLE, BUY_NOW_PASS, scores, entries, stops, or targets.
- Reads recommend_YYYYMMDD.csv and backtest_top1/top3_trades_*.csv only.
- Writes validation reports under data/ by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger("validation_engine_v393")

_DATE_RE = re.compile(r"(20\d{6})")
RET_CANDIDATE_COLS = [
    "net_pct",
    "ret_pct",
    "ret_5d_%",
    "ret_5d",
    "RETURN_5D_%",
    "RETURN_5D",
]

TRIAGE_TYPES = {
    "OFFICIAL_BUY",
    "ENTRY_CLEAN_OBSERVE",
    "HIGH_SCORE_OBSERVE",
    "CHASE_RISK",
    "HOLDING_MANAGE",
    "IGNORE",
}

SHADOW_FLAG_COLS = [
    "SHADOW_MACRO_RELAXED_ELIGIBLE",
    "SHADOW_ENTRY_RELAXED_ELIGIBLE",
    "SHADOW_SCORE_RELAXED_ELIGIBLE",
    "MACRO_RELAXED_SHADOW_PASS",
]


def _date_from_name(path: Path) -> str | None:
    m = _DATE_RE.search(path.name)
    return m.group(1) if m else None


def _norm_code(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    return digits.zfill(6) if digits else ""


def _truthy(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "pass", "eligible", "buy"}


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError) as exc:
        logger.debug("float 변환 실패: %r (%s)", value, exc)
        return default


def _read_csv(path: Path) -> pd.DataFrame | None:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, dtype={"종목코드": str, "code": str}, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            logger.warning("CSV 로드 실패: %s (%s)", path, exc)
            return None
    logger.warning("CSV 인코딩 판독 실패: %s", path)
    return None


def _recommend_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in data_dir.glob("recommend_*.csv"):
        if "latest" in p.name.lower():
            continue
        if _date_from_name(p) is None:
            continue
        files.append(p)
    return sorted(files)


def _trade_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pat in ("backtest_top1_trades_*.csv", "backtest_top3_trades_*.csv"):
        for p in data_dir.glob(pat):
            if "latest" in p.name.lower():
                continue
            files.append(p)
    return sorted(files)


def _first_existing(row: pd.Series, names: Iterable[str], default=None):
    for name in names:
        if name in row.index:
            val = row.get(name)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                return val
    return default


def _choose_ret_col(df: pd.DataFrame) -> str | None:
    for col in RET_CANDIDATE_COLS:
        if col in df.columns:
            return col
    return None


def _infer_triage_type(row: pd.Series) -> str:
    existing = str(row.get("CANDIDATE_TRIAGE_TYPE", "") or "").strip().upper()
    if existing in TRIAGE_TYPES:
        return existing

    top_pick = _truthy(row.get("TOP_PICK", 0))
    eligible = _truthy(row.get("BUY_NOW_ELIGIBLE", 0))
    buy_now_pass = _truthy(row.get("BUY_NOW_PASS", 0))
    final_score = _safe_float(_first_existing(row, ["FINAL_SCORE", "DISPLAY_SCORE"]), 0.0) or 0.0
    elite_score = _safe_float(row.get("ELITE_SCORE"), 0.0) or 0.0
    gap_val = _safe_float(_first_existing(row, ["ENTRY_GAP_PCT", "GAP_PCT"]), 999.0)
    gap_pct = abs(999.0 if gap_val is None else gap_val)
    vwap_val = _safe_float(row.get("VWAP_GAP"), 0.0)
    vwap_gap = abs(0.0 if vwap_val is None else vwap_val)
    poc_val = _safe_float(row.get("POC_GAP"), 0.0)
    poc_gap = abs(0.0 if poc_val is None else poc_val)
    rr_val = _safe_float(row.get("RR_NOW_TP1"), 0.0)
    rr = 0.0 if rr_val is None else rr_val
    route = str(row.get("ROUTE", "") or "").strip().upper()
    holding_flag = _truthy(row.get("IS_HOLDING", 0)) or _truthy(row.get("HOLDING_MANAGE_FLAG", 0))

    if top_pick and eligible:
        return "OFFICIAL_BUY"
    if holding_flag:
        return "HOLDING_MANAGE"

    # 현재 가격 위치가 깨끗한 관찰 후보. v3.9.24 컬럼이 없는 legacy CSV도 진단 가능하게 fallback 제공.
    if buy_now_pass and gap_pct <= 3.0 and vwap_gap <= 15.0 and poc_gap <= 25.0 and rr >= 1.10:
        return "ENTRY_CLEAN_OBSERVE"

    # 점수는 높지만 진입 위치/BUY_NOW 조건이 막힌 후보.
    if final_score >= 75.0 or elite_score >= 70.0:
        if gap_pct > 5.0 or vwap_gap > 20.0 or poc_gap > 40.0 or rr < 1.10:
            return "CHASE_RISK"
        return "HIGH_SCORE_OBSERVE"

    if route in {"ATTACK", "ARMED"} and (gap_pct > 5.0 or vwap_gap > 20.0 or poc_gap > 40.0 or rr < 1.10):
        return "CHASE_RISK"

    return "IGNORE"


def _infer_funnel_stage(row: pd.Series, triage_type: str) -> str:
    existing = str(row.get("OFFICIAL_FUNNEL_STAGE", "") or "").strip()
    if existing:
        return existing

    top_pick = _truthy(row.get("TOP_PICK", 0))
    eligible = _truthy(row.get("BUY_NOW_ELIGIBLE", 0))
    buy_now_pass = _truthy(row.get("BUY_NOW_PASS", 0))

    if top_pick and eligible:
        return "OFFICIAL_BUY"
    if top_pick and not eligible:
        return "TOP_PICK_BUT_NOT_BUY_NOW_ELIGIBLE"
    if triage_type == "ENTRY_CLEAN_OBSERVE" or buy_now_pass:
        return "ENTRY_READY_BUT_NOT_TOP_PICK"
    if triage_type == "HIGH_SCORE_OBSERVE":
        return "HIGH_SCORE_BUT_ENTRY_BLOCKED"
    if triage_type == "CHASE_RISK":
        return "ROUTE_ACTIVE_BUT_CHASE_RISK"
    if triage_type == "HOLDING_MANAGE":
        return "HOLDING_MANAGE_NOT_NEW_BUY"
    return "NOT_OFFICIAL_CANDIDATE"


def _reason_from_row(row: pd.Series, triage_type: str) -> tuple[str, str]:
    r1 = str(row.get("OFFICIAL_BLOCK_REASON_1", "") or "").strip()
    r2 = str(row.get("OFFICIAL_BLOCK_REASON_2", "") or "").strip()
    if r1 or r2:
        return r1 or "-", r2 or "-"

    if triage_type == "OFFICIAL_BUY":
        return "공식 신규매수", "TOP_PICK + BUY_NOW_ELIGIBLE"
    if triage_type == "ENTRY_CLEAN_OBSERVE":
        return "TOP_PICK=0", "진입 위치는 양호하나 공식 Top Pick 아님"
    if triage_type == "HIGH_SCORE_OBSERVE":
        return "BUY_NOW_PASS=0 또는 진입 위치 불량", "고점수 관찰 후보"
    if triage_type == "CHASE_RISK":
        return "추격/손익비 위험", "GAP/VWAP/POC/RR 조건 확인 필요"
    if triage_type == "HOLDING_MANAGE":
        return "신규매수 아님", "보유관리 관점으로 분리"
    return "공식 조건 미충족", "관찰 우선순위 낮음"


def load_recommend_snapshots(data_dir: str | Path) -> pd.DataFrame:
    """Load recommend_YYYYMMDD.csv rows and normalize v3.9.24 validation columns."""
    data_path = Path(data_dir)
    frames: list[pd.DataFrame] = []

    for path in _recommend_files(data_path):
        signal_date = _date_from_name(path)
        if not signal_date:
            continue
        df = _read_csv(path)
        if df is None or df.empty:
            continue

        code_col = "종목코드" if "종목코드" in df.columns else "code" if "code" in df.columns else None
        if code_col is None:
            logger.debug("종목코드 컬럼 없음: %s", path)
            continue

        tmp = df.copy()
        tmp["signal_date"] = signal_date
        tmp["code"] = tmp[code_col].map(_norm_code)
        tmp["name"] = tmp.get("종목명", tmp.get("name", "")).astype(str)
        tmp["recommend_file"] = path.name
        tmp = tmp[tmp["code"].astype(bool)].copy()
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["official_buy_signal"] = out.apply(
        lambda r: _truthy(r.get("TOP_PICK", 0)) and _truthy(r.get("BUY_NOW_ELIGIBLE", 0)), axis=1
    )
    out["CANDIDATE_TRIAGE_TYPE"] = out.apply(_infer_triage_type, axis=1)
    out["OFFICIAL_FUNNEL_STAGE"] = out.apply(
        lambda r: _infer_funnel_stage(r, str(r.get("CANDIDATE_TRIAGE_TYPE", ""))), axis=1
    )
    reasons = out.apply(lambda r: _reason_from_row(r, str(r.get("CANDIDATE_TRIAGE_TYPE", ""))), axis=1)
    out["OFFICIAL_BLOCK_REASON_1"] = [a for a, _ in reasons]
    out["OFFICIAL_BLOCK_REASON_2"] = [b for _, b in reasons]

    for col in SHADOW_FLAG_COLS:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].map(_truthy)

    return out


def load_realized_trades(data_dir: str | Path) -> pd.DataFrame:
    """Load realized backtest rows by signal_date+code."""
    data_path = Path(data_dir)
    frames: list[pd.DataFrame] = []

    for path in _trade_files(data_path):
        df = _read_csv(path)
        if df is None or df.empty:
            continue
        if "date" not in df.columns or "code" not in df.columns:
            logger.debug("date/code 컬럼 없음: %s", path)
            continue
        tmp = df.copy()
        tmp["signal_date"] = tmp["date"].astype(str).str.replace("-", "", regex=False).str[:8]
        tmp["code"] = tmp["code"].map(_norm_code)
        tmp["trade_source"] = path.name
        ret_col = _choose_ret_col(tmp)
        if ret_col is not None:
            tmp["realized_ret_pct"] = tmp[ret_col].map(lambda x: _safe_float(x))
        else:
            tmp["realized_ret_pct"] = None
        tmp["tp1_hit_bool"] = tmp.get("tp1_hit", pd.Series(False, index=tmp.index)).map(_truthy)
        tmp["stop_hit_bool"] = tmp.get("stop_hit", pd.Series(False, index=tmp.index)).map(_truthy)
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(
            columns=["signal_date", "code", "realized_ret_pct", "tp1_hit_bool", "stop_hit_bool", "trade_source"]
        )

    out = pd.concat(frames, ignore_index=True)
    out["_has_result"] = out["realized_ret_pct"].notna().astype(int)
    out = out.sort_values(["signal_date", "code", "_has_result"], ascending=[True, True, False])
    out = out.drop_duplicates(["signal_date", "code"], keep="first")
    cols = ["signal_date", "code", "realized_ret_pct", "tp1_hit_bool", "stop_hit_bool", "trade_source"]
    return out[cols].drop(columns=[c for c in [] if c in out.columns])


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _median(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return None
    return round(float(pd.Series(vals).median()), 4)


def _win_rate(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return None
    return round(sum(v > 0 for v in vals) / len(vals) * 100.0, 2)


def _rate(flags: Iterable[bool]) -> float | None:
    vals = [bool(v) for v in flags]
    if not vals:
        return None
    return round(sum(vals) / len(vals) * 100.0, 2)


def _no_buy_grade(entry_clean_n: int, entry_clean_avg: float | None, all_n: int, all_avg: float | None) -> str:
    if all_n < 3:
        return "DATA_INSUFFICIENT"
    if entry_clean_n >= 2 and entry_clean_avg is not None and entry_clean_avg >= 2.0:
        return "TOO_CONSERVATIVE_WARNING"
    if all_avg is not None and all_avg <= -1.0:
        return "DEFENSIVE_SUCCESS"
    return "INCONCLUSIVE"


def summarize_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ret_vals = g["realized_ret_pct"] if "realized_ret_pct" in g.columns else pd.Series(dtype=float)
        has_result = ret_vals.notna()
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "N": int(len(g)),
                "RESULT_N": int(has_result.sum()),
                "WIN_RATE_%": _win_rate(ret_vals),
                "AVG_RET_%": _mean(ret_vals),
                "MEDIAN_RET_%": _median(ret_vals),
                "TP1_HIT_RATE_%": _rate(g.loc[has_result, "tp1_hit_bool"]) if "tp1_hit_bool" in g.columns else None,
                "STOP_HIT_RATE_%": _rate(g.loc[has_result, "stop_hit_bool"]) if "stop_hit_bool" in g.columns else None,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_no_buy_decision_validation(row_df: pd.DataFrame) -> pd.DataFrame:
    """Validate official no-buy days using realized performance of near-miss candidates."""
    if row_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for signal_date, g in row_df.groupby("signal_date", dropna=False):
        official_count = int(g["official_buy_signal"].map(bool).sum())
        if official_count > 0:
            continue

        near = g[g["CANDIDATE_TRIAGE_TYPE"].isin(["ENTRY_CLEAN_OBSERVE", "HIGH_SCORE_OBSERVE", "CHASE_RISK"])]
        entry = near[near["CANDIDATE_TRIAGE_TYPE"] == "ENTRY_CLEAN_OBSERVE"]
        high = near[near["CANDIDATE_TRIAGE_TYPE"] == "HIGH_SCORE_OBSERVE"]
        chase = near[near["CANDIDATE_TRIAGE_TYPE"] == "CHASE_RISK"]
        all_ret = near["realized_ret_pct"] if "realized_ret_pct" in near.columns else pd.Series(dtype=float)
        entry_ret = entry["realized_ret_pct"] if "realized_ret_pct" in entry.columns else pd.Series(dtype=float)
        all_n = int(all_ret.notna().sum())
        entry_n = int(entry_ret.notna().sum())
        all_avg = _mean(all_ret)
        entry_avg = _mean(entry_ret)

        rows.append(
            {
                "SIGNAL_DATE": signal_date,
                "OFFICIAL_BUY_COUNT": official_count,
                "NEAR_MISS_COUNT": int(len(near)),
                "RESULT_N": all_n,
                "ENTRY_CLEAN_N": int(len(entry)),
                "ENTRY_CLEAN_RESULT_N": entry_n,
                "ENTRY_CLEAN_AVG_RET_%": entry_avg,
                "HIGH_SCORE_N": int(len(high)),
                "HIGH_SCORE_AVG_RET_%": _mean(high["realized_ret_pct"]) if "realized_ret_pct" in high.columns else None,
                "CHASE_RISK_N": int(len(chase)),
                "CHASE_RISK_AVG_RET_%": _mean(chase["realized_ret_pct"]) if "realized_ret_pct" in chase.columns else None,
                "NEAR_MISS_AVG_RET_%": all_avg,
                "NEAR_MISS_WIN_RATE_%": _win_rate(all_ret),
                "NO_BUY_DECISION_GRADE": _no_buy_grade(entry_n, entry_avg, all_n, all_avg),
            }
        )

    return pd.DataFrame(rows).sort_values("SIGNAL_DATE").reset_index(drop=True) if rows else pd.DataFrame()


def _shadow_hint(result_n: int, avg_ret: float | None, win_rate: float | None) -> str:
    if result_n < 5:
        return "NEED_MORE_N"
    if avg_ret is not None and avg_ret >= 2.0 and (win_rate is None or win_rate >= 55.0):
        return "PROMOTION_CANDIDATE"
    if avg_ret is not None and avg_ret <= -1.5:
        return "REJECT_SHADOW"
    return "KEEP_SHADOW"


def build_shadow_candidate_validation(row_df: pd.DataFrame) -> pd.DataFrame:
    """Validate v3.9.24 shadow relaxed flags without changing production decisions."""
    if row_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for flag in SHADOW_FLAG_COLS:
        if flag not in row_df.columns:
            continue
        subset = row_df[row_df[flag].map(bool)].copy()
        ret_vals = subset["realized_ret_pct"] if "realized_ret_pct" in subset.columns else pd.Series(dtype=float)
        result_n = int(ret_vals.notna().sum())
        avg_ret = _mean(ret_vals)
        win = _win_rate(ret_vals)
        rows.append(
            {
                "SHADOW_FLAG": flag,
                "N": int(len(subset)),
                "RESULT_N": result_n,
                "WIN_RATE_%": win,
                "AVG_RET_%": avg_ret,
                "MEDIAN_RET_%": _median(ret_vals),
                "TP1_HIT_RATE_%": _rate(subset.loc[ret_vals.notna(), "tp1_hit_bool"]) if "tp1_hit_bool" in subset.columns else None,
                "STOP_HIT_RATE_%": _rate(subset.loc[ret_vals.notna(), "stop_hit_bool"]) if "stop_hit_bool" in subset.columns else None,
                "SHADOW_PROMOTION_HINT": _shadow_hint(result_n, avg_ret, win),
            }
        )

    return pd.DataFrame(rows)


def build_validation_engine_v393(data_dir: str | Path = "data", out_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build row-level, no-buy, and shadow validation reports.

    Returns:
        (row_level_df, no_buy_df, shadow_df, summary_payload)
    """
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    snapshots = load_recommend_snapshots(data_path)
    trades = load_realized_trades(data_path)

    if snapshots.empty:
        row_level = pd.DataFrame()
    else:
        row_level = snapshots.merge(trades, on=["signal_date", "code"], how="left", suffixes=("", "_trade"))
        row_level["HAS_RESULT"] = row_level["realized_ret_pct"].notna() if "realized_ret_pct" in row_level.columns else False
        row_level["IS_WIN"] = row_level["realized_ret_pct"].map(lambda x: bool(x > 0) if pd.notna(x) else False)

    no_buy = build_no_buy_decision_validation(row_level)
    shadow = build_shadow_candidate_validation(row_level)
    triage_summary = summarize_by_group(row_level, ["CANDIDATE_TRIAGE_TYPE"]) if not row_level.empty else pd.DataFrame()
    funnel_summary = summarize_by_group(row_level, ["OFFICIAL_FUNNEL_STAGE"]) if not row_level.empty else pd.DataFrame()

    asof = None
    if not snapshots.empty and "signal_date" in snapshots.columns:
        asof = str(snapshots["signal_date"].max())
    if asof is None:
        asof = datetime.now().strftime("%Y%m%d")

    if not row_level.empty:
        row_level.to_csv(out_path / "validation_engine_v393_latest.csv", index=False, encoding="utf-8-sig")
    no_buy.to_csv(out_path / "no_buy_decision_validation_latest.csv", index=False, encoding="utf-8-sig")
    shadow.to_csv(out_path / "shadow_candidate_validation_latest.csv", index=False, encoding="utf-8-sig")
    if not triage_summary.empty:
        triage_summary.to_csv(out_path / "candidate_triage_validation_latest.csv", index=False, encoding="utf-8-sig")
    if not funnel_summary.empty:
        funnel_summary.to_csv(out_path / "official_funnel_validation_latest.csv", index=False, encoding="utf-8-sig")

    grade_counts = no_buy["NO_BUY_DECISION_GRADE"].value_counts(dropna=False).to_dict() if not no_buy.empty else {}
    shadow_hints = shadow["SHADOW_PROMOTION_HINT"].value_counts(dropna=False).to_dict() if not shadow.empty else {}
    summary = {
        "version": "v3.9.3",
        "asof": asof,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_rows": int(len(row_level)),
        "realized_result_rows": int(row_level["HAS_RESULT"].sum()) if not row_level.empty and "HAS_RESULT" in row_level.columns else 0,
        "no_buy_days": int(len(no_buy)),
        "no_buy_grade_counts": {str(k): int(v) for k, v in grade_counts.items()},
        "shadow_hint_counts": {str(k): int(v) for k, v in shadow_hints.items()},
        "notes": [
            "Measurement-only validation engine.",
            "Does not change TOP_PICK / BUY_NOW_ELIGIBLE / BUY_NOW_PASS / scoring formulas.",
            "Validates CANDIDATE_TRIAGE_TYPE, OFFICIAL_FUNNEL_STAGE and SHADOW_* columns when present.",
        ],
    }
    with open(out_path / "validation_engine_v393_latest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return row_level, no_buy, shadow, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation Engine v3.9.3 reports")
    parser.add_argument("--data-dir", default="data", help="Directory containing recommend/backtest CSV files")
    parser.add_argument("--out-dir", default="data", help="Directory to write validation reports")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    row_level, no_buy, shadow, summary = build_validation_engine_v393(args.data_dir, args.out_dir)
    print("[v3.9.3] Validation Engine reports generated")
    print(f"- rows: {len(row_level)}")
    print(f"- no_buy_days: {len(no_buy)}")
    print(f"- shadow_flags: {len(shadow)}")
    print(f"- asof: {summary.get('asof')}")


if __name__ == "__main__":
    main()
