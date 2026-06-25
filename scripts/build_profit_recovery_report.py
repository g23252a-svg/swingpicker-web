# -*- coding: utf-8 -*-
"""Build v3.9.29 Profit Recovery Suite diagnostics from current recommend CSV.

Usage:
    python scripts/build_profit_recovery_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_finalize import (
    add_entry_risk_columns,
    add_entry_edge_columns,
    add_july_profit_defense_columns,
    add_profit_recovery_suite_columns,
    finalize_sort,
)

DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def _load_current() -> pd.DataFrame:
    path = DATA / "recommend_latest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"종목코드": str})
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    return df


def _apply_layers(df: pd.DataFrame) -> pd.DataFrame:
    out = add_entry_risk_columns(df)
    out = add_entry_edge_columns(out)
    out = add_july_profit_defense_columns(out, enforce=True)
    out = add_profit_recovery_suite_columns(out, enforce=True)
    out = finalize_sort(out)
    out["LDY_RANK"] = range(1, len(out) + 1)
    return out


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = _apply_layers(_load_current())

    wanted = [
        "LDY_RANK", "종목코드", "종목명", "시장", "업종_대분류", "종가", "ROUTE", "상태",
        "TOP_PICK", "BUY_NOW_ELIGIBLE", "BUY_NOW_GRADE", "DISPLAY_SCORE", "FINAL_SCORE",
        "STRUCT_SCORE", "TIMING_SCORE", "ELITE_SCORE", "RR_NOW_TP1", "ENTRY_GAP_PCT",
        "ret_1d_%", "ret_5d_%", "VWAP_GAP", "POC_GAP", "Vol_Quality", "MARKET_BREADTH",
        "JULY_PROFIT_DEFENSE_SCORE", "JULY_PROFIT_DEFENSE_LEVEL", "JULY_PROFIT_BLOCK_FLAG",
        "PROFIT_RECOVERY_SCORE", "PROFIT_RECOVERY_TIER", "PROFIT_RECOVERY_SETUP",
        "PROFIT_RECOVERY_BLOCK_FLAG", "PROFIT_RECOVERY_SIZE_MULT", "PROFIT_RECOVERY_ACTION",
        "PROFIT_RECOVERY_REASON", "BUY_NOW_REASON",
    ]
    cols = [c for c in wanted if c in df.columns]
    top = df[cols].head(50).copy()
    top.to_csv(REPORTS / "profit_recovery_current_top50.csv", index=False, encoding="utf-8-sig")

    official = int(((pd.to_numeric(df.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int) == 1)
                    & (pd.to_numeric(df.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0).astype(int) == 1)).sum())
    summary = {
        "rows": int(len(df)),
        "official_new_buy_count": official,
        "profit_recovery_tier_counts": df.get("PROFIT_RECOVERY_TIER", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        "profit_recovery_setup_counts": df.get("PROFIT_RECOVERY_SETUP", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
        "profit_recovery_block_count": int(pd.to_numeric(df.get("PROFIT_RECOVERY_BLOCK_FLAG", 0), errors="coerce").fillna(0).astype(int).sum()),
        "july_block_count": int(pd.to_numeric(df.get("JULY_PROFIT_BLOCK_FLAG", 0), errors="coerce").fillna(0).astype(int).sum()),
        "top10": top[[c for c in ["LDY_RANK", "종목코드", "종목명", "PROFIT_RECOVERY_SCORE", "PROFIT_RECOVERY_TIER", "PROFIT_RECOVERY_ACTION"] if c in top.columns]].head(10).to_dict("records"),
    }
    (REPORTS / "profit_recovery_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
