# -*- coding: utf-8 -*-
"""v22.3.12 Official Buy Validation.

공식 신규매수 기준(TOP_PICK + BUY_NOW_ELIGIBLE)의 실제 성과와,
TOP_PICK이지만 BUY_NOW_ELIGIBLE=0으로 보류한 종목의 이후 결과를
별도 데이터셋으로 누적한다.

Measurement-only script:
- BUY_NOW_ELIGIBLE / TOP_PICK / 점수 산식은 변경하지 않는다.
- recommend_YYYYMMDD.csv와 backtest_top1/top3_trades_*.csv를 날짜+종목코드로 조인한다.
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

logger = logging.getLogger("official_buy_validation")

_DATE_RE = re.compile(r"(20\d{6})")


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
    return s in {"1", "true", "t", "yes", "y", "pass", "eligible"}


def _safe_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError) as exc:
        logger.debug("float 변환 실패: %r (%s)", value, exc)
        return None


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
    files = []
    for p in data_dir.glob("recommend_*.csv"):
        if "latest" in p.name.lower():
            continue
        if _date_from_name(p) is None:
            continue
        files.append(p)
    return sorted(files)


def load_top_pick_signals(data_dir: str | Path) -> pd.DataFrame:
    """recommend_YYYYMMDD.csv에서 TOP_PICK 신호와 보류 신호를 추출."""
    data_path = Path(data_dir)
    rows: list[dict] = []

    for path in _recommend_files(data_path):
        signal_date = _date_from_name(path)
        if not signal_date:
            continue
        df = _read_csv(path)
        if df is None or df.empty:
            continue
        if "TOP_PICK" not in df.columns:
            logger.debug("TOP_PICK 컬럼 없음: %s", path)
            continue

        code_col = "종목코드" if "종목코드" in df.columns else "code" if "code" in df.columns else None
        name_col = "종목명" if "종목명" in df.columns else "name" if "name" in df.columns else None
        if code_col is None:
            logger.debug("종목코드 컬럼 없음: %s", path)
            continue

        for _, r in df[df["TOP_PICK"].map(_truthy)].iterrows():
            code = _norm_code(r.get(code_col))
            if not code:
                continue
            eligible = _truthy(r.get("BUY_NOW_ELIGIBLE", 0))
            rows.append(
                {
                    "signal_date": signal_date,
                    "code": code,
                    "name": str(r.get(name_col, "") or ""),
                    "top_pick": True,
                    "official_buy_signal": bool(eligible),
                    "top_pick_holdout": not bool(eligible),
                    "buy_now_eligible": bool(eligible),
                    "buy_now_grade": str(r.get("BUY_NOW_GRADE", "") or ""),
                    "buy_now_score": _safe_float(r.get("BUY_NOW_SCORE")),
                    "final_score": _safe_float(r.get("FINAL_SCORE")),
                    "display_score": _safe_float(r.get("DISPLAY_SCORE")),
                    "elite_score": _safe_float(r.get("ELITE_SCORE")),
                    "route": str(r.get("ROUTE", "") or ""),
                    "rr_now_tp1": _safe_float(r.get("RR_NOW_TP1")),
                    "entry_gap_pct": _safe_float(r.get("ENTRY_GAP_PCT", r.get("GAP_PCT"))),
                    "vwap_gap": _safe_float(r.get("VWAP_GAP")),
                    "poc_gap": _safe_float(r.get("POC_GAP")),
                    "entry_edge_level": str(r.get("ENTRY_EDGE_LEVEL", "") or ""),
                    "recommend_file": path.name,
                }
            )

    return pd.DataFrame(rows)


def _trade_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pat in ("backtest_top1_trades_*.csv", "backtest_top3_trades_*.csv"):
        for p in data_dir.glob(pat):
            if "latest" in p.name.lower():
                # latest는 날짜별 파일과 중복될 가능성이 높아서 제외
                continue
            files.append(p)
    return sorted(files)


def load_realized_trades(data_dir: str | Path) -> pd.DataFrame:
    """backtest_top1/top3 trades에서 날짜+종목코드별 실현 결과를 로드."""
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
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(
            columns=["signal_date", "code", "outcome", "net_pct", "ret_pct", "days_held", "trade_source"]
        )

    out = pd.concat(frames, ignore_index=True)
    keep_cols = [
        c
        for c in [
            "signal_date",
            "code",
            "outcome",
            "net_pct",
            "ret_pct",
            "days_held",
            "fill_date",
            "exit_price",
            "tp1_hit",
            "stop_hit",
            "trade_source",
        ]
        if c in out.columns
    ]
    out = out[keep_cols].copy()
    out["net_pct"] = out.get("net_pct", pd.Series(dtype=float)).map(_safe_float)
    out["ret_pct"] = out.get("ret_pct", pd.Series(dtype=float)).map(_safe_float)

    # 같은 날짜+종목이 Top1/Top3 또는 여러 검증 파일에 중복될 수 있어 첫 결과만 사용.
    # 최신 파일 중복보다 날짜별 파일을 우선하되, net_pct가 있는 행을 우선한다.
    out["_has_result"] = out["net_pct"].notna().astype(int)
    out = out.sort_values(["signal_date", "code", "_has_result"], ascending=[True, True, False])
    out = out.drop_duplicates(["signal_date", "code"], keep="first")
    return out.drop(columns=["_has_result"])


def _mean_pct(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _win_rate(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return None
    return round(sum(v > 0 for v in vals) / len(vals) * 100.0, 2)


def build_official_buy_validation(data_dir: str | Path, out_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """공식 신규매수/보류 TOP_PICK 검증 테이블과 요약 JSON 생성."""
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    signals = load_top_pick_signals(data_path)
    trades = load_realized_trades(data_path)

    if signals.empty:
        result = pd.DataFrame()
    else:
        result = signals.merge(trades, on=["signal_date", "code"], how="left", suffixes=("", "_trade"))
        result["has_result"] = result["net_pct"].notna()
        result["is_win"] = result["net_pct"].map(lambda x: bool(x > 0) if pd.notna(x) else False)
        result["cash_vs_top_pick_pct"] = result.apply(
            lambda r: round(-float(r["net_pct"]), 4)
            if bool(r.get("top_pick_holdout")) and pd.notna(r.get("net_pct"))
            else None,
            axis=1,
        )
        result["cash_verdict"] = result.apply(_cash_verdict, axis=1)
        result["official_verdict"] = result.apply(_official_verdict, axis=1)

    summary = summarize_validation(result)
    asof = None
    if not signals.empty:
        asof = str(signals["signal_date"].max())
    if asof is None:
        asof = datetime.now().strftime("%Y%m%d")

    payload = {
        "version": "22.3.12",
        "asof": asof,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "notes": [
            "official_buy_signal = TOP_PICK + BUY_NOW_ELIGIBLE",
            "top_pick_holdout = TOP_PICK=1 and BUY_NOW_ELIGIBLE=0",
            "cash_vs_top_pick_pct = 0% cash return - holdout TOP_PICK net_pct",
            "measurement-only: recommendation formulas are not modified",
        ],
    }

    csv_latest = out_path / "official_buy_validation_latest.csv"
    json_latest = out_path / "official_buy_validation_latest.json"
    csv_dated = out_path / f"official_buy_validation_{asof}.csv"
    json_dated = out_path / f"official_buy_validation_{asof}.json"

    result.to_csv(csv_latest, index=False, encoding="utf-8-sig")
    result.to_csv(csv_dated, index=False, encoding="utf-8-sig")
    for p in (json_latest, json_dated):
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return result, payload


def _cash_verdict(row) -> str:
    if not bool(row.get("top_pick_holdout")):
        return "NOT_HOLDOUT"
    net = row.get("net_pct")
    if pd.isna(net):
        return "PENDING"
    if float(net) < 0:
        return "CASH_AVOIDED_LOSS"
    if float(net) > 0:
        return "CASH_OPPORTUNITY_COST"
    return "CASH_NEUTRAL"


def _official_verdict(row) -> str:
    if not bool(row.get("official_buy_signal")):
        return "NOT_OFFICIAL_BUY"
    net = row.get("net_pct")
    if pd.isna(net):
        return "PENDING"
    return "WIN" if float(net) > 0 else "LOSS_OR_FLAT"


def summarize_validation(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "signal_days": 0,
            "top_pick_signals": 0,
            "official_buy_signals": 0,
            "official_buy_results": 0,
            "official_buy_win_rate": None,
            "official_buy_avg_net_pct": None,
            "no_official_buy_days": 0,
            "top_pick_holdout_signals": 0,
            "top_pick_holdout_results": 0,
            "holdout_top_pick_win_rate": None,
            "holdout_top_pick_avg_net_pct": None,
            "cash_vs_top_pick_avg_pct": None,
            "cash_avoided_loss_days": 0,
            "cash_opportunity_cost_days": 0,
        }

    official = df[df["official_buy_signal"].astype(bool)]
    official_done = official[official["net_pct"].notna()]
    holdout = df[df["top_pick_holdout"].astype(bool)]
    holdout_done = holdout[holdout["net_pct"].notna()]

    day_official = df.groupby("signal_date")["official_buy_signal"].sum()
    no_official_buy_days = int((day_official == 0).sum())

    return {
        "signal_days": int(df["signal_date"].nunique()),
        "top_pick_signals": int(len(df)),
        "official_buy_signals": int(len(official)),
        "official_buy_results": int(len(official_done)),
        "official_buy_win_rate": _win_rate(official_done["net_pct"]),
        "official_buy_avg_net_pct": _mean_pct(official_done["net_pct"]),
        "no_official_buy_days": no_official_buy_days,
        "top_pick_holdout_signals": int(len(holdout)),
        "top_pick_holdout_results": int(len(holdout_done)),
        "holdout_top_pick_win_rate": _win_rate(holdout_done["net_pct"]),
        "holdout_top_pick_avg_net_pct": _mean_pct(holdout_done["net_pct"]),
        "cash_vs_top_pick_avg_pct": _mean_pct(holdout_done["cash_vs_top_pick_pct"]),
        "cash_avoided_loss_days": int((holdout_done["net_pct"] < 0).sum()),
        "cash_opportunity_cost_days": int((holdout_done["net_pct"] > 0).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official buy validation tracker")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    df, payload = build_official_buy_validation(args.data_dir, args.out_dir)
    s = payload.get("summary", {})
    print("📌 Official Buy Validation 완료")
    print(f"  공식 신규매수 신호: {s.get('official_buy_signals', 0)}건")
    print(f"  공식 결과 표본: {s.get('official_buy_results', 0)}건")
    print(f"  TOP_PICK 보류 표본: {s.get('top_pick_holdout_results', 0)}건")
    print(f"  현금 vs 보류 TOP_PICK 평균: {s.get('cash_vs_top_pick_avg_pct')}")
    print(f"  rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
