# -*- coding: utf-8 -*-
"""
validation.py — rank_validation + reality_check
═══════════════════════════════════════════════════
[v14] #9 collector.py 분할 — 검증/리포트 전용

참고: make_rank_validation_report (335줄)은 규모가 크므로,
collector.py에서 그대로 유지하고 여기서는 가벼운 래퍼/유틸만 제공.
실제 이전은 점진적으로 진행.
"""
import os
import logging
from typing import Optional, List, Dict
from glob import glob

import pandas as pd

logger = logging.getLogger(__name__)


def list_snapshot_days(out_dir: str) -> List[str]:
    """price_snapshot 파일이 존재하는 날짜 목록"""
    pattern = os.path.join(out_dir, "price_snapshot_*.csv")
    days = []
    for f in sorted(glob(pattern)):
        base = os.path.basename(f)
        # price_snapshot_20260212.csv → 20260212
        ymd = base.replace("price_snapshot_", "").replace(".csv", "")
        if ymd != "latest" and len(ymd) == 8 and ymd.isdigit():
            days.append(ymd)
    return days


def load_close_map(out_dir: str, ymd: str) -> Dict[str, float]:
    """특정 날짜의 종가 맵 {종목코드: 종가}"""
    path = os.path.join(out_dir, f"price_snapshot_{ymd}.csv")
    if not os.path.exists(path):
        path = os.path.join(out_dir, "price_snapshot_latest.csv")
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path, dtype={"종목코드": str})
        if "종목코드" not in df.columns or "종가" not in df.columns:
            return {}
        df["종목코드"] = df["종목코드"].str.zfill(6)
        return dict(zip(df["종목코드"], pd.to_numeric(df["종가"], errors="coerce").fillna(0)))
    except Exception as e:
        logger.debug(f"close_map 로드 실패 {ymd}: {e}")
        return {}


def load_price_maps(out_dir: str, ymd: str) -> Dict[str, Dict[str, float]]:
    """종가 + 고가 + 저가 맵"""
    path = os.path.join(out_dir, f"price_snapshot_{ymd}.csv")
    if not os.path.exists(path):
        path = os.path.join(out_dir, "price_snapshot_latest.csv")
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path, dtype={"종목코드": str})
        df["종목코드"] = df["종목코드"].str.zfill(6)
        result = {}
        for _, row in df.iterrows():
            code = row["종목코드"]
            result[code] = {
                "close": float(row.get("종가", 0) or 0),
                "high": float(row.get("고가", 0) or 0),
                "low": float(row.get("저가", 0) or 0),
                "open": float(row.get("시가", 0) or 0),
            }
        return result
    except Exception as e:
        logger.debug(f"price_maps 로드 실패 {ymd}: {e}")
        return {}


def pick_recommend_file_per_day(out_dir: str) -> Dict[str, str]:
    """날짜별 추천 CSV 파일 매핑 {YYYYMMDD: filepath}"""
    pattern = os.path.join(out_dir, "recommend_*.csv")
    result = {}
    for f in sorted(glob(pattern)):
        base = os.path.basename(f)
        ymd = base.replace("recommend_", "").replace(".csv", "")
        if ymd != "latest" and ymd != "latest_cp949" and len(ymd) == 8 and ymd.isdigit():
            result[ymd] = f
    return result


def run_reality_check(out_dir: str, trade_ymd: str) -> None:
    """간단한 현실성 점검 (전일 추천 vs 오늘 결과)"""
    # 전일 추천 파일 로드
    rec_files = pick_recommend_file_per_day(out_dir)
    if not rec_files:
        logger.info("reality_check: 추천 파일 없음")
        return

    # 가장 최근 추천일
    last_day = sorted(rec_files.keys())[-1]
    if last_day >= trade_ymd:
        logger.info("reality_check: 아직 결과 미확인 (당일)")
        return

    try:
        rec_df = pd.read_csv(rec_files[last_day], dtype={"종목코드": str})
        close_map = load_close_map(out_dir, trade_ymd)

        if close_map and "종목코드" in rec_df.columns and "매수가" in rec_df.columns:
            rec_df["종목코드"] = rec_df["종목코드"].str.zfill(6)
            rec_df["today_close"] = rec_df["종목코드"].map(close_map)
            rec_df["ret_%"] = (rec_df["today_close"] / rec_df["매수가"] - 1) * 100

            valid = rec_df.dropna(subset=["ret_%"])
            if not valid.empty:
                avg_ret = valid["ret_%"].mean()
                win_rate = (valid["ret_%"] > 0).mean() * 100
                logger.info(f"🔍 Reality Check [{last_day}→{trade_ymd}]: "
                           f"평균 {avg_ret:+.2f}%, 승률 {win_rate:.0f}%")

                # 저장
                out_path = os.path.join(out_dir, f"reality_check_{trade_ymd}.csv")
                valid.to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        logger.warning(f"reality_check 에러: {e}")
