"""
scripts/validate_anti_struct_reversal.py
=========================================
[v3.9.23b] Anti-STRUCT Reversal Shadow 검증 스크립트.

평가 명시 production 승격 조건:
  · 월별 3개월 중 2개월 이상 EV+
  · macro_risk별 최소 NORMAL/CAUTION에서 EV+
  · 표본 n ≥ 300
  · 빠른 손절률 기존보다 낮거나 같음
  · MDD 기존보다 낮거나 같음
  · 알파 +1.0%p 이상

검증 차원:
  1. 월별 안정성 (시간 일반화)
  2. macro_risk 국면별 분리 (시장 모드 의존성)
  3. 기존 TOP_PICK / BUY_NOW_ELIGIBLE / ANTI_STRUCT_REVERSAL 비교
  4. 위험 지표 (MDD, 빠른 손절률, 손익비)

사용:
    python scripts/validate_anti_struct_reversal.py
    → reports/anti_struct_reversal_validation_YYYYMMDD.json
"""
import os
import sys
import json
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─── 설정 ───
HORIZON_DAYS = 5
ROUND_TRIP_COST_PCT = 0.71
MIN_N_BUCKET = 30  # 구간별 최소 표본
DATA_DIR = "data"
REPORTS_DIR = "reports"


def offset_bday(days_list, ymd, n):
    try:
        i = days_list.index(ymd)
        target_i = i + n
        if target_i < 0 or target_i >= len(days_list):
            return None
        return days_list[target_i]
    except ValueError:
        if n > 0:
            for d in days_list:
                if d > ymd:
                    return d
        return None


def load_price(ymd):
    p = f"{DATA_DIR}/price_snapshot_{ymd}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, dtype={"종목코드": str})
    if "종목코드" not in df.columns:
        return None
    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    return df.set_index("종목코드")


def compute_returns_with_features(horizon=HORIZON_DAYS):
    """모든 추천 종목 × 5일 실현 수익률 + 시점 정보.

    Returns: DataFrame with columns
      [rec_ymd, code, name, ret_net, win, month,
       STRUCT, TIMING, AI, BALANCE, DISPLAY, ELITE, FINAL, AXIS_MEAN,
       RR_NOW_TP1, ENTRY_GAP_PCT, EBS, TOP_PICK, ROUTE,
       BUY_NOW_GRADE, BUY_NOW_ELIGIBLE, BUY_NOW_SCORE,
       ANTI_STRUCT_REVERSAL_FLAG, ANTI_STRUCT_REVERSAL_TYPE,
       ANTI_STRUCT_REVERSAL_SCORE,
       이격도, MFI14, ret_1d, ret_5d, VWAP_GAP, ENTRY_RISK_LEVEL]
    """
    price_files = sorted(glob.glob(f"{DATA_DIR}/price_snapshot_*.csv"))
    trade_days = sorted([
        os.path.basename(f).replace("price_snapshot_", "").replace(".csv", "")
        for f in price_files
    ])

    results = []
    for rec_ymd in trade_days:
        rec_path = f"{DATA_DIR}/recommend_{rec_ymd}.csv"
        if not os.path.exists(rec_path):
            continue
        entry_ymd = offset_bday(trade_days, rec_ymd, 1)
        if entry_ymd is None:
            continue
        exit_ymd = offset_bday(trade_days, entry_ymd, horizon)
        if exit_ymd is None:
            continue

        try:
            rec = pd.read_csv(rec_path, dtype={"종목코드": str})
        except Exception:
            continue
        if "종목코드" not in rec.columns:
            continue
        rec["종목코드"] = rec["종목코드"].astype(str).str.zfill(6)

        ep = load_price(entry_ymd)
        xp = load_price(exit_ymd)
        if ep is None or xp is None:
            continue

        for _, row in rec.iterrows():
            code = row["종목코드"]
            if code not in ep.index or code not in xp.index:
                continue
            ep_val = ep.loc[code, "시가"]
            xp_val = xp.loc[code, "종가"]
            if not (ep_val and ep_val > 0 and xp_val and xp_val > 0):
                continue
            ret_gross = (xp_val / ep_val - 1) * 100
            ret_net = ret_gross - ROUND_TRIP_COST_PCT

            results.append({
                "rec_ymd": rec_ymd,
                "code": code,
                "name": row.get("종목명", ""),
                "ret_net": ret_net,
                "win": int(ret_net > 0),
                "month": rec_ymd[:6],
                "STRUCT": pd.to_numeric(row.get("STRUCT_SCORE"), errors="coerce"),
                "TIMING": pd.to_numeric(row.get("TIMING_SCORE"), errors="coerce"),
                "AI": pd.to_numeric(row.get("AI_SCORE"), errors="coerce"),
                "BALANCE": pd.to_numeric(
                    row.get("BALANCE_SCORE"), errors="coerce"
                ),
                "DISPLAY": pd.to_numeric(
                    row.get("DISPLAY_SCORE"), errors="coerce"
                ),
                "ELITE": pd.to_numeric(row.get("ELITE_SCORE"), errors="coerce"),
                "FINAL": pd.to_numeric(row.get("FINAL_SCORE"), errors="coerce"),
                "AXIS_MEAN": pd.to_numeric(
                    row.get("AXIS_MEAN"), errors="coerce"
                ),
                "RR_NOW_TP1": pd.to_numeric(
                    row.get("RR_NOW_TP1"), errors="coerce"
                ),
                "ENTRY_GAP_PCT": pd.to_numeric(
                    row.get("ENTRY_GAP_PCT"), errors="coerce"
                ),
                "EBS": pd.to_numeric(row.get("EBS"), errors="coerce"),
                "TOP_PICK": int(
                    str(row.get("TOP_PICK", "0")).strip()
                    in ("1", "1.0", "TRUE", "Y", "YES")
                ),
                "ROUTE": str(row.get("ROUTE", "")).upper(),
                "BUY_NOW_GRADE": str(row.get("BUY_NOW_GRADE", "")).upper(),
                "BUY_NOW_ELIGIBLE": int(
                    str(row.get("BUY_NOW_ELIGIBLE", "0")).strip()
                    in ("1", "1.0", "TRUE", "Y", "YES")
                ),
                "BUY_NOW_SCORE": pd.to_numeric(
                    row.get("BUY_NOW_SCORE"), errors="coerce"
                ),
                "ANTI_STRUCT_REVERSAL_FLAG": int(
                    str(row.get("ANTI_STRUCT_REVERSAL_FLAG", "0")).strip()
                    in ("1", "1.0", "TRUE", "Y", "YES")
                ),
                "ANTI_STRUCT_REVERSAL_TYPE": str(
                    row.get("ANTI_STRUCT_REVERSAL_TYPE", "")
                ),
                "ANTI_STRUCT_REVERSAL_SCORE": pd.to_numeric(
                    row.get("ANTI_STRUCT_REVERSAL_SCORE"), errors="coerce"
                ),
                "이격도": pd.to_numeric(row.get("이격도"), errors="coerce"),
                "MFI14": pd.to_numeric(row.get("MFI14"), errors="coerce"),
                "ret_1d": pd.to_numeric(row.get("ret_1d_%"), errors="coerce"),
                "ret_5d": pd.to_numeric(row.get("ret_5d_%"), errors="coerce"),
                "VWAP_GAP": pd.to_numeric(row.get("VWAP_GAP"), errors="coerce"),
                "ENTRY_RISK_LEVEL": str(
                    row.get("ENTRY_RISK_LEVEL", "")
                ).upper(),
                "MACRO_RISK": str(row.get("MACRO_RISK", "")).upper(),
            })

    return pd.DataFrame(results)


def stat_block(sub, label, baseline_win=None, baseline_ret=None):
    """한 그룹의 통계 + 위험 지표.

    [v3.9.23b] 모든 numeric을 Python native (int/float)로 강제 캐스팅.
    JSON 직렬화 시 numpy.bool_/int64/float64가 문자열화되는 버그 방지.
    """
    n = int(len(sub))
    if n == 0:
        return {"label": str(label), "n": 0, "insufficient": True}

    win = float(sub["win"].mean()) * 100
    ret = float(sub["ret_net"].mean())

    # 손익비
    wins = sub.loc[sub["win"] == 1, "ret_net"]
    losses = sub.loc[sub["win"] == 0, "ret_net"]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    b_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else None

    # 빠른 손절률 (ret_net <= -5)
    fast_loss_rate = float((sub["ret_net"] <= -5.0).mean()) * 100

    # MDD 근사 (개별 종목의 최악 손실)
    worst = float(sub["ret_net"].min()) if n > 0 else 0.0

    result = {
        "label": str(label),
        "n": int(n),
        "win_rate": round(float(win), 2),
        "avg_ret": round(float(ret), 3),
        "avg_win_ret": round(float(avg_win), 3),
        "avg_loss_ret": round(float(avg_loss), 3),
        "b_ratio": round(float(b_ratio), 3) if b_ratio else None,
        "fast_loss_rate": round(float(fast_loss_rate), 2),
        "worst_loss": round(float(worst), 3),
        "insufficient": bool(n < MIN_N_BUCKET),
    }
    if baseline_win is not None:
        result["win_alpha"] = round(float(win - baseline_win), 2)
    if baseline_ret is not None:
        result["ret_alpha"] = round(float(ret - baseline_ret), 3)
    return result


def validate_monthly(df, baseline_win, baseline_ret):
    """월별 안정성 검증."""
    out = {}
    for month in sorted(df["month"].unique()):
        sub = df[df["month"] == month]
        out[month] = stat_block(sub, month, baseline_win, baseline_ret)
    return out


def validate_macro(df, baseline_win, baseline_ret):
    """macro_risk 국면별 검증.

    [v3.9.23a-fix] MACRO_RISK 빈 값(legacy CSV)은 UNKNOWN 버킷으로 분리.
    """
    out = {}
    # 빈 값/NaN을 UNKNOWN으로 통합
    df_work = df.copy()
    df_work["_macro_key"] = (
        df_work["MACRO_RISK"]
        .fillna("UNKNOWN")
        .replace("", "UNKNOWN")
        .astype(str)
        .str.upper()
    )
    for macro in sorted(df_work["_macro_key"].unique()):
        sub = df_work[df_work["_macro_key"] == macro]
        out[macro] = stat_block(sub, macro, baseline_win, baseline_ret)
    return out


def validate_groups(df_all):
    """기존 / TOP_PICK / BUY_NOW / ANTI_STRUCT_REVERSAL 비교."""
    baseline_win = df_all["win"].mean() * 100
    baseline_ret = df_all["ret_net"].mean()

    groups = {
        "baseline_all": df_all,
        "TOP_PICK_1": df_all[df_all["TOP_PICK"] == 1],
        "BUY_NOW_ELIGIBLE_1": df_all[df_all["BUY_NOW_ELIGIBLE"] == 1],
        "ANTI_STRUCT_REVERSAL_FLAG_1": df_all[
            df_all["ANTI_STRUCT_REVERSAL_FLAG"] == 1
        ],
        "ANTI_STRUCT_REVERSAL_BASIC": df_all[
            df_all["ANTI_STRUCT_REVERSAL_TYPE"] == "BASIC"
        ],
        "ANTI_STRUCT_REVERSAL_STRONG": df_all[
            df_all["ANTI_STRUCT_REVERSAL_TYPE"] == "STRONG"
        ],
        "ANTI_STRUCT_REVERSAL_CHAMPION": df_all[
            df_all["ANTI_STRUCT_REVERSAL_TYPE"] == "CHAMPION"
        ],
        # 교집합
        "ASR_AND_BUY_NOW": df_all[
            (df_all["ANTI_STRUCT_REVERSAL_FLAG"] == 1)
            & (df_all["BUY_NOW_ELIGIBLE"] == 1)
        ],
        "ASR_AND_TOP_PICK": df_all[
            (df_all["ANTI_STRUCT_REVERSAL_FLAG"] == 1)
            & (df_all["TOP_PICK"] == 1)
        ],
    }

    out = {}
    for name, sub in groups.items():
        out[name] = stat_block(sub, name, baseline_win, baseline_ret)
    return out


# ════════════════════════════════════════════════════════════════════
# [v3.9.23c] ASR_STRONG 세분화 슬라이스 검증
#
# 평가 명시 3대 슬라이싱 축:
#   ① ASR_STRONG 단독 유니버스 격리 검증 (n=98)
#   ② 국면 필터링 (ASR_STRONG minus NORMAL) 조합
#   ③ CHAMPION 꼬리 위험 진단
# ════════════════════════════════════════════════════════════════════


def validate_asr_slices(df_all, baseline_win, baseline_ret):
    """[v3.9.23c] ASR 타입/국면 세분화 슬라이스 검증.

    Returns: dict[slice_name] = {
        "group": stat_block 결과 (전체 표본 기준),
        "monthly": validate_monthly 결과,
        "macro": validate_macro 결과,
        "promotion": check_promotion_criteria 결과,
        "tail_diagnosis": (CHAMPION 전용) 꼬리 위험 진단,
    }

    슬라이스:
      - ASR_BASIC                : TYPE=BASIC만
      - ASR_STRONG               : TYPE=STRONG만 (★ 가장 유망)
      - ASR_CHAMPION             : TYPE=CHAMPION만 (B-Ratio 문제)
      - ASR_STRONG_CAUTION       : STRONG + macro=CAUTION
      - ASR_STRONG_CRITICAL      : STRONG + macro=CRITICAL
      - ASR_STRONG_EXCLUDE_NORMAL: STRONG - macro=NORMAL
      - ASR_STRONG_NORMAL_ONLY   : STRONG + macro=NORMAL (대조군)
    """
    slices = {
        "ASR_BASIC": df_all[df_all["ANTI_STRUCT_REVERSAL_TYPE"] == "BASIC"],
        "ASR_STRONG": df_all[df_all["ANTI_STRUCT_REVERSAL_TYPE"] == "STRONG"],
        "ASR_CHAMPION": df_all[
            df_all["ANTI_STRUCT_REVERSAL_TYPE"] == "CHAMPION"
        ],
        "ASR_STRONG_CAUTION": df_all[
            (df_all["ANTI_STRUCT_REVERSAL_TYPE"] == "STRONG")
            & (df_all["MACRO_RISK"] == "CAUTION")
        ],
        "ASR_STRONG_CRITICAL": df_all[
            (df_all["ANTI_STRUCT_REVERSAL_TYPE"] == "STRONG")
            & (df_all["MACRO_RISK"] == "CRITICAL")
        ],
        "ASR_STRONG_EXCLUDE_NORMAL": df_all[
            (df_all["ANTI_STRUCT_REVERSAL_TYPE"] == "STRONG")
            & (df_all["MACRO_RISK"] != "NORMAL")
            & (df_all["MACRO_RISK"].notna())
            & (df_all["MACRO_RISK"] != "")
        ],
        "ASR_STRONG_NORMAL_ONLY": df_all[
            (df_all["ANTI_STRUCT_REVERSAL_TYPE"] == "STRONG")
            & (df_all["MACRO_RISK"] == "NORMAL")
        ],
    }

    # 그룹 통계 (slice 자체 + baseline)
    # check_promotion_criteria가 비교할 baseline_all 통계 (진짜 전체 데이터)
    baseline_stat = stat_block(df_all, "baseline_all", baseline_win, baseline_ret)

    out = {}
    for name, sub in slices.items():
        # 1. 슬라이스 자체 통계
        group_stat = stat_block(sub, name, baseline_win, baseline_ret)
        # 2. 슬라이스 월별
        monthly = validate_monthly(sub, baseline_win, baseline_ret)
        # 3. 슬라이스 macro_risk
        macro = validate_macro(sub, baseline_win, baseline_ret)
        # 4. promotion 체크용 group_stats
        # check_promotion_criteria가 "ANTI_STRUCT_REVERSAL_FLAG_1" 키 보니까
        # 그 형태로 재구성. baseline_all은 진짜 전체 데이터 통계.
        promo_groups = {
            "baseline_all": baseline_stat,
            "ANTI_STRUCT_REVERSAL_FLAG_1": group_stat,
        }
        promotion = check_promotion_criteria(monthly, macro, promo_groups)

        out[name] = {
            "group": group_stat,
            "monthly": monthly,
            "macro": macro,
            "promotion": promotion,
        }

    # CHAMPION 꼬리 위험 진단 (평가 ③)
    out["ASR_CHAMPION"]["tail_diagnosis"] = (
        _diagnose_champion_tail_risk(slices["ASR_CHAMPION"])
    )

    return out


def _diagnose_champion_tail_risk(champion_df):
    """[v3.9.23c] CHAMPION 타입 꼬리 위험 진단.

    평가 ③: 극단적 손실 케이스의 원인 분리
      - 유동성 부족? (거래대금)
      - 단기 급락 시 낙하산 부족? (ret_5d 패턴)
      - 손익비 자체 구조?
    """
    if len(champion_df) == 0:
        return {"available": False, "reason": "CHAMPION 표본 없음"}

    losses = champion_df[champion_df["win"] == 0].copy()
    wins = champion_df[champion_df["win"] == 1].copy()

    if len(losses) == 0:
        return {"available": False, "reason": "손실 케이스 없음"}

    # 손실 분포
    loss_p5 = float(losses["ret_net"].quantile(0.05)) if len(losses) > 0 else 0
    loss_p25 = float(losses["ret_net"].quantile(0.25)) if len(losses) > 0 else 0
    loss_median = float(losses["ret_net"].median()) if len(losses) > 0 else 0
    loss_mean = float(losses["ret_net"].mean()) if len(losses) > 0 else 0

    # 가장 큰 손실 5개
    worst_5 = losses.nsmallest(min(5, len(losses)), "ret_net")
    worst_cases = []
    for _, row in worst_5.iterrows():
        worst_cases.append({
            "date": str(row.get("rec_ymd", "")),
            "code": str(row.get("code", "")),
            "name": str(row.get("name", "")),
            "ret_net": round(float(row["ret_net"]), 2),
            "ret_5d_prior": round(float(row.get("ret_5d", 0) or 0), 2),
            "ret_1d_prior": round(float(row.get("ret_1d", 0) or 0), 2),
            "macro": str(row.get("MACRO_RISK", "")),
        })

    # 평균 wins vs losses (B-Ratio 분해)
    avg_win = float(wins["ret_net"].mean()) if len(wins) > 0 else 0
    avg_loss = float(losses["ret_net"].mean()) if len(losses) > 0 else 0
    b_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else None

    # 손실 케이스의 macro 분포 (어느 국면에서 깨지나)
    macro_dist = (
        losses["MACRO_RISK"].fillna("UNKNOWN")
        .replace("", "UNKNOWN")
        .value_counts()
        .to_dict()
    )
    macro_dist = {str(k): int(v) for k, v in macro_dist.items()}

    return {
        "available": True,
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
        "avg_win_ret": round(avg_win, 3),
        "avg_loss_ret": round(avg_loss, 3),
        "b_ratio": round(b_ratio, 3) if b_ratio else None,
        "loss_distribution": {
            "p5": round(loss_p5, 2),
            "p25": round(loss_p25, 2),
            "median": round(loss_median, 2),
            "mean": round(loss_mean, 2),
        },
        "worst_5_cases": worst_cases,
        "loss_macro_distribution": macro_dist,
        "note": (
            "B-Ratio < 1.0이면 '조금 벌고 크게 깨지는' 구조. "
            "worst_5_cases의 ret_5d_prior가 크면 급등 후 음봉 진입 패턴, "
            "macro_distribution에서 NORMAL 비중이 크면 횡보장에서 깨짐."
        ),
    }


def check_promotion_criteria(monthly_stats, macro_stats, group_stats):
    """production 승격 조건 체크.

    ★ 모든 기준은 ASR subset 기준 (monthly_stats, macro_stats 모두 ASR-only).
    ★ [v3.9.23b] 모든 passed/numeric을 Python native로 강제 캐스팅.
       numpy.bool_/int64가 json.dump(default=str)로 문자열화되는 버그 방지.
    """
    criteria = {}

    # 1. 월별 3개월 중 2개월 이상 EV+ (현실: 데이터 기간 따라 조정)
    months_ev_positive = sum(
        1 for m, s in monthly_stats.items()
        if not s.get("insufficient") and s.get("avg_ret", 0) > 0
    )
    months_sufficient = sum(
        1 for s in monthly_stats.values() if not s.get("insufficient")
    )
    criteria["monthly_ev_positive"] = {
        "value": f"{months_ev_positive}/{months_sufficient}",
        "passed": bool(months_sufficient >= 3 and months_ev_positive >= 2),
        "note": "ASR 월별 3개월 중 2개월 이상 EV+ (표본 ≥30/월)",
    }

    # 2. NORMAL/CAUTION에서 EV+ (ASR 기준)
    normal_caution_ev = []
    for macro in ("NORMAL", "CAUTION"):
        if macro in macro_stats and not macro_stats[macro].get("insufficient"):
            normal_caution_ev.append(
                bool(macro_stats[macro].get("avg_ret", 0) > 0)
            )
    criteria["macro_normal_caution_ev"] = {
        "value": (
            f"NORMAL/CAUTION 중 {sum(normal_caution_ev)}/"
            f"{len(normal_caution_ev)} 통과"
            if normal_caution_ev else "ASR 표본 부족"
        ),
        "passed": bool(
            bool(normal_caution_ev) and all(normal_caution_ev)
        ),
        "note": "ASR NORMAL/CAUTION에서 모두 EV+ 필요 (각 표본 ≥30)",
    }

    # 3. n ≥ 300 (ASR_FLAG=1 표본)
    asr_n = int(
        group_stats.get("ANTI_STRUCT_REVERSAL_FLAG_1", {}).get("n", 0)
    )
    criteria["sample_size"] = {
        "value": f"n={asr_n}",
        "passed": bool(asr_n >= 300),
        "note": "ANTI_STRUCT_REVERSAL_FLAG=1 표본 ≥ 300",
    }

    # 4. 빠른 손절률 기존 이하 (ASR vs baseline 직접 비교 — group_stats 사용)
    asr_fast = float(
        group_stats.get("ANTI_STRUCT_REVERSAL_FLAG_1", {}).get(
            "fast_loss_rate", 100
        )
    )
    baseline_fast = float(
        group_stats.get("baseline_all", {}).get("fast_loss_rate", 0)
    )
    criteria["fast_loss_rate"] = {
        "value": f"ASR={asr_fast:.1f}% vs baseline={baseline_fast:.1f}%",
        "passed": bool(asr_fast <= baseline_fast),
        "note": "빠른 손절률(-5% 이내) baseline 이하",
    }

    # 5. MDD 기존 이하 (ASR vs baseline)
    asr_worst = float(
        group_stats.get("ANTI_STRUCT_REVERSAL_FLAG_1", {}).get(
            "worst_loss", -100
        )
    )
    baseline_worst = float(
        group_stats.get("baseline_all", {}).get("worst_loss", -100)
    )
    criteria["worst_loss"] = {
        "value": f"ASR={asr_worst:.1f}% vs baseline={baseline_worst:.1f}%",
        "passed": bool(asr_worst >= baseline_worst),  # 덜 나쁜 손실
        "note": "최악 손실 baseline보다 덜함",
    }

    # 6. 평균 알파 ≥ +1.0%p (group_stats — n 가중 평균)
    ret_alpha = float(
        group_stats.get("ANTI_STRUCT_REVERSAL_FLAG_1", {}).get(
            "ret_alpha", -99
        )
    )
    criteria["return_alpha"] = {
        "value": f"+{ret_alpha:.2f}%p",
        "passed": bool(ret_alpha >= 1.0),
        "note": "ASR 평균수익 알파 ≥ +1.0%p (n-가중 평균)",
    }

    # 7. ★ [v3.9.23a-fix → v3.9.23c-hotfix] 월별 알파 산술 평균 (4월 편향 보정)
    #
    # 평가 명시 v3.9.23c-hotfix: 유효 월 1개월만으로 통과 위험.
    # ASR_STRONG_CAUTION 같은 경우 유효 월이 4월 하나뿐인데도 +6%p로
    # "unbiased alpha 통과"처럼 보이는 문제.
    #
    # 강화 규칙 (★ 평가 권장):
    #   · 유효 월 ≥ 3 AND mean ≥ +1.0%p → PASS
    #   · 유효 월 1~2개월 → OBSERVE (passed=False, "유효 월 부족" 표시)
    #   · 유효 월 0개월 → FAIL (표본 부족)
    REQUIRED_VALID_MONTHS = 3
    valid_monthly_alphas = [
        float(s["ret_alpha"]) for m, s in monthly_stats.items()
        if not s.get("insufficient") and "ret_alpha" in s
    ]
    n_valid_months = len(valid_monthly_alphas)

    if n_valid_months == 0:
        criteria["return_alpha_unbiased"] = {
            "value": "ASR 월별 표본 부족",
            "passed": False,
            "valid_months": 0,
            "required_valid_months": REQUIRED_VALID_MONTHS,
            "unbiased_alpha_status": "INSUFFICIENT_DATA",
            "note": (
                "월별 ret_alpha 산술 평균 ≥ +1.0%p "
                f"(유효 월 ≥ {REQUIRED_VALID_MONTHS}개 필요)"
            ),
        }
    else:
        mean_monthly_alpha = float(
            sum(valid_monthly_alphas) / n_valid_months
        )
        if n_valid_months < REQUIRED_VALID_MONTHS:
            # ★ 핵심 강화: 유효 월 부족 시 PASS 불가 — "OBSERVE" 표시
            criteria["return_alpha_unbiased"] = {
                "value": (
                    f"{mean_monthly_alpha:+.2f}%p "
                    f"({n_valid_months}개월 산술평균) — 유효 월 부족"
                ),
                "passed": False,
                "valid_months": n_valid_months,
                "required_valid_months": REQUIRED_VALID_MONTHS,
                "unbiased_alpha_status": "OBSERVE",
                "note": (
                    f"월별 ret_alpha 산술 평균 ≥ +1.0%p "
                    f"(유효 월 ≥ {REQUIRED_VALID_MONTHS}개 필요, "
                    f"현재 {n_valid_months}개)"
                ),
            }
        else:
            criteria["return_alpha_unbiased"] = {
                "value": (
                    f"{mean_monthly_alpha:+.2f}%p "
                    f"({n_valid_months}개월 산술평균)"
                ),
                "passed": bool(mean_monthly_alpha >= 1.0),
                "valid_months": n_valid_months,
                "required_valid_months": REQUIRED_VALID_MONTHS,
                "unbiased_alpha_status": (
                    "PASS" if mean_monthly_alpha >= 1.0 else "FAIL"
                ),
                "note": (
                    "월별 ret_alpha 산술 평균 ≥ +1.0%p (4월 편향 제거)"
                ),
            }

    all_passed = bool(all(c["passed"] for c in criteria.values()))
    n_passed = int(sum(1 for c in criteria.values() if c["passed"]))
    n_total = int(len(criteria))

    return {
        "criteria": criteria,
        "n_passed": n_passed,
        "n_total": n_total,
        "all_passed": all_passed,
        "recommendation": (
            "✅ PRODUCTION 승격 권장"
            if all_passed
            else f"❌ 보류 — {n_passed}/{n_total} 통과"
        ),
    }


def main():
    print("[v3.9.23c] Anti-STRUCT Reversal Shadow + STRONG 슬라이스 검증...")
    df = compute_returns_with_features()
    print(f"  표본 로드: {len(df)}건")

    if df.empty:
        print("  ❌ 데이터 없음")
        return

    baseline_win = df["win"].mean() * 100
    baseline_ret = df["ret_net"].mean()
    print(f"  기준 (전체): win={baseline_win:.1f}%, ret={baseline_ret:+.2f}%")

    # ─── ASR subset 분리 (v3.9.23a-fix) ───
    asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1].copy()
    print(f"  ASR subset: {len(asr)}건")

    # 전체 universe (참고용)
    monthly_all = validate_monthly(df, baseline_win, baseline_ret)
    macro_all = validate_macro(df, baseline_win, baseline_ret)
    # ASR subset
    monthly_asr = validate_monthly(asr, baseline_win, baseline_ret)
    macro_asr = validate_macro(asr, baseline_win, baseline_ret)
    groups = validate_groups(df)
    promotion = check_promotion_criteria(monthly_asr, macro_asr, groups)

    # [v3.9.23c] 슬라이스 검증 — 평가 명시 3대 축
    slices = validate_asr_slices(df, baseline_win, baseline_ret)
    print(f"  슬라이스: {len(slices)}개")

    # 무결성
    asr_total = sum(s.get("n", 0) for s in monthly_asr.values())
    print(f"  무결성: sum(monthly_asr.n)={asr_total} == |ASR|={len(asr)} "
          f"{'✓' if asr_total == len(asr) else '✗'}")

    # 출력
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "version": "v3.9.23c-slices",
        "horizon_bdays": HORIZON_DAYS,
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "n_total": int(len(df)),
        "n_asr": int(len(asr)),
        "baseline": {
            "win_rate": round(float(baseline_win), 2),
            "avg_ret": round(float(baseline_ret), 3),
        },
        # 전체 universe — 참고용
        "by_month_all": monthly_all,
        "by_macro_risk_all": macro_all,
        # ASR subset — production 승격 판단 기준
        "by_month_asr": monthly_asr,
        "by_macro_risk_asr": macro_asr,
        "by_group": groups,
        "promotion_check": promotion,
        "promotion_check_basis": (
            "ASR subset 월별/macro_risk별 통계 기반 (monthly_asr, macro_asr)"
        ),
        # [v3.9.23c] 슬라이스 — STRONG 중심 세분화 검증
        "asr_slices": slices,
        "asr_slices_basis": (
            "평가 명시 3대 축: ASR_STRONG 단독 + STRONG-NORMAL + "
            "CHAMPION 꼬리위험"
        ),
    }

    Path(REPORTS_DIR).mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    out_path = f"{REPORTS_DIR}/anti_struct_reversal_validation_{today}.json"

    # [v3.9.23b] numpy 타입 안전 직렬화 — default=str로 문자열화되는 버그 방지
    def _json_default(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # 마지막 폴백 — str (datetime 등)
        return str(obj)

    # [v3.9.23-hotfix] Windows cp949 → utf-8 명시 (한글/이모지 안전)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2,
                  default=_json_default)
    print(f"\n→ 저장: {out_path}")

    # 콘솔 요약
    print("\n" + "=" * 80)
    print("【월별 안정성 — 전체 universe (참고용)】")
    print("=" * 80)
    print(f"{'월':<8} {'n':>5} {'승률':>7} {'알파':>7} {'평균수익':>9} {'최악':>7}")
    for m, s in monthly_all.items():
        if s.get("insufficient"):
            print(f"  {m}  n={s['n']} [표본 부족]")
        else:
            print(
                f"  {m}  n={s['n']:>4}  "
                f"win={s['win_rate']:>5.1f}%  "
                f"Δ{s['win_alpha']:>+5.1f}p  "
                f"ret={s['avg_ret']:>+6.2f}%  "
                f"worst={s['worst_loss']:>+6.1f}%"
            )

    print("\n" + "=" * 80)
    print("【월별 안정성 — ★ ASR subset (승격 판단 기준)】")
    print("=" * 80)
    print(f"{'월':<8} {'n':>5} {'승률':>7} {'알파':>7} {'평균수익':>9} {'최악':>7}")
    for m, s in monthly_asr.items():
        if s.get("insufficient"):
            print(f"  {m}  n={s['n']} [표본 부족]")
        else:
            print(
                f"  {m}  n={s['n']:>4}  "
                f"win={s['win_rate']:>5.1f}%  "
                f"Δ{s['win_alpha']:>+5.1f}p  "
                f"ret={s['avg_ret']:>+6.2f}%  "
                f"worst={s['worst_loss']:>+6.1f}%"
            )

    print("\n" + "=" * 80)
    print("【macro_risk 국면별 — 전체 universe (참고용)】")
    print("=" * 80)
    for macro_name, s in macro_all.items():
        if s.get("insufficient"):
            print(f"  {macro_name}  n={s['n']} [표본 부족]")
        else:
            print(
                f"  {macro_name:<12} n={s['n']:>4}  "
                f"win={s['win_rate']:>5.1f}%  Δ{s['win_alpha']:>+5.1f}p  "
                f"ret={s['avg_ret']:>+6.2f}%"
            )

    print("\n" + "=" * 80)
    print("【macro_risk 국면별 — ★ ASR subset (승격 판단 기준)】")
    print("=" * 80)
    for macro_name, s in macro_asr.items():
        if s.get("insufficient"):
            print(f"  {macro_name}  n={s['n']} [표본 부족]")
        else:
            print(
                f"  {macro_name:<12} n={s['n']:>4}  "
                f"win={s['win_rate']:>5.1f}%  Δ{s['win_alpha']:>+5.1f}p  "
                f"ret={s['avg_ret']:>+6.2f}%"
            )

    print("\n" + "=" * 80)
    print("【그룹별 비교】")
    print("=" * 80)
    for name, s in groups.items():
        if s.get("insufficient") and s.get("n", 0) == 0:
            print(f"  {name}: 표본 없음")
            continue
        print(
            f"  {name:<32} n={s['n']:>5}  "
            f"win={s.get('win_rate', 0):>5.1f}% "
            f"(Δ{s.get('win_alpha', 0):>+5.1f}p)  "
            f"ret={s.get('avg_ret', 0):>+6.2f}% "
            f"(Δ{s.get('ret_alpha', 0):>+5.2f}p)  "
            f"fast_loss={s.get('fast_loss_rate', 0):>4.1f}%  "
            f"worst={s.get('worst_loss', 0):>+6.1f}%"
        )

    print("\n" + "=" * 80)
    print("【Production 승격 체크 (★ ASR subset 기준)】")
    print("=" * 80)
    for k, c in promotion["criteria"].items():
        icon = "✅" if c["passed"] else "❌"
        print(f"  {icon} {k:<28} {c['value']:<35} ({c['note']})")
    print(f"\n  결과: {promotion['recommendation']}")
    print(f"  통과: {promotion['n_passed']}/{promotion['n_total']}")

    # [v3.9.23c] 슬라이스 요약
    print("\n" + "=" * 80)
    print("【ASR 슬라이스 비교 — ★ v3.9.23c 평가 명시 3대 축】")
    print("=" * 80)
    print(
        f"{'슬라이스':<32} {'n':>5} {'승률':>7} "
        f"{'평균수익':>9} {'B-Ratio':>8} {'알파':>8} {'승격':>5}"
    )
    print("-" * 80)
    for sname, sdata in slices.items():
        g = sdata["group"]
        p = sdata["promotion"]
        if g.get("insufficient") and g.get("n", 0) < 30:
            mark = "표본부족"
            print(f"  {sname:<30} n={g['n']:>4}  [{mark}]")
            continue
        n_pass = p.get("n_passed", 0)
        n_tot = p.get("n_total", 0)
        print(
            f"  {sname:<30} n={g['n']:>4}  "
            f"{g.get('win_rate', 0):>5.1f}%  "
            f"{g.get('avg_ret', 0):>+6.2f}%  "
            f"{(g.get('b_ratio') or 0):>6.2f}  "
            f"{g.get('ret_alpha', 0):>+5.2f}p  "
            f"{n_pass}/{n_tot}"
        )

    # CHAMPION 꼬리위험 진단
    print("\n" + "=" * 80)
    print("【CHAMPION 꼬리 위험 진단 — 평가 ③】")
    print("=" * 80)
    tail = slices.get("ASR_CHAMPION", {}).get("tail_diagnosis", {})
    if not tail.get("available"):
        print(f"  진단 불가: {tail.get('reason', '?')}")
    else:
        print(
            f"  wins n={tail['n_wins']}, losses n={tail['n_losses']}  "
            f"평균승={tail['avg_win_ret']:+.2f}%  "
            f"평균손={tail['avg_loss_ret']:+.2f}%  "
            f"B-Ratio={tail['b_ratio']}"
        )
        dist = tail["loss_distribution"]
        print(
            f"  손실분포: p5={dist['p5']:+.1f}%  p25={dist['p25']:+.1f}%  "
            f"median={dist['median']:+.1f}%  mean={dist['mean']:+.1f}%"
        )
        print(f"  손실 macro 분포: {tail['loss_macro_distribution']}")
        print(f"  최악 케이스 5개:")
        for c in tail["worst_5_cases"]:
            print(
                f"    {c['date']} {c['name']:<14} ret={c['ret_net']:+.1f}%  "
                f"ret_5d={c['ret_5d_prior']:+.1f}%  "
                f"ret_1d={c['ret_1d_prior']:+.1f}%  "
                f"macro={c['macro']}"
            )


if __name__ == "__main__":
    main()
