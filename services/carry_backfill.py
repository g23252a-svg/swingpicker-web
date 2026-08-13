# -*- coding: utf-8 -*-
"""[v60] 보유(CARRY) 행이 보강 단계를 건너뛰어 3개월간 점수가 비어 있던 것 복원.

■ v59가 남긴 질문에 대한 답
  v59에서 "업종 결측 30행이 **어느 경로로 늦게 합류하는지**는 특정하지 못했다"고
  적었다. 특정했다. 그리고 "늦게 합류"가 아니었다 — **다른(짧은) 경로로 다시
  만들어진다.**

  파이프라인 순서:
    Stage 3 run_scoring   … 업종 분류 · 섹터 모멘텀 · 수급 · 시총기준일
                            + LDY/TOTAL/RANK_SCORE · 벤치_60d_* 대입
    Stage 4 enrich_news   … NEWS_SCORE
    Stage 5 run_calibration … ★ 여기서 CARRY 행이 concat으로 **합류**
    Stage 6 finalize

  CARRY 행은 Stage 5에서 붙으므로 Stage 3·4가 부여하는 컬럼을 **구조적으로 받을
  수 없다.** `_refresh_carry_rows`는 analyze→trigger→ML→build_global_score만
  돌린다(보강 단계는 없다).

■ 실측으로 확인한 붕괴 사슬 (2026-08 배치)
  녹십자(006280)
    08/04 FRESH            업종_대분류=바이오·의약품  LDY_SCORE=39.2
    08/05 FRESH            업종_대분류=바이오·의약품  LDY_SCORE=39.8
    08/06 CARRY_REFRESHED  업종_대분류=∅             LDY_SCORE=∅   ← 여기서 죽는다
    08/07 CARRY_REFRESHED  업종_대분류=∅             LDY_SCORE=∅
  지아이텍(382480)
    05/14 FRESH            업종_대분류=조선·기계·설비
    05/15 CARRY_REFRESHED  ∅
    08/07 CARRY_LEGACY     ∅   ← **2개월 반째 같은 구멍을 복사 중**
  한 번 CARRY가 되면 그 구멍이 legacy 복사로 무한 전파된다. 8/07 배치에서
  업종 결측 30종목은 **100% CARRY 행**이었다(FRESH 285행은 결측 0).

■ 왜 고칠 수 있는가 — 대부분이 결정적으로 재계산 가능하다
  LDY_SCORE·TOTAL_SCORE·RANK_SCORE는 `pipeline_calibrate.py`가
  **DISPLAY_SCORE에 그대로 대입**하는 별칭이다(457행). 최근 10일 FRESH
  2,750행 전수 검증: 세 컬럼 모두 DISPLAY_SCORE와 100% 일치, 반례 0.
  그리고 CARRY 41행 전부 DISPLAY_SCORE를 갖고 있다.
  → 없는 값을 추정하는 게 아니라, **한 줄 대입을 캐리 합류 뒤에 다시 하는 것**이다.

■ 이 결측이 실제로 무엇을 망쳤나
  1. 트리맵 색: `chart_components`가 `color="LDY_SCORE"`로 칠한다. NaN이
     0.0으로 강제되어 **보유 종목 30개가 점수 0인 것처럼** 표시됐다.
  2. `services/data_store`는 DISPLAY_SCORE→TOTAL_SCORE→LDY_SCORE→RANK_SCORE
     순으로 폴백한다. 캐리 행에서 이 폴백 사슬이 통째로 끊겨 있었다.
  3. v59가 고친 켈리 섹터 배수 오염의 **공급원**이 바로 이 구멍이었다.

■ 채우지 않는 것 (없는 값을 만들어내지 않는다)
  NEWS_SCORE·STRATEGY 계열은 해당 단계를 **실제로 돌리지 않았으므로** 비워
  둔다. NEWS_SCORE=0.0은 "확인했고 특이사항 없음"을 뜻하는 값이라(뉴스 단계의
  기본값) 그것을 캐리 행에 넣으면 **하지 않은 확인을 했다고 말하는 것**이다.
  대신 사유를 명시하고 리포트에 남긴다.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from services.sector_repair import (SECTOR_COL, SECTOR_DETAIL_COL,
                                    SECTOR_RAW_COL, blank_mask, repair_sector)

logger = logging.getLogger("carry_backfill")

BUILD_MODE_COL = "ROW_BUILD_MODE"
FRESH_MODE = "FRESH"
CARRY_MODES = ("CARRY_REFRESHED", "CARRY_LEGACY")

DISPLAY_COL = "DISPLAY_SCORE"
# DISPLAY_SCORE의 순수 별칭 — pipeline_calibrate 457행과 같은 정의(SSOT).
DISPLAY_ALIASES = ("LDY_SCORE", "TOTAL_SCORE", "RANK_SCORE")
# 섹터 단위 집계값 — 종목이 아니라 **섹터**의 속성이므로 같은 섹터에서 가져온다.
# (캐리 3행만으로 다시 평균을 내면 '섹터 평균'이 아니라 '그 3행의 평균'이 된다.)
SECTOR_AGG_COLS = ("SECTOR_RET_5D", "SECTOR_RS", "SECTOR_RANK")

# 배치 단위 스칼라 — FRESH 행 전체가 같은 값을 공유한다. 그래서 FRESH의 합의값을
# 그대로 캐리 행에 복사할 수 있다(값이 하나로 유일할 때만).
#   CALIBRATION_MODE·CAL_N_TRADES는 캐리 블록이 `get_calibration_mode`를 호출해
#   놓고도 캐리 행에 **기록하지 않았다**(pipeline_calibrate 539행 부근).
BATCH_SCALAR_FROM_FRESH = ("CALIBRATION_MODE", "CAL_N_TRADES",
                           "벤치_60d_KOSPI_%", "벤치_60d_KOSDAQ_%", "시총기준일")

# **의도적으로 비워 두는 컬럼** — 재계산이 불가능하거나, 해당 단계를 돌리지 않았다.
#   이 목록이 곧 "캐리 행에 없어도 되는 컬럼"의 계약이다. FRESH가 항상 채우는
#   컬럼이 캐리에 없으면서 여기 없으면 tests/test_v60_carry_backfill.py가 실패한다.
#   **없는 값을 만들어내지 않기 위한 목록이지, 면제 목록이 아니다** — 새 항목을
#   넣을 때는 "왜 계산할 수 없는가"를 적어야 한다.
INTENTIONALLY_BLANK = {
    "NEWS_SCORE": "뉴스 단계(Stage 4)를 캐리 행에는 돌리지 않았다 — 0.0은 "
                  "'확인했고 특이사항 없음'을 뜻하므로 넣지 않는다",
    "STRATEGY": "전략 엔진을 캐리 행에는 돌리지 않았다",
    "STRATEGY_SCORE": "전략 엔진을 캐리 행에는 돌리지 않았다",
    "STRATEGY_HORIZON": "전략 엔진을 캐리 행에는 돌리지 않았다",
    "AI_COMMENT": "LLM 코멘트를 캐리 행에는 생성하지 않았다",
    "개인순매수": "수급은 당일 유니버스에 대해서만 수집한다. 유니버스에서 빠진 "
              "캐리 종목은 수집 대상이 아니어서 값이 없다 — 0은 '순매수 없음'"
              "이라는 관측값이므로 대신 쓸 수 없다",
    "ML_RAW_SCORE": "legacy 캐리는 OHLCV 재수집이 실패한 행이다(그래서 legacy다). "
                    "ML 입력이 없으므로 점수를 만들 수 없다",
    "ML_TRUSTED": "ML을 돌리지 못했으므로 신뢰 여부를 말할 수 없다 — "
                  "기본값 True/False 어느 쪽도 거짓말이 된다",
    "ML_EFFECTIVE_SCORE": "ML_RAW_SCORE에서 파생되므로 원점수가 없으면 만들 수 없다",
    "V23_ATR_Pct": "OHLCV에서 계산하는 지표 — legacy 캐리는 OHLCV가 없다",
    "V23_OBV_Slope": "OHLCV에서 계산하는 지표 — legacy 캐리는 OHLCV가 없다",
    "V23_Upper_Shadow_Ratio": "OHLCV에서 계산하는 지표 — legacy 캐리는 OHLCV가 없다",
    "V23_SIGNAL_STATUS": "OHLCV에서 계산하는 지표 — legacy 캐리는 OHLCV가 없다",
}
CARRY_BLANK_REASON = "캐리 재분석 — 미수집"

# 캐리 상태 사유. legacy 행은 파이프라인이 직접 문구를 넣지만
# **CARRY_REFRESHED 행은 아무 사유도 받지 못했다** (2026-08-12 배치 실측:
# legacy 38행은 사유 있음, refreshed 12행은 공백). 화면에는 상태가 'CARRY'로만
# 뜨고 왜 그런지는 말하지 않는다 — 보유 종목인데 설명이 없다.
ROUTE_REASON_BY_MODE = {
    "CARRY_REFRESHED": "보유 이어받기 — 당일 OHLCV로 재분석 완료 "
                       "(신규 진입 판단이 아니라 보유 관리 대상)",
    "CARRY_LEGACY": "보유 이어받기 — 당일 재분석 실패로 이전 스냅샷 사용 "
                    "(지표가 묵었다)",
}

# 복원 실패가 이 비율을 넘으면 경고로 승격
BACKFILL_WARN_RATIO = 0.10


def carry_mask(df: pd.DataFrame) -> pd.Series:
    """CARRY 행 마스크. ROW_BUILD_MODE가 없으면 전부 FRESH로 본다."""
    if df is None or len(df) == 0:
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)
    if BUILD_MODE_COL not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    mode = df[BUILD_MODE_COL].astype("object")
    return mode.isin(CARRY_MODES).fillna(False).astype(bool)


def _fill_blanks(out: pd.DataFrame, mask: pd.Series, col: str, values) -> int:
    """`mask` 행 중 `col`이 빈 곳만 채운다. 채운 건수를 돌려준다.

    이미 값이 있는 행은 건드리지 않는다 — 캐리 행이 스냅샷에서 물려받은 값이
    있다면 그것이 그 시점의 진실이고, 여기서 덮어쓸 근거가 없다.
    """
    if col not in out.columns:
        out[col] = pd.NA
    target = mask & blank_mask(out, col)
    n = int(target.sum())
    if not n:
        return 0
    orig_dtype = out[col].dtype
    out[col] = out[col].astype("object")
    if isinstance(values, pd.Series):
        out.loc[target, col] = values.loc[target]
    else:
        out.loc[target, col] = values
    # 원래 dtype 복원. object로 남기면 숫자 컬럼이 조용히 object가 되어
    # 뒤쪽 연산·차트가 엉뚱하게 동작한다 (v55.4에서 pandas 2/3 dtype 계약으로
    # 같은 유형의 사고를 겪었다). 복원 실패는 로그로 남긴다.
    try:
        if pd.api.types.is_numeric_dtype(orig_dtype):
            # **값을 잃는 복원은 하지 않는다.** 전부 NaN이던 float 컬럼에 문자열
            # 사유를 채우는 경우가 있는데(NEWS_REASON), 거기서 to_numeric을 하면
            # 방금 채운 문자열이 NaN으로 지워진다 — 복원이 데이터를 삼킨다.
            cand = pd.to_numeric(out[col], errors="coerce")
            if int(cand.isna().sum()) <= int(out[col].isna().sum()):
                out[col] = cand.astype(orig_dtype)
            else:
                logger.info(
                    f"[v60] {col} dtype 복원 생략 — {orig_dtype}로 되돌리면 "
                    f"방금 채운 비수치 값이 지워진다")
        elif str(orig_dtype) != "object":
            out[col] = out[col].astype(orig_dtype)
    except Exception as e:
        logger.warning(
            f"[v60] {col} dtype 복원 실패 ({orig_dtype} 유지 불가): {e}")
    return n


def backfill_carry_rows(
    df: pd.DataFrame,
    *,
    bench_map: Optional[dict] = None,
    mcap_ymd=None,
    individual_net_map: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """CARRY 행에 Stage 3/4가 부여했어야 할 컬럼을 결정적으로 복원.

    반환: (df, report). 리포트는 컬럼별 복원 건수와 **의도적으로 비운 컬럼**을
    함께 담는다 — 조용히 채우면 근본 원인이 또 숨는다(v55.4~v59 반복 유형).
    """
    report = {
        "ok": True, "rows": 0, "carry_rows": 0, "filled": {},
        "left_blank": {}, "sector_repair": None, "still_blank": {},
        "note": "",
    }
    if df is None or len(df) == 0:
        report["ok"] = False
        report["note"] = "빈 프레임"
        return df, report

    report["rows"] = int(len(df))
    mask = carry_mask(df)
    report["carry_rows"] = int(mask.sum())
    if not bool(mask.any()):
        return df, report

    out = df.copy()

    # ── 0) 업종 복구 (v59 SSOT 재사용 — 다른 분류기를 쓰면 기준이 갈린다)
    out, sec_rep = repair_sector(out)
    report["sector_repair"] = sec_rep
    mask = carry_mask(out)          # repair_sector가 copy를 돌려주므로 재계산

    # ── 1) DISPLAY_SCORE 별칭 3종 (본체)
    if DISPLAY_COL in out.columns:
        disp = pd.to_numeric(out[DISPLAY_COL], errors="coerce")
        for col in DISPLAY_ALIASES:
            n = _fill_blanks(out, mask, col, disp)
            if n:
                report["filled"][col] = n
    else:
        report["note"] = f"{DISPLAY_COL} 없음 — 별칭 복원 불가"
        logger.warning(f"[v60] {report['note']}")

    # ── 2) 업종_상세 = 업종 (표시 일관성)
    if SECTOR_RAW_COL in out.columns:
        n = _fill_blanks(out, mask, SECTOR_DETAIL_COL,
                         out[SECTOR_RAW_COL].astype("object"))
        if n:
            report["filled"][SECTOR_DETAIL_COL] = n

    # ── 3) 섹터 집계값 — 같은 섹터의 **비캐리(FRESH)** 행에서 가져온다
    fresh = ~mask
    if SECTOR_COL in out.columns and bool(fresh.any()):
        for col in SECTOR_AGG_COLS:
            if col not in out.columns:
                continue
            src = out.loc[fresh & out[col].notna(), [SECTOR_COL, col]]
            if src.empty:
                continue
            # 섹터별 대표값 — 같은 섹터 안에서는 동일한 값이므로 first로 충분
            lut = src.groupby(SECTOR_COL)[col].first()
            mapped = out[SECTOR_COL].astype("object").map(lut)
            n = _fill_blanks(out, mask, col, mapped)
            if n:
                report["filled"][col] = n

    # ── 4) 벤치마크 60일 (배치 스칼라)
    if bench_map:
        for col, key in (("벤치_60d_KOSPI_%", "KOSPI"),
                         ("벤치_60d_KOSDAQ_%", "KOSDAQ")):
            v = (bench_map.get(key) or {}).get(60)
            if v is None:
                continue
            n = _fill_blanks(out, mask, col, float(v))
            if n:
                report["filled"][col] = n

    # ── 5) 시총기준일 (배치 스칼라)
    if mcap_ymd:
        n = _fill_blanks(out, mask, "시총기준일", str(mcap_ymd))
        if n:
            report["filled"]["시총기준일"] = n

    # ── 5b) 배치 스칼라 — FRESH 합의값 복사 (값이 **하나로 유일할 때만**)
    #   배치 전체가 공유하는 값이므로 캐리 행도 같은 값을 가져야 한다. 여러
    #   값이 섞여 있으면 배치 스칼라가 아니라는 뜻이니 손대지 않는다.
    fresh_rows = ~mask
    if bool(fresh_rows.any()):
        for col in BATCH_SCALAR_FROM_FRESH:
            if col not in out.columns:
                continue
            vals = out.loc[fresh_rows & out[col].notna(), col]
            uniq = pd.unique(vals.astype("object"))
            if len(uniq) != 1:
                if len(uniq) > 1:
                    report.setdefault("skipped_non_scalar", []).append(
                        f"{col}({len(uniq)}종)")
                continue
            n = _fill_blanks(out, mask, col, uniq[0])
            if n:
                report["filled"][col] = report["filled"].get(col, 0) + n

    # ── 6) 개인순매수 — 맵이 있을 때만. 없으면 0으로 채우지 않는다
    #      (0은 '순매수 없음'이라는 관측값이고, 미수집과 다르다).
    if individual_net_map and "종목코드" in out.columns:
        codes = out["종목코드"].astype(str).str.zfill(6)
        mapped = codes.map({str(k).zfill(6): v
                            for k, v in individual_net_map.items()})
        n = _fill_blanks(out, mask, "개인순매수", mapped)
        if n:
            report["filled"]["개인순매수"] = n

    # ── 6b) 캐리 상태 사유 — 상태만 보여주고 이유를 말하지 않는 일이 없게 한다
    if BUILD_MODE_COL in out.columns:
        mode = out[BUILD_MODE_COL].astype("object")
        for mode_name, text in ROUTE_REASON_BY_MODE.items():
            n = _fill_blanks(out, mask & (mode == mode_name),
                             "ROUTE_REASON", text)
            if n:
                report["filled"]["ROUTE_REASON"] = (
                    report["filled"].get("ROUTE_REASON", 0) + n)

    # ── 7) 돌리지 않은 단계는 사유를 명시한다 (조용한 공백 금지)
    for col, why in INTENTIONALLY_BLANK.items():
        if col not in out.columns:
            continue
        n = int((mask & blank_mask(out, col)).sum())
        if n:
            report["left_blank"][col] = {"rows": n, "why": why}
    for col, reason_col in (("NEWS_SCORE", "NEWS_REASON"),):
        if reason_col not in out.columns:
            continue
        blank_val = mask & blank_mask(out, col)
        n = _fill_blanks(out, blank_val, reason_col, CARRY_BLANK_REASON)
        if n:
            report["filled"][reason_col] = n
    if "HAS_NEWS" in out.columns:
        _bn = mask & blank_mask(out, "NEWS_SCORE")
        if bool(_bn.any()):
            out.loc[_bn, "HAS_NEWS"] = False

    # ── 8) 남은 공백 집계 (계약 위반 감시용)
    for col in tuple(DISPLAY_ALIASES) + SECTOR_AGG_COLS + (
            SECTOR_COL, SECTOR_DETAIL_COL):
        if col not in out.columns:
            continue
        n = int((mask & blank_mask(out, col)).sum())
        if n:
            report["still_blank"][col] = n
    return out, report


def carry_backfill_line(report: Optional[dict]) -> str:
    """배치 로그용 한 줄. 캐리 행이 없으면 빈 문자열."""
    if not report or not report.get("ok") or not report.get("carry_rows"):
        return ""
    bits = [f"캐리 {report['carry_rows']}/{report['rows']}행"]
    filled = report.get("filled") or {}
    if filled:
        total = sum(filled.values())
        top = sorted(filled.items(), key=lambda kv: -kv[1])[:4]
        bits.append(f"복원 {total}칸({len(filled)}컬럼: "
                    + ", ".join(f"{k}×{v}" for k, v in top) + ")")
    else:
        bits.append("복원 0칸")
    if report.get("still_blank"):
        bits.append("미복원 " + ", ".join(
            f"{k}×{v}" for k, v in sorted(report["still_blank"].items())))
    if report.get("left_blank"):
        bits.append(f"의도적 공백 {len(report['left_blank'])}컬럼"
                    " (미수행 단계 — 값을 만들지 않음)")
    if report.get("note"):
        bits.append(report["note"])
    return " · ".join(bits)


def is_alarming(report: Optional[dict]) -> bool:
    """계약이 깨졌다는 신호인가.

    복원 대상이 남아 있다는 것은 CARRY 경로가 또 다른 컬럼을 놓치고 있다는
    뜻이다. 의도적 공백(INTENTIONALLY_BLANK)은 여기 포함되지 않는다.
    """
    if not report or not report.get("ok"):
        return False
    carry = int(report.get("carry_rows", 0) or 0)
    if not carry:
        return False
    worst = max(report.get("still_blank", {}).values(), default=0)
    return worst > 0 and (worst / carry) >= BACKFILL_WARN_RATIO
