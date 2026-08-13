# -*- coding: utf-8 -*-
"""[v59] 업종 결측 복구 + 데이터 공백을 조용히 넘기지 않는다.

■ 무엇이 있었나 (2026-08-10 실측)
  사용자가 시장 탭에서 `❌ 로딩 실패: ('None entries cannot have not-None
  children', 업종_대분류 NaN 종목명 녹십자 ...)`를 봤다. plotly 트리맵이
  `path=[업종_대분류, 종목명]`으로 계층을 만들 때 **부모가 결측이고 자식은
  있는 행**을 거부한 것이다.

  파고 보니 결측은 우연이 아니었다. 2026-08-07 배치 326종목 중 **30종목(9.2%)**,
  8/06은 278종목 중 32종목(11.5%)이 업종_대분류 결측이었다. 그리고 그 30행은
  업종(상세)은 채워져 있는데(녹십자 = '의약품 제조업') 다음 18개 컬럼이
  **전부** 비어 있었다:
    업종_대분류 · 업종_상세 · SECTOR_RET_5D · SECTOR_RS · SECTOR_RANK ·
    NEWS_SCORE · 개인순매수 · STRATEGY(3종) · 시총기준일 ·
    LDY_SCORE · TOTAL_SCORE · RANK_SCORE · 벤치_60d_*(2종) ·
    CALIBRATION_MODE · CAL_N_TRADES
  ALPHA_SCORE·ROUTE·FINAL_SCORE·EBS·TOP_PICK은 채워져 있었다.
  → 즉 이 행들은 **섹터/뉴스/전략 단계 이후에 파이프라인에 합류**했고,
    그 단계들이 부여하는 컬럼을 받지 못했다.

■ 왜 크래시보다 나쁜 문제가 있었나
  kelly_calibrator의 섹터 모멘텀이 `astype(str).fillna("?")`로 결측을 묶었다.
  그 결과 **서로 아무 관계 없는 30종목이 '?'라는 가짜 섹터 하나**가 되어
  공통 평균수익률을 공유하고 같은 SECTOR_MOM_FACTOR를 받았다. 이 배수는
  ALPHA_SIZE_MULT → 켈리_수량으로 이어진다 — **실제 주문 수량이 오염된다.**
  v41이 검증한 것은 '강섹터 × 고알파' 상호작용이고, 섹터를 모르는 종목에
  섹터 배수를 주는 것은 그 검증 범위 밖이다. (v59에서 그쪽은 중립 1.0으로 고쳤다.)

■ 이 모듈이 하는 일
  1. **복구**: 업종(상세)이 있으면 classify_big_sector로 업종_대분류를 다시 만든다.
     원래 파이프라인이 하던 것과 **같은 분류기**를 쓴다(다른 규칙으로 채우면
     복구된 행과 정상 행이 다른 기준으로 섞인다).
  2. **표시용 라벨 분리**: 복구도 불가능한 행은 엔진 축에서는 결측으로 남기고
     (가짜 섹터 금지), 표시 계층에서만 '미분류'로 묶는다.
  3. **조용히 넘기지 않는다**: 복구 건수·복구 실패 건수·결측 컬럼 상위 목록을
     리포트로 남기고 배치 로그에 한 줄 찍는다. 결측을 조용히 채우면 근본 원인
     (행이 늦게 합류하는 것)이 또 숨는다. v55.4~v58에서 반복해 잡은 유형이다.

■ 하지 않는 일
  결측 행을 삭제하지 않는다. 그 행들도 ALPHA_SCORE·ROUTE가 있는 실제 후보다.
  또 LDY_SCORE 등 다른 결측 컬럼을 임의로 채우지 않는다 — 그건 없는 점수를
  만들어내는 것이고, 이 프로젝트가 계속 걷어내 온 과신이다.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("sector_repair")

SECTOR_COL = "업종_대분류"
SECTOR_DETAIL_COL = "업종_상세"
SECTOR_RAW_COL = "업종"
DISPLAY_UNKNOWN = "미분류"          # 표시 전용 라벨 (엔진 축에는 쓰지 않는다)
_BLANKS = ("", "nan", "none", "-", "?", DISPLAY_UNKNOWN.lower())

# 결측 비율이 이보다 크면 경고를 승격한다 (조용한 열화 방지)
MISSING_WARN_RATIO = 0.05


def blank_mask(df: pd.DataFrame, column: str) -> pd.Series:
    """결측/공백/문자열 'nan' 등을 한 번에 잡는다 (pandas 2/3 모두)."""
    if df is None or len(df) == 0 or column not in df.columns:
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)
    s = df[column]
    na = s.isna()
    txt = s.astype("object").map(
        lambda v: True if v is None else str(v).strip().lower() in _BLANKS)
    return (na | txt.fillna(True)).astype(bool)


def _restore_dtype(out: pd.DataFrame, column: str, orig_dtype) -> None:
    """복구 대입 후 원래 dtype으로 되돌린다 (조용한 dtype 열화 방지)."""
    if str(orig_dtype) == "object":
        return
    try:
        out[column] = out[column].astype(orig_dtype)
    except Exception as e:
        logger.warning(f"[v60] {column} dtype 복원 실패 ({orig_dtype}): {e}")


def repair_sector(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """업종_대분류 결측을 업종(상세)에서 재분류해 복구. (df, 리포트) 반환.

    파이프라인과 같은 분류기(classify_big_sector)를 쓴다. 분류기를 못 불러오면
    복구를 건너뛰되 그 사실을 리포트에 남긴다(조용히 성공한 척하지 않는다).
    """
    report = {
        "ok": True, "rows": 0, "missing_before": 0, "repaired": 0,
        "still_missing": 0, "missing_ratio_before": 0.0,
        "missing_ratio_after": 0.0, "classifier": None,
        "samples_still_missing": [], "note": "",
    }
    if df is None or len(df) == 0:
        report["ok"] = False
        report["note"] = "빈 프레임"
        return df, report

    out = df.copy()
    report["rows"] = int(len(out))
    if SECTOR_COL not in out.columns:
        out[SECTOR_COL] = pd.NA

    miss = blank_mask(out, SECTOR_COL)
    report["missing_before"] = int(miss.sum())
    report["missing_ratio_before"] = round(float(miss.mean()), 4)
    if not bool(miss.any()):
        return out, report

    # 분류기 SSOT는 sector_utils. collector는 그것을 재수출하므로 폴백으로만 둔다
    # (파이프라인이 쓰는 것과 다른 규칙으로 채우면 복구 행과 정상 행의 기준이
    #  달라진다).
    classifier = None
    _last_err = None
    for _mod in ("sector_utils", "collector"):
        try:
            classifier = __import__(_mod, fromlist=["classify_big_sector"]).classify_big_sector
            break
        except Exception as e:
            _last_err = e
    if classifier is None:
        report["note"] = f"classify_big_sector 참조 실패 — 복구 불가 ({_last_err})"
        logger.warning(f"[v59] {report['note']}")
    report["classifier"] = getattr(classifier, "__module__", None)

    if classifier is not None and SECTOR_RAW_COL in out.columns:
        idx = out.index[miss]
        names = out.loc[idx, "종목명"] if "종목명" in out.columns else pd.Series(
            "", index=idx)
        raws = out.loc[idx, SECTOR_RAW_COL]
        fixed = {}
        for i in idx:
            try:
                v = classifier(str(names.get(i, "") or ""),
                               str(raws.get(i, "") or ""))
            except Exception as e:      # 분류기 예외가 배치를 죽이면 안 된다
                logger.warning(f"[v59] 업종 재분류 실패 idx={i}: {e}")
                continue
            if v is not None and str(v).strip().lower() not in _BLANKS:
                fixed[i] = str(v).strip()
        if fixed:
            # [v60] dtype 계약 유지 — object로 대입한 뒤 **원래 dtype으로 되돌린다**.
            #   v59에서는 되돌리지 않아 pandas 3의 str dtype이 조용히 object가 됐다
            #   (v55.4에서 같은 유형의 dtype 계약 사고를 이미 겪었다).
            _orig = out[SECTOR_COL].dtype
            out[SECTOR_COL] = out[SECTOR_COL].astype("object")
            for i, v in fixed.items():
                out.at[i, SECTOR_COL] = v
            report["repaired"] = len(fixed)
            # 업종_상세도 같이 비어 있으면 원본 업종으로 채운다(표시 일관성).
            if SECTOR_DETAIL_COL in out.columns:
                _orig_d = out[SECTOR_DETAIL_COL].dtype
                dmiss = blank_mask(out, SECTOR_DETAIL_COL)
                out[SECTOR_DETAIL_COL] = out[SECTOR_DETAIL_COL].astype("object")
                for i in fixed:
                    if bool(dmiss.get(i, False)):
                        out.at[i, SECTOR_DETAIL_COL] = str(raws.get(i, "") or "")
                _restore_dtype(out, SECTOR_DETAIL_COL, _orig_d)
            _restore_dtype(out, SECTOR_COL, _orig)

    after = blank_mask(out, SECTOR_COL)
    report["still_missing"] = int(after.sum())
    report["missing_ratio_after"] = round(float(after.mean()), 4)
    if bool(after.any()) and "종목명" in out.columns:
        report["samples_still_missing"] = [
            str(x) for x in out.loc[after, "종목명"].head(8).tolist()]
    return out, report


def display_sector(df: pd.DataFrame, column: str = SECTOR_COL) -> pd.Series:
    """표시 전용 섹터 라벨 — 결측을 '미분류'로 묶는다.

    **엔진 축에는 쓰지 마라.** 서로 무관한 종목을 한 섹터로 취급하면
    v41이 검증한 상호작용이 오염된다(v59에서 kelly가 그 방식이었다).
    """
    if df is None or len(df) == 0 or column not in df.columns:
        return pd.Series(dtype="object")
    s = df[column].astype("object")
    return s.where(~blank_mask(df, column), DISPLAY_UNKNOWN).astype(str)


def sector_repair_line(report: Optional[dict]) -> str:
    """배치 로그용 한 줄. 문제 없으면 빈 문자열."""
    if not report or not report.get("ok"):
        return ""
    if not report.get("missing_before"):
        return ""
    bits = [f"업종 결측 {report['missing_before']}/{report['rows']}종목"
            f"({report['missing_ratio_before']:.1%})",
            f"복구 {report['repaired']}건"]
    if report.get("still_missing"):
        bits.append(f"미복구 {report['still_missing']}건")
        if report.get("samples_still_missing"):
            bits.append("예: " + ", ".join(report["samples_still_missing"][:4]))
    if report.get("note"):
        bits.append(report["note"])
    return " · ".join(bits)


def is_alarming(report: Optional[dict]) -> bool:
    """근본 원인(행이 늦게 합류)이 남아 있다는 신호인가.

    복구가 됐더라도 **애초에 결측이 많았다는 사실 자체가 문제**다 — 그 행들은
    업종만 빈 게 아니라 LDY_SCORE·NEWS_SCORE 등 17개 컬럼도 비어 있었다.
    복구로 크래시는 막지만 원인은 남는다. 그래서 비율로 경고한다.
    """
    if not report or not report.get("ok"):
        return False
    return (float(report.get("missing_ratio_before", 0.0)) >= MISSING_WARN_RATIO
            or int(report.get("still_missing", 0)) > 0)
