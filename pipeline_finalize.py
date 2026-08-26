# -*- coding: utf-8 -*-
"""pipeline_finalize.py — Stage 6: 저장 + 발송 + 검증 [v20.6.4]
═══════════════════════════════════════════════════════════════════
[v20.6.4] After-market sidecar 분리 — 추천 CSV 원본 불변 보장
 - recommend_latest.csv는 분석 시점 기준 불변
 - 시간외 가격은 aftermarket_prices_latest.csv에 별도 저장
"""
import os, logging, numpy as np, pandas as pd
from pipeline_context import PipelineContext
from shared_log import log, OUT_DIR, UTF8, ensure_dir
from collector_config import Route
from macro_filter import label_market_temp
from telegram_sender import send_telegram_auto
from validation import run_reality_check

logger = logging.getLogger(__name__)


# [v3.7.27 추가 · v3.7.29 강화] CONFIG_SNAPSHOT 로드 유틸
# Single source of truth: config는 JSON 파일에서만 읽는다.
# CSV에는 CONFIG_VERSION 문자열만 남기므로, 전체 snapshot이 필요하면 이 함수 사용.
def load_config_snapshot(trade_ymd: str = None) -> dict:
    """CONFIG_SNAPSHOT을 JSON 파일에서 로드 (fallback 체인 적용).

    Args:
        trade_ymd: YYYYMMDD 문자열. None이면 latest 파일 사용.

    Returns:
        dict: 설정 스냅샷. 모든 fallback 실패 시 빈 dict (참조 코드가 안전하게 계속 진행 가능).

    Fallback 순서:
      1) data/config_snapshot_{trade_ymd}.json (지정일)
      2) data/config_snapshot_latest.json      (최신)
      3) collector_config.DEFAULT_CONFIG.snapshot_json() (런타임)
      4) {} 빈 dict

    예전 CSV 호환 코드를 바꿀 때 사용:
        # Before (v3.7.26):
        #   snapshot = json.loads(df.iloc[0]["CONFIG_SNAPSHOT"])
        # After (v3.7.27+):
        #   from pipeline_finalize import load_config_snapshot
        #   snapshot = load_config_snapshot(trade_ymd)
    """
    import json
    from pathlib import Path

    # 1) 지정일 파일
    if trade_ymd:
        try:
            path = Path(OUT_DIR) / f"config_snapshot_{trade_ymd}.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug(f"CONFIG_SNAPSHOT dated 로드 실패 ({trade_ymd}): {e}")

    # 2) latest alias
    try:
        path_latest = Path(OUT_DIR) / "config_snapshot_latest.json"
        if path_latest.exists():
            return json.loads(path_latest.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"CONFIG_SNAPSHOT latest 로드 실패: {e}")

    # 3) 런타임 재생성 (collector_config에서 직접)
    try:
        from collector_config import DEFAULT_CONFIG as _snap
        return json.loads(_snap.snapshot_json())
    except Exception as e:
        logger.debug(f"CONFIG_SNAPSHOT 런타임 생성 실패: {e}")

    # 4) 빈 dict
    logger.info("CONFIG_SNAPSHOT 사용 불가 — 빈 dict 반환 (참조 코드는 계속 진행)")
    return {}


# ═══════════════════════════════════════════════════
#  [v22] finalize_sort SSOT + adaptive IS_NOW_ENTRY
#  ─ 참조: SwingPicker_v22_Final_Consolidated.md §2.2.6
# ═══════════════════════════════════════════════════

# SORT_SPEC — 8축 정렬의 단일 소스
# TOP_PICK × IS_NOW_ENTRY × ROUTE_PRIORITY × ELITE × RR × BALANCE × ENTRY_GAP × DISPLAY_SCORE
_SORT_ROUTE_PRIORITY = {
    "ATTACK": 1, "ARMED": 2, "WAIT": 3, "NEUTRAL": 4,
    "OVERHEAT": 5, "EXIT_WARNING": 6, "CARRY": 7,
}


# ═══════════════════════════════════════════════════════════
# [v3.9.6] PRE_ENTRY_RISK 컬럼 부여
# ═══════════════════════════════════════════════════════════
# 검증된 룰 (simulate_pre_entry_risk_shadow.py --mode rwf, B_red 5/5 통과):
#   RED:    STRUCT_SCORE 70 ≤ s ≤ 85 AND VWAP_GAP > 8
#   ORANGE: STRUCT_SCORE < 90 AND VWAP_GAP > 15  (RED와 겹치면 RED 우선)
#   GREEN:  그 외
# 표시 전용 — 자동 제외/감점 없음. 회원이 "이 종목 위험" 인지하게.
# [v3.9.7] 경계값 보정: 코드와 문구 일치 — STRUCT == 85.0도 RED에 포함
PRE_RISK_STRUCT_LO_CSV = 70.0
PRE_RISK_STRUCT_HI_CSV = 85.0   # 포함 (<=)
PRE_RISK_VWAP_RED_CSV = 8.0
PRE_RISK_STRUCT_TOP_CSV = 90.0
PRE_RISK_VWAP_ORANGE_CSV = 15.0


def add_entry_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """[v3.9.6] recommend CSV에 ENTRY_RISK_FLAG / LEVEL / REASON / RULE 컬럼 부여.
    
    원본 df는 보존, 컬럼만 추가. STRUCT_SCORE / VWAP_GAP 없는 행은 GREEN 처리.
    """
    if df is None or len(df) == 0:
        return df

    # 안전 추출 (없으면 NaN → 0)
    struct = pd.to_numeric(df.get("STRUCT_SCORE", 0), errors="coerce").fillna(0)
    vwap = pd.to_numeric(df.get("VWAP_GAP", 0), errors="coerce").fillna(0)

    # RED 마스크: STRUCT 70 ≤ s ≤ 85 AND VWAP>8
    # ([v3.9.7] HI를 inclusive로 — 문구 "70~85"와 코드 일치, 85.0 경계 누락 방지)
    red_mask = (
        (struct >= PRE_RISK_STRUCT_LO_CSV)
        & (struct <= PRE_RISK_STRUCT_HI_CSV)
        & (vwap > PRE_RISK_VWAP_RED_CSV)
    )
    # ORANGE 마스크: STRUCT<90 AND VWAP>15 (RED와 겹치면 RED 우선)
    orange_mask = (
        (struct < PRE_RISK_STRUCT_TOP_CSV)
        & (vwap > PRE_RISK_VWAP_ORANGE_CSV)
        & ~red_mask
    )

    level = np.where(red_mask, "RED",
                     np.where(orange_mask, "ORANGE", "GREEN"))
    flag = np.where((red_mask | orange_mask), 1, 0)
    rule = np.where(red_mask, "B_RED",
                    np.where(orange_mask, "C_ORANGE", ""))

    # REASON — 한글 설명 (회원이 읽기 좋게)
    reason = np.where(
        red_mask,
        "STRUCT 70~85 위험 구간 + VWAP_GAP > 8% 과열",
        np.where(
            orange_mask,
            "STRUCT < 90 + VWAP_GAP > 15% 강한 과열",
            "",
        )
    )

    df = df.copy()
    df["ENTRY_RISK_FLAG"] = flag.astype(int)
    df["ENTRY_RISK_LEVEL"] = level
    df["ENTRY_RISK_RULE"] = rule
    df["ENTRY_RISK_REASON"] = reason
    return df



# ═══════════════════════════════════════════════════════════
# [v22.3.10] ENTRY_EDGE_SCORE shadow production display
# ═══════════════════════════════════════════════════════════
# PRE_ENTRY_RISK shadow에서 가장 유망했던 B_red 룰을 하드 차단이 아니라
# 표시/감점 전용 컬럼으로 노출한다. BUY_NOW_ELIGIBLE / TOP_PICK은 절대 변경하지 않는다.
ENTRY_EDGE_BASE_SCORE = 100.0
ENTRY_EDGE_B_RED_PENALTY = 15.0


def add_entry_edge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ENTRY_EDGE shadow 컬럼을 recommend CSV에 부여한다.

    목적:
      - PRE_ENTRY_RISK B_red(STRUCT 70~85 AND VWAP_GAP>8)를
        ENTRY_EDGE_SCORE 감점/주의 표시로만 반영한다.
      - BUY_NOW_ELIGIBLE, BUY_NOW_GRADE, TOP_PICK 등 공식 추천 계약은 변경하지 않는다.

    추가 컬럼:
      ENTRY_EDGE_SCORE       : 100 기준 shadow 점수. B_red면 85.
      ENTRY_EDGE_LEVEL       : GREEN / CAUTION. 현재 하드 RED 차단 없음.
      ENTRY_EDGE_RULE        : B_RED_SHADOW 또는 빈값.
      ENTRY_EDGE_REASON      : UI 표시용 한글 사유.
      ENTRY_EDGE_SHADOW_FLAG : 감점 발생 여부(0/1).
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # ENTRY_RISK_LEVEL/RULE이 이미 있으면 SSOT로 사용하고,
    # legacy/단위 테스트용 입력처럼 없으면 원 지표로 B_red를 재계산한다.
    risk_level = out.get("ENTRY_RISK_LEVEL", pd.Series("", index=out.index))
    risk_rule = out.get("ENTRY_RISK_RULE", pd.Series("", index=out.index))
    risk_level = risk_level.astype(str).str.strip().str.upper()
    risk_rule = risk_rule.astype(str).str.strip().str.upper()

    struct = pd.to_numeric(out.get("STRUCT_SCORE", 0), errors="coerce").fillna(0)
    vwap = pd.to_numeric(out.get("VWAP_GAP", 0), errors="coerce").fillna(0)
    b_red_from_metrics = (
        (struct >= PRE_RISK_STRUCT_LO_CSV)
        & (struct <= PRE_RISK_STRUCT_HI_CSV)
        & (vwap > PRE_RISK_VWAP_RED_CSV)
    )
    b_red = (risk_rule == "B_RED") | (risk_level == "RED") | b_red_from_metrics

    score = pd.Series(ENTRY_EDGE_BASE_SCORE, index=out.index, dtype="float64")
    score.loc[b_red] = (ENTRY_EDGE_BASE_SCORE - ENTRY_EDGE_B_RED_PENALTY)

    # 이 패치는 production hard block이 아니므로 RED 레벨을 만들지 않는다.
    # 공식 신규매수 차단 여부는 기존 BUY_NOW_ELIGIBLE 계약만 따른다.
    level = np.where(b_red, "CAUTION", "GREEN")
    rule = np.where(b_red, "B_RED_SHADOW", "")
    reason = np.where(
        b_red,
        "B_red shadow 감점 -15: STRUCT 70~85 + VWAP_GAP>8 위험 조합 · 공식 매수 차단 아님",
        "",
    )

    out["ENTRY_EDGE_SCORE"] = score.round(1)
    out["ENTRY_EDGE_LEVEL"] = level
    out["ENTRY_EDGE_RULE"] = rule
    out["ENTRY_EDGE_REASON"] = reason
    out["ENTRY_EDGE_SHADOW_FLAG"] = b_red.astype(int)
    return out



# ═══════════════════════════════════════════════════════════
# [v3.9.24] Official Buy Funnel & Macro Regime Shadow
# ═══════════════════════════════════════════════════════════
# 공식 추천식을 완화하지 않고, recommend CSV에 "왜 공식 신규매수 0개인지"를
# 설명하는 퍼널/후보 유형/shadow 시뮬레이션 컬럼만 추가한다.
# 절대 계약:
#   - TOP_PICK / BUY_NOW_ELIGIBLE / BUY_NOW_GRADE / BUY_NOW_PASS 변경 금지
#   - scoring_engine.py 점수 산식 변경 금지
#   - MACRO shadow는 production hard block 완화가 아니라 진단 전용

def _v3924_truthy(value) -> bool:
    text = str(value).strip().upper()
    return text in {"1", "1.0", "TRUE", "T", "Y", "YES", "BUY", "PASS"}


def _v3924_num_series(df: pd.DataFrame, names, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _v3924_text_series(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name in df.columns:
        return df[name].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def _v3924_flag_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(False, index=df.index)
    return df[name].map(_v3924_truthy).fillna(False).astype(bool)


def _v3924_extract_fx_level(macro_msg: str):
    """`환율 1515원 [05/25]` / `USD/KRW: 1495.5`에서 환율 레벨을 추출한다."""
    import re

    msg = str(macro_msg or "")
    patterns = [
        r"환율\s*([0-9,]+(?:\.\d+)?)\s*원",
        r"USD\s*/?\s*KRW\s*[:=]?\s*([0-9,]+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, msg, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError as e:
                logger.debug("v3.9.24 FX level parse skip: %s", e)
    return None


def _v3924_ebs_pass_mask(df: pd.DataFrame) -> pd.Series:
    if "PASS_EBS" in df.columns:
        return _v3924_flag_series(df, "PASS_EBS")
    if "EBS_STATUS" in df.columns:
        s = _v3924_text_series(df, "EBS_STATUS").str.upper()
        return s.str.contains("PASS|통과", na=False)
    if "EBS" in df.columns:
        s = _v3924_text_series(df, "EBS").str.upper()
        return s.str.contains("PASS|통과", na=False) | s.str.match(r"^[6-9]/|^10/", na=False)
    return pd.Series(False, index=df.index)


def add_official_buy_funnel_columns(
    df: pd.DataFrame,
    macro_risk: str = "",
    market_breadth=None,
    macro_msg: str = "",
) -> pd.DataFrame:
    """[v3.9.24] 공식매수 퍼널/후보 유형/macro shadow 컬럼을 추가한다.

    이 함수는 measurement/display 전용이다. TOP_PICK, BUY_NOW_ELIGIBLE,
    BUY_NOW_PASS, BUY_NOW_GRADE 값은 절대 수정하지 않는다.
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    top_pick = _v3924_flag_series(out, "TOP_PICK")
    eligible = _v3924_flag_series(out, "BUY_NOW_ELIGIBLE")
    buy_pass = _v3924_flag_series(out, "BUY_NOW_PASS")
    route = _v3924_text_series(out, "ROUTE").str.upper()
    state = _v3924_text_series(out, "상태").str.upper()
    active = route.isin(["ATTACK", "ARMED"]) | state.isin(["ATTACK", "ARMED", "매수검토", "진입대기"])

    score = _v3924_num_series(out, ["ELITE_SCORE", "DISPLAY_SCORE", "FINAL_SCORE"], 0.0)
    final_score = _v3924_num_series(out, ["FINAL_SCORE", "DISPLAY_SCORE", "ELITE_SCORE"], 0.0)
    rr = _v3924_num_series(out, ["RR_NOW_TP1", "RR_MULT"], 0.0)
    gap = _v3924_num_series(out, ["ENTRY_GAP_PCT", "GAP_PCT", "gap_pct"], 99.0).abs()
    vwap = _v3924_num_series(out, ["VWAP_GAP", "VWAP_GAP_PCT"], 0.0)
    poc = _v3924_num_series(out, ["POC_GAP", "POC_GAP_PCT"], 0.0)
    no_chase = _v3924_flag_series(out, "NO_CHASE_FLAG")
    pullback_wait = _v3924_flag_series(out, "PULLBACK_WAIT_FLAG")
    ebs_pass = _v3924_ebs_pass_mask(out)

    strict = top_pick & eligible
    entry_clean = buy_pass & (gap <= 3.0) & (vwap <= 10.0) & (poc <= 30.0) & (rr >= 1.2) & (~no_chase) & (~pullback_wait)
    chase_risk = active & ((gap > 8.0) | (rr < 1.1) | no_chase | (vwap > 35.0) | (poc > 80.0))
    high_score = score >= 80.0
    holding_manage = route.eq("CARRY") | state.str.contains("보유", na=False)

    stage = pd.Series("BELOW_OFFICIAL_BAR", index=out.index, dtype="object")
    stage.loc[holding_manage] = "HOLDING_MANAGE"
    stage.loc[active & chase_risk] = "ROUTE_ACTIVE_BUT_CHASE_RISK"
    stage.loc[high_score & (~strict) & (~chase_risk)] = "HIGH_SCORE_BUT_ENTRY_BLOCKED"
    stage.loc[entry_clean & (~top_pick)] = "ENTRY_READY_BUT_NOT_TOP_PICK"
    stage.loc[top_pick & (~eligible)] = "TOP_PICK_ENTRY_BLOCKED"
    stage.loc[strict] = "OFFICIAL_BUY"

    triage = pd.Series("IGNORE", index=out.index, dtype="object")
    triage.loc[holding_manage] = "HOLDING_MANAGE"
    triage.loc[chase_risk] = "CHASE_RISK"
    triage.loc[high_score & (~strict) & (~chase_risk)] = "HIGH_SCORE_OBSERVE"
    triage.loc[entry_clean & (~strict)] = "ENTRY_CLEAN_OBSERVE"
    triage.loc[strict] = "OFFICIAL_BUY"

    reason1 = pd.Series("공식 기준 미달", index=out.index, dtype="object")
    reason2 = pd.Series("", index=out.index, dtype="object")
    reason1.loc[~top_pick] = "TOP_PICK=0"
    reason2.loc[(~top_pick) & entry_clean] = "진입조건은 양호하나 공식 Top Pick 아님"
    reason2.loc[(~top_pick) & high_score & (~entry_clean)] = "고점수이나 공식 Top Pick 아님"
    reason1.loc[top_pick & (~eligible)] = "BUY_NOW_ELIGIBLE=0"
    reason2.loc[top_pick & (~eligible) & (~buy_pass)] = "BUY_NOW_PASS=0"
    reason2.loc[chase_risk & (gap > 8.0)] = "추천가 괴리 과다"
    reason2.loc[chase_risk & (rr < 1.1)] = "RR_NOW_TP1 부족"
    reason2.loc[(vwap > 35.0) | (poc > 80.0)] = "VWAP/POC 과열"
    reason2.loc[~ebs_pass] = reason2.loc[~ebs_pass].mask(reason2.loc[~ebs_pass].eq(""), "EBS 미통과/불명")
    reason1.loc[strict] = "공식 신규매수"
    reason2.loc[strict] = "TOP_PICK + BUY_NOW_ELIGIBLE"

    # 0~100 근접도 점수: 공식식이 아니라 설명용 near-miss score.
    near = pd.Series(0.0, index=out.index, dtype="float64")
    near += np.where(active, 15, 0)
    near += np.where(top_pick, 20, 0)
    near += np.where(buy_pass, 20, 0)
    near += np.where(entry_clean, 15, 0)
    near += np.where(rr >= 1.2, 10, np.where(rr >= 1.0, 5, 0))
    near += np.where(final_score >= 75, 15, np.where(final_score >= 65, 8, 0))
    near += np.where(ebs_pass, 5, 0)
    near.loc[strict] = 100.0

    try:
        breadth_val = float(market_breadth)
    except (TypeError, ValueError) as e:
        logger.debug("v3.9.24 market breadth parse skip: %s", e)
        breadth_val = np.nan
    macro_risk_u = str(macro_risk or "").strip().upper()
    fx_level = _v3924_extract_fx_level(macro_msg)
    fx_high = fx_level is not None and fx_level >= 1500.0
    internal_weak = (not np.isnan(breadth_val)) and breadth_val < 35.0
    macro_hard = macro_risk_u in {"WARNING", "CRITICAL"}
    macro_relaxed_market_ok = fx_high and macro_hard and (not internal_weak)

    macro_mode = "NORMAL"
    if fx_high and internal_weak:
        macro_mode = "FX_HIGH_AND_INTERNAL_WEAK"
    elif fx_high:
        macro_mode = "FX_HIGH_REGIME"
    elif macro_hard:
        macro_mode = f"MACRO_{macro_risk_u}"

    out["STRICT_OFFICIAL_BUY_ELIGIBLE"] = strict.astype(int)
    out["OFFICIAL_FUNNEL_STAGE"] = stage
    out["OFFICIAL_BLOCK_REASON_1"] = reason1
    out["OFFICIAL_BLOCK_REASON_2"] = reason2
    out["OFFICIAL_NEAR_MISS_SCORE"] = near.clip(0, 100).round(1)
    out["OFFICIAL_NEAR_MISS_TYPE"] = triage
    out["CANDIDATE_TRIAGE_TYPE"] = triage

    out["MACRO_REGIME_MODE"] = macro_mode
    out["FX_HIGH_REGIME_FLAG"] = int(fx_high)
    out["FX_STALE_FLAG"] = 0  # 날짜 stale 판정은 UI v22.3.17 카드에서 run_meta 기준으로 처리
    out["MARKET_INTERNAL_WEAK_FLAG"] = int(internal_weak)
    out["MACRO_HARD_BLOCK_SHADOW"] = int(macro_hard)

    shadow_macro = active & buy_pass & (final_score >= 65) & (rr >= 1.0) & (gap <= 5.0) & bool(macro_relaxed_market_ok)
    shadow_entry = active & (final_score >= 75) & (rr >= 1.0) & (gap <= 10.0) & (~strict)
    shadow_score = active & buy_pass & (final_score >= 65) & (rr >= 1.0) & (~strict)
    out["MACRO_RELAXED_SHADOW_PASS"] = shadow_macro.astype(int)
    out["SHADOW_MACRO_RELAXED_ELIGIBLE"] = shadow_macro.astype(int)
    out["SHADOW_ENTRY_RELAXED_ELIGIBLE"] = shadow_entry.astype(int)
    out["SHADOW_SCORE_RELAXED_ELIGIBLE"] = shadow_score.astype(int)

    return out


# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# [v3.9.27] Abnormal History & Market Warning Guard
# ═══════════════════════════════════════════════════════════
# 목적:
#   - 아이로보틱스처럼 단기 진입 위치는 좋아 보이지만,
#     장기 이상이력/시장경보/초급등 후 붕괴 위험이 큰 종목을
#     공식매수뿐 아니라 관찰 후보에서도 제외한다.
#   - production guard다. Shadow가 아니며, BUY_NOW_PASS/ELIGIBLE을 실제 차단한다.
#   - 외부 실시간 조회가 없어도 CSV 내 시장경보 컬럼과 수익률 이력으로 작동한다.
ABNORMAL_HISTORY_RET120_SPIKE = 150.0
ABNORMAL_HISTORY_RET60_SPIKE = 100.0
ABNORMAL_HISTORY_RET20_SPIKE = 40.0
ABNORMAL_HISTORY_DROP_5D = -10.0
ABNORMAL_HISTORY_DROP_1D = -3.0
ABNORMAL_HISTORY_LONG_RATIO_BLOCK = 50.0
ABNORMAL_HISTORY_LONG_DD_BLOCK = -95.0
_ABNORMAL_HISTORY_HARD_WARNING_KEYWORDS = (
    "투자경고", "투자위험", "관리종목", "환기", "거래정지",
    "상장폐지", "상장적격성", "실질심사", "불성실공시",
)
_ABNORMAL_HISTORY_CAUTION_KEYWORDS = (
    "투자주의", "단기과열",
)
_ABNORMAL_HISTORY_WARNING_COLS = (
    "MARKET_WARNING", "MARKET_WARNING_TEXT", "WARNING_TYPE", "ISSUE_TYPE",
    "시장경보", "투자경고", "투자주의", "투자위험", "관리종목", "환기종목",
    "거래정지", "종목상태", "거래상태", "주의사항", "상장리스크",
)


def _ah_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _ah_text_join(df: pd.DataFrame, cols: tuple) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series("", index=df.index, dtype="object")
    out = pd.Series("", index=df.index, dtype="object")
    for c in existing:
        out = (out.astype(str) + " " + df[c].fillna("").astype(str)).str.strip()
    return out.fillna("").astype(str)


def add_abnormal_history_guard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """v3.9.27 장기 이상이력/시장경보 production guard.

    차단 조건:
      1) CSV 내 시장경보/관리/거래정지 계열 hard keyword 존재
      2) 장기 고점 대비 현재가 비율 컬럼이 있으면 50배 이상 또는 -95% 이상 훼손
      3) 최근 120/60/20일 초급등 후 5일 급락 또는 당일 급락 조합

    차단 시:
      TOP_PICK=0, BUY_NOW_ELIGIBLE=0, BUY_NOW_PASS=0, BUY_NOW_GRADE=AVOID,
      CANDIDATE_TRIAGE_TYPE=EXCLUDED_ABNORMAL_HISTORY 로 고정한다.
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    idx = out.index

    warning_text = _ah_text_join(out, _ABNORMAL_HISTORY_WARNING_COLS)
    hard_warning = warning_text.str.contains("|".join(_ABNORMAL_HISTORY_HARD_WARNING_KEYWORDS), regex=True, na=False)
    caution_warning = warning_text.str.contains("|".join(_ABNORMAL_HISTORY_CAUTION_KEYWORDS), regex=True, na=False)

    ratio_cols = [
        "LONG_HIGH_TO_CLOSE_RATIO", "MAX_PRICE_TO_CLOSE_RATIO", "ALL_TIME_HIGH_TO_CLOSE_RATIO",
        "LONG_MAX_TO_CLOSE_RATIO", "HISTORICAL_HIGH_TO_CLOSE_RATIO",
    ]
    long_ratio = pd.Series(0.0, index=idx, dtype="float64")
    for c in ratio_cols:
        if c in out.columns:
            long_ratio = pd.concat([long_ratio, _ah_num(out, c, 0.0)], axis=1).max(axis=1)

    dd_cols = ["LONG_DRAWDOWN_PCT", "ALL_TIME_DRAWDOWN_PCT", "MAX_DRAWDOWN_FROM_HIGH_PCT"]
    long_dd = pd.Series(0.0, index=idx, dtype="float64")
    for c in dd_cols:
        if c in out.columns:
            long_dd = pd.concat([long_dd, _ah_num(out, c, 0.0)], axis=1).min(axis=1)

    long_history_collapse = (long_ratio >= ABNORMAL_HISTORY_LONG_RATIO_BLOCK) | (long_dd <= ABNORMAL_HISTORY_LONG_DD_BLOCK)

    ret_120d = _ah_num(out, "ret_120d_%", 0.0)
    ret_60d = _ah_num(out, "ret_60d_%", 0.0)
    ret_20d = _ah_num(out, "ret_20d_%", 0.0)
    ret_5d = _ah_num(out, "ret_5d_%", 0.0)
    ret_1d = _ah_num(out, "ret_1d_%", 0.0)

    spike_reversal = (
        (((ret_120d >= ABNORMAL_HISTORY_RET120_SPIKE) | (ret_60d >= ABNORMAL_HISTORY_RET60_SPIKE)) & (ret_5d <= ABNORMAL_HISTORY_DROP_5D))
        | ((ret_20d >= ABNORMAL_HISTORY_RET20_SPIKE) & (ret_5d <= ABNORMAL_HISTORY_DROP_5D) & (ret_1d <= ABNORMAL_HISTORY_DROP_1D))
    )

    block = hard_warning | long_history_collapse | spike_reversal
    warn_only = (~block) & caution_warning

    reason = pd.Series("", index=idx, dtype="object")
    guard_type = pd.Series("", index=idx, dtype="object")
    reason.loc[hard_warning] = "시장경보/관리/거래정지 계열 리스크"
    guard_type.loc[hard_warning] = "MARKET_WARNING"
    reason.loc[long_history_collapse] = "장기 고점 대비 과도한 훼손/비정상 수정주가 이력"
    guard_type.loc[long_history_collapse] = "LONG_HISTORY_COLLAPSE"
    reason.loc[spike_reversal] = "초급등 후 단기 급락 — 눌림 착시/테마 붕괴 위험"
    guard_type.loc[spike_reversal] = "SPIKE_REVERSAL"
    reason.loc[warn_only] = "투자주의/단기과열 계열 주의 신호"
    guard_type.loc[warn_only] = "MARKET_CAUTION"

    out["ABNORMAL_HISTORY_GUARD_FLAG"] = block.astype(int)
    out["ABNORMAL_HISTORY_GUARD_LEVEL"] = np.where(block, "BLOCK", np.where(warn_only, "WARN", "CLEAR"))
    out["ABNORMAL_HISTORY_GUARD_TYPE"] = guard_type
    out["ABNORMAL_HISTORY_GUARD_REASON"] = reason
    out["MARKET_WARNING_GUARD_FLAG"] = hard_warning.astype(int)
    out["MARKET_CAUTION_GUARD_FLAG"] = caution_warning.astype(int)
    out["LONG_HISTORY_COLLAPSE_FLAG"] = long_history_collapse.astype(int)
    out["SPIKE_REVERSAL_GUARD_FLAG"] = spike_reversal.astype(int)
    out["LONG_HIGH_TO_CLOSE_RATIO_USED"] = long_ratio.round(2)

    if block.any():
        if "ORIGINAL_CANDIDATE_TRIAGE_TYPE" not in out.columns:
            out["ORIGINAL_CANDIDATE_TRIAGE_TYPE"] = out.get("CANDIDATE_TRIAGE_TYPE", pd.Series("", index=idx)).astype(str)
        if "ORIGINAL_BUY_NOW_PASS" not in out.columns:
            out["ORIGINAL_BUY_NOW_PASS"] = out.get("BUY_NOW_PASS", pd.Series(0, index=idx))
        if "ORIGINAL_BUY_NOW_GRADE" not in out.columns:
            out["ORIGINAL_BUY_NOW_GRADE"] = out.get("BUY_NOW_GRADE", pd.Series("", index=idx)).astype(str)

        out.loc[block, "TOP_PICK"] = 0
        if "TOP_PICK_TYPE" in out.columns:
            out.loc[block, "TOP_PICK_TYPE"] = ""
        out.loc[block, "BUY_NOW_ELIGIBLE"] = 0
        out.loc[block, "BUY_NOW_PASS"] = 0
        out.loc[block, "BUY_NOW_GRADE"] = "AVOID"
        if "BUY_NOW_SCORE" in out.columns:
            out.loc[block, "BUY_NOW_SCORE"] = 0
        out.loc[block, "CANDIDATE_TRIAGE_TYPE"] = "EXCLUDED_ABNORMAL_HISTORY"
        out.loc[block, "OFFICIAL_FUNNEL_STAGE"] = "EXCLUDED_ABNORMAL_HISTORY"
        out.loc[block, "OFFICIAL_BLOCK_REASON_1"] = "ABNORMAL_HISTORY_GUARD"
        out.loc[block, "OFFICIAL_BLOCK_REASON_2"] = reason.loc[block]
        if "NO_BUY_BREAKER_DECISION" in out.columns:
            out.loc[block, "NO_BUY_BREAKER_DECISION"] = "REJECT_ABNORMAL_HISTORY_GUARD"
        if "ENTRY_EDGE_LEVEL" in out.columns:
            out.loc[block, "ENTRY_EDGE_LEVEL"] = "BLOCK"
        if "ENTRY_EDGE_REASON" in out.columns:
            out.loc[block, "ENTRY_EDGE_REASON"] = reason.loc[block]

    return out

# [v3.9.26] Evidence-Gated No-Buy Breaker
# ═══════════════════════════════════════════════════════════
# 목적:
#   - 공식 TOP_PICK + BUY_NOW_ELIGIBLE 0개 고착을 완화하되,
#     검증 N=0 가설 룰을 production으로 승격하지 않는다.
#   - scripts/no_buy_breaker_backtest_v3926.py가 만든 검증 리포트에서
#     PASS 룰이 있을 때만 최대 1개를 공식 후보로 승격한다.
#   - 검증 리포트가 없거나 PASS 룰이 없으면 기존 공식매수 0개를 유지한다.
NO_BUY_BREAKER_MIN_N = 20
NO_BUY_BREAKER_MAX_PICKS = 1
NO_BUY_BREAKER_OUTPUT_COLS = [
    "NO_BUY_BREAKER_RULE_ID",
    "NO_BUY_BREAKER_VALIDATED",
    "NO_BUY_BREAKER_N",
    "NO_BUY_BREAKER_WIN_RATE_5D",
    "NO_BUY_BREAKER_AVG_RET_5D",
    "NO_BUY_BREAKER_ALPHA_5D",
    "NO_BUY_BREAKER_DECISION",
]


def _nbb_to_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """No-Buy Breaker용 안전 numeric Series."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _nbb_to_str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    """No-Buy Breaker용 안전 string Series."""
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def get_no_buy_breaker_rule_mask(df: pd.DataFrame, rule_id: str) -> pd.Series:
    """v3.9.26 검증/production 공통 후보 룰.

    이 함수는 룰 후보만 판정한다. production 승격 여부는 반드시
    검증 리포트의 PASS/REJECT 결과를 별도로 확인해야 한다.
    """
    if df is None or len(df) == 0:
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)

    route = _nbb_to_str(df, "ROUTE").str.upper().str.strip()
    risk = _nbb_to_str(df, "ENTRY_RISK_LEVEL", "GREEN").str.upper().str.strip()
    buy_pass = _nbb_to_num(df, "BUY_NOW_PASS", 0)
    pass_ebs = _nbb_to_num(df, "PASS_EBS", 0)
    volume = _nbb_to_num(df, "거래대금(억원)", 0)
    final = _nbb_to_num(df, "FINAL_SCORE", 0)
    struct = _nbb_to_num(df, "STRUCT_SCORE", 0)
    timing = _nbb_to_num(df, "TIMING_SCORE", 0)
    ai = _nbb_to_num(df, "AI_SCORE", 0)
    rr = _nbb_to_num(df, "RR_NOW_TP1", 0)
    gap = _nbb_to_num(df, "ENTRY_GAP_PCT", 99)
    if "ENTRY_GAP_PCT" not in df.columns and "GAP_PCT" in df.columns:
        gap = _nbb_to_num(df, "GAP_PCT", 99)
    vwap = _nbb_to_num(df, "VWAP_GAP", 0)
    poc = _nbb_to_num(df, "POC_GAP", 0)
    mfi = _nbb_to_num(df, "MFI14", 50)
    ret_1d = _nbb_to_num(df, "ret_1d_%", 0)
    ret_5d = _nbb_to_num(df, "ret_5d_%", 0)

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


def _load_validated_no_buy_breaker_rules(out_dir: str = None) -> list:
    """검증 리포트에서 production PASS 룰을 읽는다.

    scripts/no_buy_breaker_backtest_v3926.py가 생성하는
    no_buy_breaker_rules_latest.csv/json 중 하나라도 존재하면 사용한다.
    리포트가 없거나 PASS 룰이 없으면 빈 리스트를 반환한다.
    """
    import json
    from pathlib import Path

    base = Path(out_dir or OUT_DIR)
    csv_path = base / "no_buy_breaker_rules_latest.csv"
    json_path = base / "no_buy_breaker_backtest_latest.json"
    rules = []

    try:
        if csv_path.exists():
            rdf = pd.read_csv(csv_path)
            if len(rdf) > 0:
                decision = rdf.get("DECISION", pd.Series("", index=rdf.index)).astype(str).str.upper()
                passed = rdf[decision == "PASS_PRODUCTION_GATE"].copy()
                for _, row in passed.iterrows():
                    n = int(pd.to_numeric(row.get("N", 0), errors="coerce") or 0)
                    if n < NO_BUY_BREAKER_MIN_N:
                        continue
                    rules.append({
                        "rule_id": str(row.get("RULE_ID", "")),
                        "n": n,
                        "win_rate_5d": float(pd.to_numeric(row.get("WIN_RATE_5D", 0), errors="coerce") or 0.0),
                        "avg_ret_5d": float(pd.to_numeric(row.get("AVG_RET_5D", 0), errors="coerce") or 0.0),
                        "avg_alpha_5d": float(pd.to_numeric(row.get("AVG_ALPHA_5D", 0), errors="coerce") or 0.0),
                    })
        elif json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            for row in payload.get("rules", []):
                if str(row.get("DECISION", "")).upper() != "PASS_PRODUCTION_GATE":
                    continue
                n = int(row.get("N", 0) or 0)
                if n < NO_BUY_BREAKER_MIN_N:
                    continue
                rules.append({
                    "rule_id": str(row.get("RULE_ID", "")),
                    "n": n,
                    "win_rate_5d": float(row.get("WIN_RATE_5D", 0.0) or 0.0),
                    "avg_ret_5d": float(row.get("AVG_RET_5D", 0.0) or 0.0),
                    "avg_alpha_5d": float(row.get("AVG_ALPHA_5D", 0.0) or 0.0),
                })
    except Exception as e:
        logger.warning(f"⚠️ No-Buy Breaker 검증 리포트 로드 실패 (fallback 비활성): {e}")
        return []

    # 성과가 좋은 룰 우선. 평균수익률/승률/N 순으로 정렬.
    rules = [r for r in rules if r.get("rule_id")]
    rules.sort(key=lambda r: (r.get("avg_ret_5d", 0), r.get("win_rate_5d", 0), r.get("n", 0)), reverse=True)
    return rules


def apply_evidence_gated_no_buy_breaker(df: pd.DataFrame, rules: list = None, out_dir: str = None) -> pd.DataFrame:
    """v3.9.26 검증 통과형 No-Buy Breaker production gate.

    동작 원칙:
      1. 기존 공식 신규매수(TOP_PICK=1 AND BUY_NOW_ELIGIBLE=1)가 있으면 개입하지 않는다.
      2. 검증 리포트에서 PASS_PRODUCTION_GATE를 받은 룰이 없으면 개입하지 않는다.
      3. PASS 룰 후보가 현재 CSV에도 있으면 점수 우선순위로 최대 1개만 TOP_PICK/ELIGIBLE 승격한다.
      4. 모든 행에 NO_BUY_BREAKER_* 진단 컬럼을 남겨 UI/CSV에서 이유를 확인할 수 있게 한다.
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    for col in NO_BUY_BREAKER_OUTPUT_COLS:
        if col not in out.columns:
            if col in {"NO_BUY_BREAKER_RULE_ID", "NO_BUY_BREAKER_DECISION"}:
                out[col] = ""
            elif col in {"NO_BUY_BREAKER_WIN_RATE_5D", "NO_BUY_BREAKER_AVG_RET_5D", "NO_BUY_BREAKER_ALPHA_5D"}:
                out[col] = 0.0
            else:
                out[col] = 0

    top_pick = _nbb_to_num(out, "TOP_PICK", 0).astype(int)
    eligible = _nbb_to_num(out, "BUY_NOW_ELIGIBLE", 0).astype(int)
    official_existing = int(((top_pick == 1) & (eligible == 1)).sum())
    if official_existing > 0:
        out["NO_BUY_BREAKER_DECISION"] = "SKIP_EXISTING_OFFICIAL_BUY"
        return out

    if rules is None:
        rules = _load_validated_no_buy_breaker_rules(out_dir or OUT_DIR)

    rules = [r for r in rules if int(r.get("n", 0) or 0) >= NO_BUY_BREAKER_MIN_N and str(r.get("rule_id", "")).strip()]
    if not rules:
        out["NO_BUY_BREAKER_DECISION"] = "REJECT_NO_VALIDATED_RULE"
        return out

    candidates = []
    for rule in rules:
        rid = str(rule.get("rule_id", ""))
        mask = get_no_buy_breaker_rule_mask(out, rid)
        if not mask.any():
            continue
        tmp = out.loc[mask].copy()
        tmp["_NBB_RULE_ID"] = rid
        tmp["_NBB_N"] = int(rule.get("n", 0) or 0)
        tmp["_NBB_WIN_RATE_5D"] = float(rule.get("win_rate_5d", 0.0) or 0.0)
        tmp["_NBB_AVG_RET_5D"] = float(rule.get("avg_ret_5d", 0.0) or 0.0)
        tmp["_NBB_ALPHA_5D"] = float(rule.get("avg_alpha_5d", 0.0) or 0.0)
        candidates.append(tmp)

    if not candidates:
        out["NO_BUY_BREAKER_DECISION"] = "REJECT_NO_CURRENT_CANDIDATE"
        return out

    cand = pd.concat(candidates, axis=0)
    # 동일 종목이 여러 PASS 룰에 걸리면 검증 성과가 좋은 룰 우선.
    cand["_SORT_RULE_RET"] = pd.to_numeric(cand.get("_NBB_AVG_RET_5D", 0), errors="coerce").fillna(0)
    cand["_SORT_FINAL"] = _nbb_to_num(cand, "FINAL_SCORE", 0)
    cand["_SORT_ELITE"] = _nbb_to_num(cand, "ELITE_SCORE", 0)
    cand["_SORT_RR"] = _nbb_to_num(cand, "RR_NOW_TP1", 0)
    cand["_SORT_GAP"] = _nbb_to_num(cand, "ENTRY_GAP_PCT", 99)
    cand = cand.sort_values(
        ["_SORT_RULE_RET", "_NBB_WIN_RATE_5D", "_NBB_N", "_SORT_FINAL", "_SORT_ELITE", "_SORT_RR", "_SORT_GAP"],
        ascending=[False, False, False, False, False, False, True],
    )
    selected_idx = cand.index[:NO_BUY_BREAKER_MAX_PICKS]

    out.loc[selected_idx, "TOP_PICK"] = 1
    if "TOP_PICK_TYPE" not in out.columns:
        out["TOP_PICK_TYPE"] = ""
    out.loc[selected_idx, "TOP_PICK_TYPE"] = "NO_BUY_BREAKER_VALIDATED"
    out.loc[selected_idx, "BUY_NOW_ELIGIBLE"] = 1
    if "BUY_NOW_PASS" in out.columns:
        out.loc[selected_idx, "BUY_NOW_PASS"] = 1

    for idx in selected_idx:
        row = cand.loc[idx]
        out.loc[idx, "NO_BUY_BREAKER_RULE_ID"] = row.get("_NBB_RULE_ID", "")
        out.loc[idx, "NO_BUY_BREAKER_VALIDATED"] = 1
        out.loc[idx, "NO_BUY_BREAKER_N"] = int(row.get("_NBB_N", 0) or 0)
        out.loc[idx, "NO_BUY_BREAKER_WIN_RATE_5D"] = round(float(row.get("_NBB_WIN_RATE_5D", 0.0) or 0.0), 2)
        out.loc[idx, "NO_BUY_BREAKER_AVG_RET_5D"] = round(float(row.get("_NBB_AVG_RET_5D", 0.0) or 0.0), 2)
        out.loc[idx, "NO_BUY_BREAKER_ALPHA_5D"] = round(float(row.get("_NBB_ALPHA_5D", 0.0) or 0.0), 2)
        out.loc[idx, "NO_BUY_BREAKER_DECISION"] = "ALLOW_MAX_ONE_OFFICIAL_PICK"

    not_selected = ~out.index.isin(selected_idx)
    out.loc[not_selected & (out["NO_BUY_BREAKER_DECISION"].astype(str) == ""), "NO_BUY_BREAKER_DECISION"] = "NOT_SELECTED"
    return out


# [v3.9.28] July Profit Defense Gate — 실전 손실 방어 + Top ranking 재정렬 보조
# ═══════════════════════════════════════════════════════════
# 데이터 근거(2026-02~06 backtest_top3_trades_latest join):
#   - 최근 5영업일 상승률(ret_5d_%)이 낮은 구간(≤5)이 과열 구간보다 손실이 작음
#   - MARKET_BREADTH≥45 + ret_5d≤5 + Vol_Quality≥1.2 + STRUCT≥80 조합은
#     표본은 작지만 5~6월 손실 국면에서 상대적으로 양호하게 재현됨
#   - 반대로 ret_5d>20, VWAP/POC 과열, spike/abnormal guard, ENTRY_EDGE BLOCK은
#     신규 진입에서 손실 리스크가 급격히 커짐
# 이 레이어는 수익 보장이 아니라 "7월 생존/방어" 목적의 production 안전장치다.
JULY_PROFILE_BREADTH_MIN = 45.0
JULY_PROFILE_RET5_MAX = 5.0
JULY_PROFILE_VOLQ_MIN = 1.2
JULY_PROFILE_STRUCT_MIN = 80.0
JULY_PROFILE_FINAL_MIN = 75.0
JULY_PROFILE_VWAP_MAX = 20.0
JULY_PROFILE_POC_MAX = 60.0
JULY_BLOCK_BREADTH_MIN = 35.0
JULY_BLOCK_RET5_MAX = 20.0
JULY_BLOCK_VWAP_MAX = 35.0
JULY_BLOCK_POC_MAX = 80.0


def _july_num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _july_flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).eq(1)


def _july_str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


# ── 벡터화 사유 빌더 헬퍼 (v3.12.1) ─────────────────────────────
# 기존 `for i in out.index` f-string 루프를 대체. 출력 byte-identical 보장.
def _fmt1f(series: pd.Series, idx) -> pd.Series:
    """f'{x:.1f}' 벡터화 (C printf '%.1f' — Python format과 동일 반올림).

    NaN → 'nan' 문자열. 호출측에서 notna/임계 마스크로 제외되므로 안전.
    """
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return pd.Series(np.char.mod("%.1f", arr), index=idx, dtype="object")


def _join_reason_capped(idx, bit_series, cap: int, empty_fill: str) -> pd.Series:
    """조각 Series 목록을 ' · '로 결합. 행별 최대 cap개만 채택(원본 bits[:cap]).

    각 bit_series 원소는 미발동 시 '' 이어야 한다. 발동(비어있지 않음) 조각을
    앞에서부터 cap개까지만 채택하고, 하나도 없으면 empty_fill로 채운다.
    완전 벡터화 — 행 루프 없음.
    """
    count = pd.Series(0, index=idx, dtype="int64")
    result = pd.Series("", index=idx, dtype="object")
    for b in bit_series:
        include = (b.fillna("") != "") & (count < cap)
        result = result + pd.Series("", index=idx, dtype="object").mask(include, " · " + b)
        count = count + include.astype("int64")
    result = result.mask(count == 0, empty_fill)
    return result.str.replace("^ · ", "", regex=True)


def add_july_profit_defense_columns(df: pd.DataFrame, enforce: bool = True) -> pd.DataFrame:
    """v3.9.28 July Profit Defense Gate.

    핵심 역할:
      1) 모든 행에 JULY_PROFIT_* 진단 컬럼을 부여한다.
      2) 명백한 손실 위험 조합은 신규 진입 신호에서 제외한다.
      3) finalize_sort가 JULY_PROFIT_DEFENSE_SCORE를 보조 정렬축으로 사용하게 한다.

    엄격히 과최적화하지 않기 위해 '승격'은 하지 않고, 위험 후보 차단/강등만 한다.
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    breadth = _july_num(out, "MARKET_BREADTH", np.nan)
    ret5 = _july_num(out, "ret_5d_%", np.nan)
    vwap = _july_num(out, "VWAP_GAP", np.nan)
    poc = _july_num(out, "POC_GAP", np.nan)
    volq = _july_num(out, "Vol_Quality", np.nan)
    struct = _july_num(out, "STRUCT_SCORE", np.nan)
    final = _july_num(out, "FINAL_SCORE", np.nan)
    rsi = _july_num(out, "RSI14", np.nan)
    sector_rs = _july_num(out, "SECTOR_RS", np.nan)

    entry_edge = _july_str(out, "ENTRY_EDGE_LEVEL").str.upper().str.strip()
    entry_risk = _july_str(out, "ENTRY_RISK_LEVEL").str.upper().str.strip()

    abnormal_guard = _july_flag(out, "ABNORMAL_HISTORY_GUARD_FLAG")
    spike_guard = _july_flag(out, "SPIKE_REVERSAL_GUARD_FLAG")
    market_warn_guard = _july_flag(out, "MARKET_WARNING_GUARD_FLAG")
    collapse_guard = _july_flag(out, "LONG_HISTORY_COLLAPSE_FLAG")

    # 최근 손실국면에서 살아남은 실전 profile. 둘 중 하나의 quality 축(STRUCT 또는 FINAL)을 요구한다.
    profile_pass = (
        (breadth.fillna(0) >= JULY_PROFILE_BREADTH_MIN)
        & (ret5.fillna(99) <= JULY_PROFILE_RET5_MAX)
        & (volq.fillna(0) >= JULY_PROFILE_VOLQ_MIN)
        & ((struct.fillna(0) >= JULY_PROFILE_STRUCT_MIN) | (final.fillna(0) >= JULY_PROFILE_FINAL_MIN))
        & (vwap.fillna(999) <= JULY_PROFILE_VWAP_MAX)
        & (poc.fillna(999) <= JULY_PROFILE_POC_MAX)
        & (~abnormal_guard) & (~spike_guard)
        & (~entry_edge.eq("BLOCK"))
    )

    hard_block = (
        abnormal_guard | spike_guard | entry_edge.eq("BLOCK")
        | (ret5.fillna(0) > JULY_BLOCK_RET5_MAX)
        | (vwap.fillna(0) > JULY_BLOCK_VWAP_MAX)
        | (poc.fillna(0) > JULY_BLOCK_POC_MAX)
        | ((breadth.fillna(50) < JULY_BLOCK_BREADTH_MIN) & (ret5.fillna(0) > JULY_PROFILE_RET5_MAX))
        | (entry_risk.eq("ORANGE") & (ret5.fillna(0) > 10.0))
    )

    score = pd.Series(50.0, index=out.index, dtype="float64")
    score += np.where(breadth >= JULY_PROFILE_BREADTH_MIN, 10.0, 0.0)
    score -= np.where(breadth < JULY_BLOCK_BREADTH_MIN, 15.0, 0.0)
    score += np.where(ret5 <= JULY_PROFILE_RET5_MAX, 12.0, 0.0)
    score -= np.where(ret5 > 10.0, 6.0, 0.0)
    score -= np.where(ret5 > JULY_BLOCK_RET5_MAX, 12.0, 0.0)
    score += np.where(volq >= JULY_PROFILE_VOLQ_MIN, 8.0, 0.0)
    score += np.where(struct >= JULY_PROFILE_STRUCT_MIN, 8.0, 0.0)
    score += np.where(final >= JULY_PROFILE_FINAL_MIN, 6.0, 0.0)
    score += np.where(vwap <= 15.0, 6.0, 0.0)
    score -= np.where(vwap > JULY_PROFILE_VWAP_MAX, 6.0, 0.0)
    score -= np.where(vwap > JULY_BLOCK_VWAP_MAX, 12.0, 0.0)
    score += np.where(poc <= 40.0, 6.0, 0.0)
    score -= np.where(poc > JULY_PROFILE_POC_MAX, 6.0, 0.0)
    score -= np.where(poc > JULY_BLOCK_POC_MAX, 12.0, 0.0)
    score += np.where(rsi >= 55.0, 4.0, 0.0)
    score -= np.where(rsi < 45.0, 4.0, 0.0)
    score += np.where(sector_rs >= 0.0, 4.0, 0.0)
    score -= np.where(market_warn_guard | collapse_guard, 6.0, 0.0)
    score -= np.where(abnormal_guard | spike_guard | entry_edge.eq("BLOCK"), 20.0, 0.0)
    score = pd.Series(np.clip(score, 0.0, 100.0), index=out.index).round(1)

    level = pd.Series("CAUTION", index=out.index, dtype="object")
    level.loc[profile_pass & (~hard_block)] = "PASS"
    level.loc[hard_block] = "BLOCK"

    # 사유 빌더 — 완전 벡터화(v3.12.1, 원본 bits[:4] 루프와 byte-identical)
    _idx = out.index
    _E = lambda: pd.Series("", index=_idx, dtype="object")
    _r5 = pd.to_numeric(ret5, errors="coerce"); _rf = _fmt1f(_r5, _idx)
    _bd = pd.to_numeric(breadth, errors="coerce"); _bf = _fmt1f(_bd, _idx)
    _vw = pd.to_numeric(vwap, errors="coerce"); _vf = _fmt1f(_vw, _idx)
    _pc = pd.to_numeric(poc, errors="coerce"); _pf = _fmt1f(_pc, _idx)
    _ee = entry_edge.astype("object"); _er = entry_risk.astype("object")

    _m_r_hot = (_r5 > JULY_BLOCK_RET5_MAX).fillna(False)
    _m_r_up = (_r5 > JULY_PROFILE_RET5_MAX).fillna(False) & ~_m_r_hot
    _m_v_hot = (_vw > JULY_BLOCK_VWAP_MAX).fillna(False)
    _m_v_up = (_vw > JULY_PROFILE_VWAP_MAX).fillna(False) & ~_m_v_hot
    _m_p_hot = (_pc > JULY_BLOCK_POC_MAX).fillna(False)
    _m_p_up = (_pc > JULY_PROFILE_POC_MAX).fillna(False) & ~_m_p_hot

    _bits = [
        _E().mask(profile_pass.astype(bool) & ~hard_block.astype(bool), "7월 방어 profile 통과"),
        _E().mask(_m_r_up, "5일 상승 " + _rf + f"% > {JULY_PROFILE_RET5_MAX:.0f}%")
            .mask(_m_r_hot, "5일 과열 " + _rf + "%"),
        _E().mask((_bd < JULY_BLOCK_BREADTH_MIN).fillna(False), "시장폭 약함 " + _bf),
        _E().mask(_m_v_up, "VWAP " + _vf + f" > {JULY_PROFILE_VWAP_MAX:.0f}")
            .mask(_m_v_hot, "VWAP 과열 " + _vf),
        _E().mask(_m_p_up, "POC " + _pf + f" > {JULY_PROFILE_POC_MAX:.0f}")
            .mask(_m_p_hot, "POC 과열 " + _pf),
        _E().mask(abnormal_guard.astype(bool), "비정상 이력 guard"),
        _E().mask(spike_guard.astype(bool), "급등반전 guard"),
        _E().mask(_ee.eq("BLOCK"), "ENTRY_EDGE BLOCK"),
        _E().mask(_er.eq("ORANGE") & (_r5 > 10.0).fillna(False), "ORANGE + 단기과열"),
    ]
    reasons = _join_reason_capped(_idx, _bits, 4, "중립/관찰")

    out["JULY_PROFIT_DEFENSE_SCORE"] = score
    out["JULY_PROFIT_PROFILE_PASS"] = profile_pass.astype(int)
    out["JULY_PROFIT_BLOCK_FLAG"] = hard_block.astype(int)
    out["JULY_PROFIT_DEFENSE_LEVEL"] = level
    out["JULY_PROFIT_DEFENSE_REASON"] = reasons

    if not enforce:
        return out

    top_pick = _july_flag(out, "TOP_PICK")
    buy_eligible = _july_flag(out, "BUY_NOW_ELIGIBLE")
    buy_pass = _july_flag(out, "BUY_NOW_PASS")
    is_now = _july_flag(out, "IS_NOW_ENTRY")
    route = _july_str(out, "ROUTE").str.upper().str.strip()
    new_entry_like = top_pick | buy_eligible | buy_pass | is_now | route.isin(["ATTACK", "ARMED"])
    block_new_entry = hard_block & new_entry_like

    if "NEW_ENTRY_BLOCKED" not in out.columns:
        out["NEW_ENTRY_BLOCKED"] = False
    out.loc[block_new_entry, "NEW_ENTRY_BLOCKED"] = True

    if "STOP_OVERRIDE_REASON" not in out.columns:
        out["STOP_OVERRIDE_REASON"] = ""
    else:
        out["STOP_OVERRIDE_REASON"] = out["STOP_OVERRIDE_REASON"].astype("object")
    for idx in out.index[block_new_entry]:
        prior = str(out.at[idx, "STOP_OVERRIDE_REASON"] or "").strip()
        add = "JULY_PROFIT_DEFENSE: " + str(out.at[idx, "JULY_PROFIT_DEFENSE_REASON"])
        out.at[idx, "STOP_OVERRIDE_REASON"] = f"{prior} | {add}" if prior else add

    # 신규 매수 계약 차단. 점수 좋은 종목을 '승격'하지는 않고 손실위험 후보만 제거한다.
    for col in ["BUY_NOW_ELIGIBLE", "BUY_NOW_PASS", "TOP_PICK", "IS_NOW_ENTRY"]:
        if col in out.columns:
            out.loc[block_new_entry, col] = 0
    if "BUY_NOW_GRADE" in out.columns:
        out.loc[block_new_entry, "BUY_NOW_GRADE"] = "AVOID"
    if "BUY_NOW_REASON" in out.columns:
        for idx in out.index[block_new_entry]:
            prior = str(out.at[idx, "BUY_NOW_REASON"] or "").strip()
            add = "7월 손실방어 차단"
            out.at[idx, "BUY_NOW_REASON"] = f"{prior} · {add}" if prior else add
    if "TOP_PICK_TYPE" in out.columns:
        out["TOP_PICK_TYPE"] = out["TOP_PICK_TYPE"].astype("object")
        out.loc[block_new_entry, "TOP_PICK_TYPE"] = ""

    # ATTACK/ARMED 신규진입만 WAIT로 내린다. CARRY/보유 판단은 훼손하지 않는다.
    active_route = block_new_entry & route.isin(["ATTACK", "ARMED"])
    if active_route.any():
        out.loc[active_route, "ROUTE"] = Route.WAIT
        if "상태" in out.columns:
            out.loc[active_route, "상태"] = Route.WAIT
        if "IS_ACTIVE" in out.columns:
            out.loc[active_route, "IS_ACTIVE"] = False
        if "IS_WATCH" in out.columns:
            out.loc[active_route, "IS_WATCH"] = True

    return out

# [v3.9.29] Profit Recovery Suite — 손실 조합 차단 + 회복 후보 점수화
# ═══════════════════════════════════════════════════════════
# 데이터 근거(2026-02~06 backtest_top3_trades_20260624 + recommend 스냅샷 join):
#   - ret_5d>=5 & VWAP_GAP>=20/POC_GAP>=60 조합은 표본 대부분이 손절권으로 악화.
#   - ret_5d<=0 & MARKET_BREADTH>=50 조합은 최근 표본에서 상대적으로 우수.
#   - FINAL>=80 & MARKET_BREADTH>=60 조합도 방어적 회복장에서 우수.
# 목적:
#   1) 공식 신규매수 승격은 하지 않는다.
#   2) 이미 생성된 신규진입 후보 중 명백한 손실 충돌 조합은 차단/감액한다.
#   3) 무매수일에도 사용자가 볼 수 있는 관찰 우선순위와 비중 multiplier를 제공한다.
RECOVERY_BREADTH_GOOD = 50.0
RECOVERY_BREADTH_STRONG = 60.0
RECOVERY_RET5_PULLBACK_MAX = 0.0
RECOVERY_RET5_OVERHEAT_MIN = 5.0
RECOVERY_RET5_STRONG_OVERHEAT_MIN = 10.0
RECOVERY_VWAP_FOMO_MIN = 20.0
RECOVERY_VWAP_STRICT_MIN = 15.0
RECOVERY_POC_FOMO_MIN = 60.0
RECOVERY_POC_STRICT_MIN = 40.0
RECOVERY_WEAK_PULLBACK_RET5 = -10.0


def _rec_num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _rec_flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).eq(1)


def _rec_str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def add_profit_recovery_suite_columns(df: pd.DataFrame, enforce: bool = True) -> pd.DataFrame:
    """v3.9.29 Profit Recovery Suite.

    추가 컬럼:
      PROFIT_RECOVERY_SCORE      : 0~100 회복장 대응 점수
      PROFIT_RECOVERY_TIER       : A / B / C / BLOCK
      PROFIT_RECOVERY_SETUP      : PULLBACK_BREADTH / QUALITY_BREADTH / WATCH / RISK_COLLISION
      PROFIT_RECOVERY_BLOCK_FLAG : 신규진입 차단 플래그
      PROFIT_RECOVERY_SIZE_MULT  : 실전 비중 multiplier(0.00~0.70)
      PROFIT_RECOVERY_ACTION     : UI/운영용 문구
      PROFIT_RECOVERY_REASON     : 사람이 읽는 사유

    이 함수는 좋은 후보를 공식 승격하지 않는다. 손실 조합 차단/비중 축소/정렬 보조만 수행한다.
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    breadth = _rec_num(out, "MARKET_BREADTH", np.nan)
    ret5 = _rec_num(out, "ret_5d_%", np.nan)
    ret1 = _rec_num(out, "ret_1d_%", np.nan)
    vwap = _rec_num(out, "VWAP_GAP", np.nan)
    poc = _rec_num(out, "POC_GAP", np.nan)
    volq = _rec_num(out, "Vol_Quality", np.nan)
    final = _rec_num(out, "FINAL_SCORE", np.nan)
    display = _rec_num(out, "DISPLAY_SCORE", np.nan)
    struct = _rec_num(out, "STRUCT_SCORE", np.nan)
    timing = _rec_num(out, "TIMING_SCORE", np.nan)
    elite = _rec_num(out, "ELITE_SCORE", np.nan)
    rr = _rec_num(out, "RR_NOW_TP1", np.nan)
    rsi = _rec_num(out, "RSI14", np.nan)
    sector_rs = _rec_num(out, "SECTOR_RS", np.nan)
    guard_penalty = _rec_num(out, "GUARD_PENALTY_TOTAL", 0.0).fillna(0.0)

    july_level = _rec_str(out, "JULY_PROFIT_DEFENSE_LEVEL").str.upper().str.strip()
    entry_edge = _rec_str(out, "ENTRY_EDGE_LEVEL").str.upper().str.strip()
    entry_risk = _rec_str(out, "ENTRY_RISK_LEVEL").str.upper().str.strip()
    route = _rec_str(out, "ROUTE").str.upper().str.strip()

    abnormal_guard = _rec_flag(out, "ABNORMAL_HISTORY_GUARD_FLAG")
    spike_guard = _rec_flag(out, "SPIKE_REVERSAL_GUARD_FLAG")
    market_warning = _rec_flag(out, "MARKET_WARNING_GUARD_FLAG")
    long_collapse = _rec_flag(out, "LONG_HISTORY_COLLAPSE_FLAG")
    july_block = _rec_flag(out, "JULY_PROFIT_BLOCK_FLAG")
    data_integrity_bad = ~_rec_flag(out, "DATA_INTEGRITY_OK") if "DATA_INTEGRITY_OK" in out.columns else pd.Series(False, index=out.index)

    # 손실을 크게 만든 조합: 강한 단기상승 + VWAP/POC 괴리 = 추격 손절 확률 높음.
    fomo_collision = (
        ((ret5.fillna(0) >= RECOVERY_RET5_OVERHEAT_MIN) & (vwap.fillna(0) >= RECOVERY_VWAP_FOMO_MIN))
        | ((ret5.fillna(0) >= RECOVERY_RET5_OVERHEAT_MIN) & (poc.fillna(0) >= RECOVERY_POC_FOMO_MIN))
        | ((ret5.fillna(0) >= RECOVERY_RET5_STRONG_OVERHEAT_MIN) & (vwap.fillna(0) >= RECOVERY_VWAP_STRICT_MIN))
        | ((ret5.fillna(0) >= RECOVERY_RET5_STRONG_OVERHEAT_MIN) & (poc.fillna(0) >= RECOVERY_POC_STRICT_MIN))
    )

    # 약한 시장에서 낙폭만 큰 종목은 회복장이 아니라 추가 하락/손절 가능성이 높다.
    weak_knife = (ret5.fillna(0) <= RECOVERY_WEAK_PULLBACK_RET5) & (breadth.fillna(50) < 45.0)
    weak_sector_low_rsi = (sector_rs.fillna(0) <= 0.0) & (rsi.fillna(50) <= 55.0) & (ret5.fillna(0) <= -3.0)
    guard_collision = abnormal_guard | spike_guard | market_warning | long_collapse | entry_edge.eq("BLOCK") | july_block | data_integrity_bad

    hard_block = fomo_collision | weak_knife | guard_collision

    recovery_quality_core = (
        ((final.fillna(0) >= 65.0) | (display.fillna(0) >= 65.0) | (struct.fillna(0) >= 80.0))
        & (rr.fillna(0) >= 0.8)
        & ((timing.fillna(0) >= 15.0) | (volq.fillna(0) >= 1.4))
    )
    pullback_breadth = (
        (ret5.fillna(99) <= RECOVERY_RET5_PULLBACK_MAX)
        & (ret5.fillna(-99) >= -15.0)
        & (breadth.fillna(0) >= RECOVERY_BREADTH_GOOD)
        & (volq.fillna(0) >= 1.0)
        & (vwap.fillna(999) <= 20.0)
        & (poc.fillna(999) <= 60.0)
        & recovery_quality_core
        & (~guard_collision)
    )
    quality_breadth = (
        ((final.fillna(0) >= 80.0) | (display.fillna(0) >= 80.0) | (struct.fillna(0) >= 90.0))
        & (breadth.fillna(0) >= RECOVERY_BREADTH_STRONG)
        & (ret5.fillna(99) <= 5.0)
        & (vwap.fillna(999) <= 20.0)
        & (volq.fillna(0) >= 1.0)
        & (rr.fillna(0) >= 0.8)
        & (~guard_collision)
    )
    realistic_rr = (
        (rr.fillna(99) <= 1.3)
        & (rr.fillna(0) >= 0.6)
        & (breadth.fillna(0) >= RECOVERY_BREADTH_GOOD)
        & (poc.fillna(999) <= 40.0)
        & (~guard_collision)
    )

    score = pd.Series(45.0, index=out.index, dtype="float64")
    score += np.where(breadth >= RECOVERY_BREADTH_GOOD, 10.0, 0.0)
    score += np.where(breadth >= RECOVERY_BREADTH_STRONG, 6.0, 0.0)
    score -= np.where(breadth < 35.0, 12.0, 0.0)
    score += np.where(ret5 <= 0.0, 12.0, 0.0)
    score += np.where((ret5 > 0.0) & (ret5 <= 5.0), 4.0, 0.0)
    score -= np.where(ret5 >= 5.0, 8.0, 0.0)
    score -= np.where(ret5 >= 10.0, 8.0, 0.0)
    score += np.where(volq >= 1.2, 6.0, 0.0)
    score += np.where(final >= 75.0, 8.0, 0.0)
    score += np.where(display >= 75.0, 4.0, 0.0)
    score += np.where(struct >= 90.0, 8.0, 0.0)
    score += np.where(timing >= 40.0, 4.0, 0.0)
    score += np.where(elite >= 50.0, 4.0, 0.0)
    score += np.where((rr >= 0.8) & (rr <= 2.0), 5.0, 0.0)
    score -= np.where(rr > 5.0, 4.0, 0.0)
    score += np.where(vwap <= 15.0, 5.0, 0.0)
    score -= np.where(vwap >= RECOVERY_VWAP_FOMO_MIN, 8.0, 0.0)
    score += np.where(poc <= 40.0, 5.0, 0.0)
    score -= np.where(poc >= RECOVERY_POC_FOMO_MIN, 8.0, 0.0)
    score += np.where(sector_rs >= 0.0, 4.0, 0.0)
    score += np.where(rsi >= 50.0, 3.0, 0.0)
    score -= np.where(weak_sector_low_rsi, 7.0, 0.0)
    score -= np.where(guard_penalty > 0.0, np.minimum(guard_penalty, 20.0) * 0.4, 0.0)
    score += np.where(pullback_breadth, 10.0, 0.0)
    score += np.where(quality_breadth, 8.0, 0.0)
    score += np.where(realistic_rr, 4.0, 0.0)
    score -= np.where(fomo_collision, 28.0, 0.0)
    score -= np.where(weak_knife, 18.0, 0.0)
    score -= np.where(guard_collision, 30.0, 0.0)
    score = pd.Series(np.clip(score, 0.0, 100.0), index=out.index).round(1)

    setup = pd.Series("WATCH", index=out.index, dtype="object")
    setup.loc[pullback_breadth] = "PULLBACK_BREADTH"
    setup.loc[quality_breadth & ~pullback_breadth] = "QUALITY_BREADTH"
    setup.loc[realistic_rr & ~(pullback_breadth | quality_breadth)] = "REALISTIC_RR"
    setup.loc[fomo_collision] = "FOMO_COLLISION"
    setup.loc[weak_knife] = "WEAK_KNIFE"
    setup.loc[guard_collision] = "GUARD_BLOCK"

    tier = pd.Series("C", index=out.index, dtype="object")
    tier.loc[(score >= 65.0) & (pullback_breadth | quality_breadth | realistic_rr)] = "B"
    tier.loc[(score >= 82.0) & (pullback_breadth | quality_breadth)] = "A"
    tier.loc[hard_block] = "BLOCK"

    size_mult = pd.Series(0.20, index=out.index, dtype="float64")
    size_mult.loc[tier.eq("B")] = 0.35
    size_mult.loc[tier.eq("A")] = 0.50
    size_mult.loc[pullback_breadth | quality_breadth] = np.maximum(size_mult.loc[pullback_breadth | quality_breadth], 0.50)
    size_mult.loc[(tier.eq("A")) & (july_level.eq("PASS"))] = 0.70
    size_mult.loc[hard_block] = 0.0
    size_mult = size_mult.round(2)

    action = pd.Series("관찰", index=out.index, dtype="object")
    action.loc[tier.eq("A") & ~hard_block] = "소액 후보"
    action.loc[(tier.eq("A")) & (july_level.eq("PASS")) & ~hard_block] = "최우선 관찰"
    action.loc[tier.eq("B") & ~hard_block] = "관찰 후보"
    action.loc[hard_block] = "신규진입 차단"

    # 사유 빌더 — 완전 벡터화(v3.12.1, 원본 bits[:5] 루프와 byte-identical)
    _idx2 = out.index
    _E2 = lambda: pd.Series("", index=_idx2, dtype="object")
    _bd2 = pd.to_numeric(breadth, errors="coerce")
    _r52 = pd.to_numeric(ret5, errors="coerce")
    _bits2 = [
        _E2().mask(pullback_breadth.astype(bool), "회복형 눌림+시장폭"),
        _E2().mask(quality_breadth.astype(bool), "품질+시장폭"),
        _E2().mask(realistic_rr.astype(bool), "현실형 RR"),
        _E2().mask(fomo_collision.astype(bool), "단기상승+VWAP/POC 추격충돌"),
        _E2().mask(weak_knife.astype(bool), "약한 시장 낙폭주"),
        _E2().mask(weak_sector_low_rsi.astype(bool), "섹터 약세+RSI 약함"),
        _E2().mask(guard_collision.astype(bool), "기존 guard/JULY 차단"),
        _E2().mask(_bd2.notna(), "시장폭 " + _fmt1f(_bd2, _idx2)),
        _E2().mask(_r52.notna(), "5일 " + _fmt1f(_r52, _idx2) + "%"),
    ]
    reasons = _join_reason_capped(_idx2, _bits2, 5, "중립")

    out["PROFIT_RECOVERY_SCORE"] = score
    out["PROFIT_RECOVERY_TIER"] = tier
    out["PROFIT_RECOVERY_SETUP"] = setup
    out["PROFIT_RECOVERY_BLOCK_FLAG"] = hard_block.astype(int)
    out["PROFIT_RECOVERY_SIZE_MULT"] = size_mult
    out["PROFIT_RECOVERY_ACTION"] = action
    out["PROFIT_RECOVERY_REASON"] = reasons

    if not enforce:
        return out

    top_pick = _rec_flag(out, "TOP_PICK")
    buy_eligible = _rec_flag(out, "BUY_NOW_ELIGIBLE")
    buy_pass = _rec_flag(out, "BUY_NOW_PASS")
    is_now = _rec_flag(out, "IS_NOW_ENTRY")
    new_entry_like = top_pick | buy_eligible | buy_pass | is_now | route.isin(["ATTACK", "ARMED"])
    block_new_entry = hard_block & new_entry_like

    if "NEW_ENTRY_BLOCKED" not in out.columns:
        out["NEW_ENTRY_BLOCKED"] = False
    out.loc[block_new_entry, "NEW_ENTRY_BLOCKED"] = True

    if "STOP_OVERRIDE_REASON" not in out.columns:
        out["STOP_OVERRIDE_REASON"] = ""
    else:
        out["STOP_OVERRIDE_REASON"] = out["STOP_OVERRIDE_REASON"].astype("object")
    for idx in out.index[block_new_entry]:
        prior = str(out.at[idx, "STOP_OVERRIDE_REASON"] or "").strip()
        add = "PROFIT_RECOVERY: " + str(out.at[idx, "PROFIT_RECOVERY_REASON"])
        out.at[idx, "STOP_OVERRIDE_REASON"] = f"{prior} | {add}" if prior else add

    for col in ["BUY_NOW_ELIGIBLE", "BUY_NOW_PASS", "TOP_PICK", "IS_NOW_ENTRY"]:
        if col in out.columns:
            out.loc[block_new_entry, col] = 0
    if "BUY_NOW_GRADE" in out.columns:
        out.loc[block_new_entry, "BUY_NOW_GRADE"] = "AVOID"
    if "BUY_NOW_REASON" in out.columns:
        for idx in out.index[block_new_entry]:
            prior = str(out.at[idx, "BUY_NOW_REASON"] or "").strip()
            add = "수익회복 패치 차단"
            out.at[idx, "BUY_NOW_REASON"] = f"{prior} · {add}" if prior else add
    if "TOP_PICK_TYPE" in out.columns:
        out["TOP_PICK_TYPE"] = out["TOP_PICK_TYPE"].astype("object")
        out.loc[block_new_entry, "TOP_PICK_TYPE"] = ""

    active_route = block_new_entry & route.isin(["ATTACK", "ARMED"])
    if active_route.any():
        out.loc[active_route, "ROUTE"] = Route.WAIT
        if "상태" in out.columns:
            out.loc[active_route, "상태"] = Route.WAIT
        if "IS_ACTIVE" in out.columns:
            out.loc[active_route, "IS_ACTIVE"] = False
        if "IS_WATCH" in out.columns:
            out.loc[active_route, "IS_WATCH"] = True

    # 켈리/추천금액은 추천 계약을 바꾸지 않는 범위에서 안전 multiplier만 반영한다.
    # 0.70 이하로 cap하여 7월 회복 국면에서도 풀베팅을 방지한다.
    for amount_col in ["추천금액(만원)", "켈리_수량", "추천수량"]:
        if amount_col in out.columns:
            vals = pd.to_numeric(out[amount_col], errors="coerce")
            has_val = vals.notna() & (vals > 0)
            out.loc[has_val, amount_col] = (vals.loc[has_val] * size_mult.loc[has_val]).round(0)

    return out


def _compute_is_now_entry_vectorized(df: pd.DataFrame) -> pd.Series:
    """IS_NOW_ENTRY — shared_utils.compute_is_now_entry 벡터 적용.
    
    ATR_Pct(decimal, ml_engine) 우선, 없으면 ATR_PCT(percentage, stop_logic) 허용.
    """
    try:
        from shared_utils import compute_is_now_entry as _cine
    except ImportError:
        # v22 신규 함수 미탑재 환경 — fallback: ROUTE==ATTACK
        route = df.get("ROUTE", pd.Series("", index=df.index))
        return (route.isin(["ATTACK"])).astype(int)
    
    close = pd.to_numeric(df.get("종가", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    entry = pd.to_numeric(df.get("추천매수가", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    # ATR_Pct(decimal) 우선, ATR_PCT(percentage)도 허용 — 내부 정규화
    atr = df.get("ATR_Pct", df.get("ATR_PCT", pd.Series(0.02, index=df.index)))
    mcap = pd.to_numeric(df.get("시가총액(억원)", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    
    return pd.Series(
        [_cine(c, e, a, m) for c, e, a, m in zip(close, entry, atr, mcap)],
        index=df.index,
        dtype=int,
    )


def finalize_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Loss-defense SORT_SPEC — production decision first.
    
    정렬 우선순위 (내림차순 기준, 낮은 ROUTE_PRIORITY가 먼저):
      1. PRODUCTION_BUY (엄격 품질게이트 통과 종목)
      2. QUALITY_GUARD_SCORE (현금/관찰일 때 가장 가까운 후보)
      3. TOP_PICK / IS_NOW_ENTRY / ROUTE_PRIORITY
      4. 기존 방어·회복·ELITE·RR·BALANCE 축
    
    다른 단계에서 IS_NOW_ENTRY를 ROUTE==ATTACK으로 세팅했어도 
    여기서 adaptive로 덮어쓴다. 순서가 SSOT.
    """
    df = df.copy()

    # IS_NOW_ENTRY adaptive 재계산 (항상 덮어쓰기 — 이전 단계의 단순 ROUTE==ATTACK 치환)
    df["IS_NOW_ENTRY"] = _compute_is_now_entry_vectorized(df)

    # ROUTE_PRIORITY (정렬용 임시 컬럼)
    route = df.get("ROUTE", pd.Series("", index=df.index)).astype(str)
    df["_ROUTE_PRIORITY"] = route.map(_SORT_ROUTE_PRIORITY).fillna(99).astype(int)

    # 정렬 축 모두 존재 확인 (없으면 중립 값으로 채움)
    for col, default in [
        ("PRODUCTION_BUY", 0), ("QUALITY_GUARD_SCORE", 0),
        ("TOP_PICK", 0), ("IS_NOW_ENTRY", 0),
        ("JULY_PROFIT_DEFENSE_SCORE", 50),
        ("PROFIT_RECOVERY_SCORE", 50),
        ("ALPHA_SCORE", 0),  # [v29] 미검증/결측 → 0 (정렬 영향 없음)
        ("ELITE_SCORE", 0), ("RR_NOW_TP1", 0),
        ("BALANCE_SCORE", 0), ("ENTRY_GAP_PCT", 99),
        ("DISPLAY_SCORE", 0),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # SORT_SPEC 적용
    # [v29] ALPHA_SCORE를 ELITE보다 앞에 — OOS 검증 통과한 유일한 종목 점수
    # (알파 십분위 실측 승률 24%→41% 단조 vs ELITE는 역방향).
    df = df.sort_values(
        by=["PRODUCTION_BUY", "QUALITY_GUARD_SCORE", "TOP_PICK", "IS_NOW_ENTRY", "_ROUTE_PRIORITY", "JULY_PROFIT_DEFENSE_SCORE", "PROFIT_RECOVERY_SCORE", "ALPHA_SCORE", "ELITE_SCORE",
            "RR_NOW_TP1", "BALANCE_SCORE", "ENTRY_GAP_PCT", "DISPLAY_SCORE"],
        ascending=[False, False, False, False, True, False, False, False, False, False, False, True, False],
        kind="mergesort",   # 안정 정렬 (동점 시 원래 순서 유지)
    ).reset_index(drop=True)

    # 임시 컬럼 제거
    df = df.drop(columns=["_ROUTE_PRIORITY"], errors="ignore")
    return df


def finalize_outputs(ctx: PipelineContext) -> None:
    from collector import make_rank_validation_report  # 아직 collector에만 있음
    df_out = ctx.df_out; trade_ymd = ctx.trade_ymd
    _am = {Route.ATTACK:1,"ATTACK":1,Route.ARMED:2,"ARMED":2,Route.WAIT:3,"WAIT":3,
           Route.NEUTRAL:4,"NEUTRAL":4,Route.OVERHEAT:5,"OVERHEAT":5,
           Route.EXIT_WARNING:6,"EXIT_WARNING":6,Route.CARRY:7,"CARRY":7}
    must_cols = [
        "LDY_RANK","종목코드","종목명","시장","업종_대분류","종가","거래대금(억원)","시가총액(억원)",
        "켈리_수량","추천금액(만원)","상태","ROUTE","ACTION_PRIORITY","IS_ACTIVE","IS_NOW_ENTRY","IS_WATCH",
        "DISPLAY_SCORE","FINAL_SCORE","STRUCT_SCORE","TIMING_SCORE","AI_SCORE","NEWS_SCORE",
        "ELITE_SCORE","AXIS_MEAN","AXIS_GAP","BALANCE_SCORE","RR_NOW_TP1","ENTRY_GAP_PCT","ELITE_REASON","TOP_PICK",
        "추천매수가","손절가","추천매도가1","추천매도가2","TRIGGER","V_POWER","거래강도",
        "VWAP","POC_GAP","NEWS_REASON","TTM_SQUEEZE_CNT","Low_Trend_PCT","RSI14","이격도",
        "SCORE_REASON_TOP1","SCORE_REASON_TOP2","SCORE_RISK","ROUTE_REASON",
        "MACRO_RISK","MARKET_BREADTH"]
    for c in must_cols:
        if c not in df_out.columns: df_out[c] = np.nan
    for _cm in ["CARRY_FROM_DATE","CARRY_AGE_DAYS","IS_STALE_CARRY","STALE_PENALTY",
                "ROW_BUILD_MODE","DATA_FRESHNESS_OK"]:
        if _cm not in df_out.columns:
            if _cm == "CARRY_FROM_DATE": df_out[_cm] = np.nan
            elif _cm == "IS_STALE_CARRY": df_out[_cm] = False
            elif _cm == "ROW_BUILD_MODE": df_out[_cm] = "FRESH"
            elif _cm == "DATA_FRESHNESS_OK": df_out[_cm] = True
            else: df_out[_cm] = 0
    df_out = df_out[must_cols + [c for c in df_out.columns if c not in must_cols]]
    # ══════════════════════════════════════════════════
    #  CONFIG_SNAPSHOT 저장 (v3.7.27에서 JSON 분리 · v3.7.29에서 migration 완료)
    # ══════════════════════════════════════════════════
    # 정책 — single source of truth:
    #   · CSV 행 데이터:  경량 (CONFIG_VERSION 문자열만)
    #   · JSON 파일:     config 스냅샷 전용
    #                    data/config_snapshot_{trade_ymd}.json (일자별)
    #                    data/config_snapshot_latest.json      (최신 alias)
    # 읽기:
    #   · 모든 참조 코드는 load_config_snapshot(trade_ymd) 헬퍼를 사용한다.
    #   · 예: snapshot = load_config_snapshot("20260420")
    #   · Fallback: 파일이 없으면 빈 dict 반환 (예외 없음) — 참조 코드가 그냥 계속 돌 수 있도록.
    # 주변 참조 (v3.7.29 기준 전부 이관 완료):
    #   · test_shadow_analyze.py → SKIP_KEYS에 포함 (비교에서 제외)
    try:
        from collector_config import DEFAULT_CONFIG as _snap
        # CSV에는 버전 문자열만 (작은 값, 호환성 유지)
        df_out["CONFIG_VERSION"] = _snap.config_version
        # 전체 스냅샷은 별도 JSON 파일로 — 일자별 1회 덮어쓰기
        try:
            from pathlib import Path as _P
            _snap_path = _P(OUT_DIR) / f"config_snapshot_{trade_ymd}.json"
            _snap_latest = _P(OUT_DIR) / "config_snapshot_latest.json"
            _snap_json_str = _snap.snapshot_json()
            _snap_path.write_text(_snap_json_str, encoding="utf-8")
            _snap_latest.write_text(_snap_json_str, encoding="utf-8")
            logger.info(f"✅ CONFIG_SNAPSHOT → {_snap_path.name}")
        except Exception as _ef:
            logger.warning(f"⚠️ CONFIG_SNAPSHOT JSON 저장 실패: {_ef}")
    except (ImportError, AttributeError) as e:
        logger.debug(f"CONFIG_SNAPSHOT 스킵 (구성 없음): {e}")
    except Exception as e:
        logger.warning(f"⚠️ CONFIG_SNAPSHOT 오류: {e}")
    # [v20.6] macro_risk 직접 저장
    df_out["MACRO_RISK"] = ctx.macro_risk
    df_out["MARKET_BREADTH"] = ctx.breadth.get("ALL", np.nan)

    # [v28] 시장 레짐 판정 — UP(정상 진입)/NEUTRAL(사이즈 50%)/DOWN(진입 차단)
    # 실측 (98일): UP 레짐 픽 평균 +5.1%/건, NEUTRAL +0.1%, DOWN은 손실 우세.
    # 레짐은 GO/NO-GO·사이즈만 결정하고 랭킹에는 개입하지 않는다
    # (UP 레짐에서도 POC 근접 랭킹이 점수 랭킹보다 +2.1%p 우월했음).
    try:
        from market_regime import compute_market_regime, inject_regime_columns
        _regime_info = compute_market_regime(
            trade_ymd, ctx.breadth.get("ALL"), data_dir=OUT_DIR,
        )
        df_out = inject_regime_columns(df_out, _regime_info)
        log(f"🌡️ [v28] 레짐: {_regime_info['regime']} — {_regime_info['reason']}")
    except Exception as e:
        logger.warning(f"⚠️ 레짐 판정 실패 (NEUTRAL 취급): {e}")
        df_out["MARKET_REGIME"] = "UNKNOWN"
        df_out["REGIME_REASON"] = f"판정 실패: {e}"
        df_out["REGIME_SIZE_MULT"] = 0.5
        df_out["REGIME_ALLOW_ENTRY"] = 1
    # Run Health
    _health = None
    try:
        from run_health import check_run_health, save_health
        _health = check_run_health(df_out, mcap_map=ctx.mcap_map, bench_map=ctx.bench_map,
            inv_maps=ctx.inv_maps, trade_ymd=trade_ymd)
        _health.macro_risk = ctx.macro_risk
        _health.market_breadth = ctx.breadth.get("ALL", 50.0)
        df_out = _health.inject_columns(df_out); save_health(_health, OUT_DIR, trade_ymd)
        log(_health.summary())
    except ImportError: log("ℹ️ run_health 모듈 없음")
    except Exception as e: log(f"⚠️ Run Health 실패: {e}")
    # 축 비활성 중립화 + 행동 제한
    if _health:
        _r = set(_health.reasons)
        if "NEWS_OFF" in _r: df_out["NEWS_SCORE"]=np.nan; df_out["NEWS_REASON"]="DATA_UNAVAILABLE"
        if "SECTOR_FAIL" in _r:
            for c in ["SECTOR_RANK","SECTOR_RS"]:
                if c in df_out.columns: df_out[c]=np.nan
        if "BENCH_FAIL" in _r or "BENCH_NAN" in _r:
            for _bc in ["rel_20d_%","rel_60d_%","rel_120d_%","벤치_60d_KOSPI_%","벤치_60d_KOSDAQ_%"]:
                if _bc in df_out.columns: df_out[_bc]=np.nan
        _mx = _health.max_allowed_route; _dc = 0
        if _mx != "ATTACK":
            _atk = df_out["ROUTE"]==Route.ATTACK
            if _atk.any():
                _fb = Route.ARMED if _mx=="ARMED" else Route.WAIT
                df_out.loc[_atk,"ROUTE"]=_fb; df_out.loc[_atk,"상태"]=_fb; _dc+=_atk.sum()
        if _mx == "WAIT":
            _arm = df_out["ROUTE"]==Route.ARMED
            if _arm.any(): df_out.loc[_arm,"ROUTE"]=Route.WAIT; df_out.loc[_arm,"상태"]=Route.WAIT; _dc+=_arm.sum()
        if _dc > 0:
            _cr = f"RUN_STATUS={_health.status}" if _health.status!="OK" else f"confidence={_health.confidence_score:.0f}"
            log(f"🛡️ [v20.0.2] 행동 상한 제어: {_cr} → 최대허용 {_mx}, {_dc}건 route_capped")
            df_out["IS_ACTIVE"]=df_out["ROUTE"].isin([Route.ATTACK,Route.ARMED])
            # [v22] IS_NOW_ENTRY는 finalize_sort에서 adaptive로 재계산되므로 여기선 건드리지 않음
            df_out["IS_WATCH"]=df_out["ROUTE"]==Route.WAIT
            df_out["ACTION_PRIORITY"]=df_out["ROUTE"].map(_am).fillna(7).astype(int)

            # [v22] route cap 이후 TOP_PICK positive gate 재적용
            # ATTACK→WAIT/ARMED→WAIT capped 종목은 TOP_PICK에서 탈락시켜야
            # "TOP_PICK=1 but ROUTE=WAIT" 누출 재발 차단.
            _active_mask = df_out["ROUTE"].isin([Route.ATTACK, Route.ARMED, "ATTACK", "ARMED"])
            _leaked = (df_out.get("TOP_PICK", pd.Series(0, index=df_out.index)) == 1) & (~_active_mask)
            if _leaked.any():
                _leak_n = int(_leaked.sum())
                df_out.loc[_leaked, "TOP_PICK"] = 0
                if "TOP_PICK_TYPE" in df_out.columns:
                    df_out.loc[_leaked, "TOP_PICK_TYPE"] = ""
                log(f"🎯 [v22] route cap 후 TOP_PICK 정리: {_leak_n}건 탈락")
            # [v22] route cap 발생 시 누출 여부와 무관하게 재정렬
            # (ROUTE_PRIORITY 바뀌었으므로 순서 영향)
            # finalize_sort는 아래 CSV 저장 직전에 한 번 더 호출되지만,
            # 여기서 한 번 정리해두는 게 중간 단계 일관성에 좋음.
            try:
                df_out = finalize_sort(df_out)
            except Exception as e:
                logger.warning(f"⚠️ route cap 후 재정렬 실패 (무해): {e}")

    # ── [v21.1] ACTION_PRIORITY 항상 재계산 (SSOT 보장) ──
    df_out["ACTION_PRIORITY"] = df_out["ROUTE"].map(_am).fillna(7).astype(int)

    # ── [v22] 최종 정렬 SSOT 적용 (SORT_SPEC) ──
    # TOP_PICK × IS_NOW_ENTRY(adaptive) × ROUTE × ELITE × RR × BALANCE × ENTRY_GAP × DISPLAY_SCORE
    # 정렬 직후 LDY_RANK 재부여 (stale 방지)
    try:
        df_out = finalize_sort(df_out)
        df_out["LDY_RANK"] = np.arange(1, len(df_out) + 1)
        log(f"🎯 [v22] SORT_SPEC 적용 완료 (TOP_PICK × IS_NOW_ENTRY × ELITE × RR ...)")
    except Exception as e:
        logger.warning(f"⚠️ finalize_sort 실패 (무해 — 기존 순서 유지): {e}")

    # [v3.9.6] PRE_ENTRY_RISK 컬럼 부여 — 표시 전용, 추천 제외 아님
    try:
        df_out = add_entry_risk_columns(df_out)
        n_red = int((df_out["ENTRY_RISK_LEVEL"] == "RED").sum())
        n_orange = int((df_out["ENTRY_RISK_LEVEL"] == "ORANGE").sum())
        log(f"🚨 [v3.9.6] ENTRY_RISK 컬럼 부여 완료 — RED {n_red} · ORANGE {n_orange}")
    except Exception as e:
        logger.warning(f"⚠️ add_entry_risk_columns 실패 (무해): {e}")

    # [v22.3.10] ENTRY_EDGE shadow 컬럼 부여 — 표시/감점 전용, 공식 매수식 무수정
    try:
        _eligible_before = df_out.get("BUY_NOW_ELIGIBLE", pd.Series([], dtype=object)).copy()
        df_out = add_entry_edge_columns(df_out)
        if "BUY_NOW_ELIGIBLE" in df_out.columns and len(_eligible_before) == len(df_out):
            if not df_out["BUY_NOW_ELIGIBLE"].equals(_eligible_before):
                logger.error("ENTRY_EDGE 적용 중 BUY_NOW_ELIGIBLE 변경 감지 — 원복")
                df_out["BUY_NOW_ELIGIBLE"] = _eligible_before
        n_edge = int((df_out["ENTRY_EDGE_SHADOW_FLAG"] == 1).sum())
        log(f"🧪 [v22.3.10] ENTRY_EDGE shadow 컬럼 부여 완료 — B_red 감점 {n_edge}건")
    except Exception as e:
        logger.warning(f"⚠️ add_entry_edge_columns 실패 (무해): {e}")


    # [v3.9.24] Official Buy Funnel & Macro Regime Shadow — 표시/진단 전용, 공식식 무수정
    try:
        _contract_cols = [c for c in ["TOP_PICK", "BUY_NOW_ELIGIBLE", "BUY_NOW_PASS", "BUY_NOW_GRADE"] if c in df_out.columns]
        _contract_before = df_out[_contract_cols].copy() if _contract_cols else pd.DataFrame(index=df_out.index)
        df_out = add_official_buy_funnel_columns(
            df_out,
            macro_risk=ctx.macro_risk,
            market_breadth=ctx.breadth.get("ALL", np.nan),
            macro_msg=getattr(ctx, "macro_msg", ""),
        )
        for _c in _contract_cols:
            if not df_out[_c].equals(_contract_before[_c]):
                logger.error("v3.9.24 funnel 적용 중 %s 변경 감지 — 원복", _c)
                df_out[_c] = _contract_before[_c]
        _triage_counts = df_out["CANDIDATE_TRIAGE_TYPE"].value_counts().to_dict()
        log(f"🧭 [v3.9.24] Official Buy Funnel 컬럼 부여 완료 — {_triage_counts}")
    except Exception as e:
        logger.warning(f"⚠️ add_official_buy_funnel_columns 실패 (무해): {e}")


    # [v3.9.27] Abnormal History & Market Warning Guard — production hard block
    try:
        df_out = add_abnormal_history_guard_columns(df_out)
        _ah_block = int(pd.to_numeric(df_out.get("ABNORMAL_HISTORY_GUARD_FLAG", 0), errors="coerce").fillna(0).astype(int).sum())
        _ah_warn = int((df_out.get("ABNORMAL_HISTORY_GUARD_LEVEL", pd.Series("", index=df_out.index)).astype(str) == "WARN").sum())
        if _ah_block > 0 or _ah_warn > 0:
            _ah_types = df_out.get("ABNORMAL_HISTORY_GUARD_TYPE", pd.Series("", index=df_out.index)).astype(str).value_counts().to_dict()
            log(f"🧯 [v3.9.27] Abnormal History Guard 적용 — BLOCK {_ah_block} · WARN {_ah_warn} · {_ah_types}")
        else:
            log("🧯 [v3.9.27] Abnormal History Guard 적용 — BLOCK 0")
    except Exception as e:
        logger.warning(f"⚠️ add_abnormal_history_guard_columns 실패 (기존 추천 유지): {e}")

    # [loss-defense] No-Buy Breaker는 기본 OFF. 현금 보유를 정상 결정으로 인정한다.
    # 과거 리포트가 PASS여도 운영자가 명시적으로 ENABLE_NO_BUY_BREAKER=1을
    # 설정하지 않으면 공식 후보를 억지로 만들지 않는다.
    try:
        _official_before = int(((pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int) == 1)
                                & (pd.to_numeric(df_out.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0).astype(int) == 1)).sum())
        _nbb_enabled = str(os.environ.get("ENABLE_NO_BUY_BREAKER", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if _nbb_enabled:
            df_out = apply_evidence_gated_no_buy_breaker(df_out, out_dir=OUT_DIR)
        else:
            for _c in NO_BUY_BREAKER_OUTPUT_COLS:
                if _c not in df_out.columns:
                    df_out[_c] = "" if _c in {"NO_BUY_BREAKER_RULE_ID", "NO_BUY_BREAKER_DECISION"} else 0
            df_out["NO_BUY_BREAKER_DECISION"] = "DISABLED_LOSS_DEFENSE"
        _official_after = int(((pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int) == 1)
                               & (pd.to_numeric(df_out.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0).astype(int) == 1)).sum())
        if _official_after > _official_before:
            log(f"🧬 [v3.9.26] Evidence-Gated No-Buy Breaker 공식 후보 승격: {_official_after - _official_before}건")
        else:
            _dec = df_out.get("NO_BUY_BREAKER_DECISION", pd.Series("", index=df_out.index)).astype(str).value_counts().to_dict()
            log(f"🧬 [v3.9.26] No-Buy Breaker 비활성/보류 — {_dec}")
    except Exception as e:
        logger.warning(f"⚠️ apply_evidence_gated_no_buy_breaker 실패 (기존 공식추천 유지): {e}")

    # [v3.9.28] July Profit Defense Gate — 5~6월 손실국면 기반 신규진입 방어/정렬
    try:
        _official_before_july = int(((pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int) == 1)
                                     & (pd.to_numeric(df_out.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0).astype(int) == 1)).sum())
        df_out = add_july_profit_defense_columns(df_out, enforce=True)
        _blocked_july = int(pd.to_numeric(df_out.get("JULY_PROFIT_BLOCK_FLAG", 0), errors="coerce").fillna(0).astype(int).sum())
        _profile_july = int(pd.to_numeric(df_out.get("JULY_PROFIT_PROFILE_PASS", 0), errors="coerce").fillna(0).astype(int).sum())
        _official_after_july = int(((pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int) == 1)
                                    & (pd.to_numeric(df_out.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0).astype(int) == 1)).sum())
        df_out = finalize_sort(df_out)
        df_out["LDY_RANK"] = np.arange(1, len(df_out) + 1)
        log(f"🛡️ [v3.9.28] July Profit Defense 적용 — profile_pass {_profile_july} · block {_blocked_july} · official {_official_before_july}->{_official_after_july}")
    except Exception as e:
        logger.warning(f"⚠️ July Profit Defense 실패 (기존 추천 유지): {e}")

    # [v3.9.29] Profit Recovery Suite — 손실 조합 차단 + 회복 후보 정렬/비중 보정
    try:
        _official_before_rec = int(((pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int) == 1)
                                    & (pd.to_numeric(df_out.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0).astype(int) == 1)).sum())
        df_out = add_profit_recovery_suite_columns(df_out, enforce=True)
        _blocked_rec = int(pd.to_numeric(df_out.get("PROFIT_RECOVERY_BLOCK_FLAG", 0), errors="coerce").fillna(0).astype(int).sum())
        _a_rec = int((df_out.get("PROFIT_RECOVERY_TIER", pd.Series("", index=df_out.index)).astype(str) == "A").sum())
        _official_after_rec = int(((pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int) == 1)
                                   & (pd.to_numeric(df_out.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0).astype(int) == 1)).sum())
        df_out = finalize_sort(df_out)
        df_out["LDY_RANK"] = np.arange(1, len(df_out) + 1)
        log(f"💹 [v3.9.29] Profit Recovery Suite 적용 — A {_a_rec} · block {_blocked_rec} · official {_official_before_rec}->{_official_after_rec}")
    except Exception as e:
        logger.warning(f"⚠️ Profit Recovery Suite 실패 (기존 추천 유지): {e}")

    # [v24.4] Profit Momentum Overlay (PMS) — 데이터 검증 SHADOW (공식 추천 불변)
    # 근거: per_trade_log 16,093건 walk-forward — 모멘텀확인 6피처가 H5 실현수익과
    # 안정적 순상관(OOS Top-3 현행 -1.68% → PMS +17.68%). enforce=False면 그림자 컬럼만.
    try:
        from profit_momentum import add_profit_momentum_columns, pms_summary
        try:
            from collector_config import DEFAULT_CONFIG as _PMS_CFG
            _pms_enforce = bool(getattr(getattr(_PMS_CFG, "pms", None), "pms_enforce", False))
        except Exception:
            _PMS_CFG, _pms_enforce = None, False
        df_out = add_profit_momentum_columns(df_out, config=_PMS_CFG, enforce=_pms_enforce)
        _ps = pms_summary(df_out)
        log(f"📈 [v24.4] Profit Momentum Overlay (SHADOW{'·ENFORCE정렬보조' if _pms_enforce else ''}) — "
            f"profit_pick {_ps.get('n_profit_pick', 0)} · PMS평균 {_ps.get('pms_mean', 0)} · p90 {_ps.get('pms_p90', 0)}")
    except Exception as e:
        logger.warning(f"⚠️ Profit Momentum Overlay 실패 (기존 추천 유지): {e}")

    # [v24.7] D+1 Checkpoint (조기 손실컷) — 데이터 검증 표시 컬럼 (공식 산식/손절가 불변)
    # 근거: h1↔h5 페어 1,878건 — D1≤-4% 조기청산 시 +2.5%p/건, OOS +3.94%p (비대칭 보험).
    try:
        from d1_checkpoint import add_d1_checkpoint_columns, d1_summary
        try:
            from collector_config import DEFAULT_CONFIG as _D1_CFG
        except Exception:
            _D1_CFG = None
        df_out = add_d1_checkpoint_columns(df_out, config=_D1_CFG)
        _ds = d1_summary(df_out)
        log(f"🛡️ [v24.7] D+1 Checkpoint 표시 — {_ds.get('n_with_checkpoint', 0)}/{_ds.get('n_rows', 0)}종목 컷가격 부여")
    except Exception as e:
        logger.warning(f"⚠️ D+1 Checkpoint 실패 (기존 추천 유지): {e}")

    # [v24.9] 정직 확률 레이어 — 시장기저 2일 상승률(실측) + NO_EDGE 명시 (공식 산식 불변)
    # 근거: OHLCV 29,703표본 walk-forward — 종목별 단기 상승확률 엣지 검증 실패(null),
    # 기저율(51%→37%)이 지배변수. 지어낸 확률 대신 실측치와 미검증 상태를 표시한다.
    try:
        from honest_prob import add_srp_columns, srp_summary
        df_out = add_srp_columns(df_out, data_dir=out_dir if 'out_dir' in dir() else "data")
        _ss = srp_summary(df_out)
        log(f"🎲 [v24.9] 정직확률 — 시장기저2일 {_ss.get('base_prob_2d')}% · {_ss.get('status')}")
    except Exception as e:
        logger.warning(f"⚠️ 정직확률 레이어 실패 (기존 추천 유지): {e}")

    # [v25.1] Exit Discipline Layer — 청산 규율 표시 (손절폭 조임/TP앞당김/ROUTE위험, 공식 불변)
    # 근거: 정직 h5 시뮬 — SL-7%·CARRY/NEUTRAL제외·TP+10% 결합 시 -3.17%→-1.08%.
    try:
        from exit_plan import add_exit_plan_columns, exit_summary
        try:
            from collector_config import DEFAULT_CONFIG as _EX_CFG
        except Exception:
            _EX_CFG = None
        df_out = add_exit_plan_columns(df_out, config=_EX_CFG)
        _es = exit_summary(df_out)
        log(f"🚪 [v25.1] 청산규율 — 고위험루트 {_es.get('high_avoid',0)} · 주의 {_es.get('caution',0)} · OK {_es.get('ok',0)}")
    except Exception as e:
        logger.warning(f"⚠️ Exit Plan 레이어 실패 (기존 추천 유지): {e}")

    # [loss-defense v1] 최종 production 계약. 이 단계는 후보를 새로 승격하지
    # 않고 기존 TOP_PICK+BUY_NOW_ELIGIBLE를 더 엄격하게 거부할 수만 있다.
    try:
        from services.recommendation_quality import apply_recommendation_quality_guard
        # [v29→v31.3] 알파 점수 — 품질게이트 전에 이번 실행에서 학습+검증.
        # 배치 러너가 매번 새 체크아웃이라 모델 바이너리(.gitignore)가 남지
        # 않음 → '메타는 검증통과인데 모델 없음 → 영구 미사용' 버그 수정.
        # 학습 패널은 선행수익률이 확정된 과거(≥6영업일 전)만 포함 — 누출 없음.
        # [v32] 알파는 진입 게이트 SSOT — 반드시 품질게이트보다 먼저 스코어링/게이팅.
        try:
            from alpha_engine import train_and_save as _alpha_train
            from alpha_engine import score_today as _alpha_score_today
            from alpha_engine import apply_alpha_entry_gate as _alpha_gate
            from alpha_engine import recompute_route_with_alpha as _alpha_route
            _am = _alpha_train(OUT_DIR, trade_ymd)
            if _am.get("validated"):
                log(f"🧠 [v29] 알파 학습 — OOS IC {_am.get('mean_ic')} (t={_am.get('ic_t')}) "
                    f"AUC {_am.get('auc')} → 검증 통과")
            else:
                log(f"🧠 [v29] 알파 학습 — 검증 미통과 ({_am.get('reason', 'gate 미달')}) → 미사용")
            df_out = _alpha_score_today(df_out, data_dir=OUT_DIR, trade_ymd=trade_ymd)
            # [v32.1] ROUTE 자체 치료 — 상태 판정을 데이터 방향으로 재계산.
            # (기존 ATTACK -4.64%p → 신 ATTACK +3.70%p, 순서 정상화)
            try:
                _rt_old = df_out.get("ROUTE", pd.Series(dtype=str)).astype(str).str.upper()
                _atk_old = int(_rt_old.isin(["ATTACK", "ARMED"]).sum())
                df_out = _alpha_route(df_out)
                if int(pd.to_numeric(df_out.get("ROUTE_ALPHA_HEALED", 0), errors="coerce").fillna(0).max() if len(df_out) else 0):
                    _rt_new = df_out["ROUTE"].astype(str).str.upper()
                    _atk_new = int(_rt_new.isin(["ATTACK", "ARMED"]).sum())
                    log(f"🧭 [v32.1] ROUTE 치료 — ATTACK/ARMED {_atk_old}→{_atk_new} (강도=알파·상태=구조)")
            except Exception as _re:
                logger.warning(f"⚠️ ROUTE 치료 실패 (기존 ROUTE 유지): {_re}")
            # [v32] 알파 전면 진입 게이트 — ROUTE 거부권 대체.
            _tp_before = int(pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int).sum())
            df_out = _alpha_gate(df_out)
            if int(pd.to_numeric(df_out.get("ALPHA_GATE_ACTIVE", 0), errors="coerce").fillna(0).iloc[0] if len(df_out) else 0):
                _tp_after = int(pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int).sum())
                _reg = str(df_out.get("MARKET_REGIME", pd.Series([""])).iloc[0]) if len(df_out) else ""
                log(f"🧠 [v32] 알파 전면 진입 게이트 적용 — 레짐 {_reg} · TOP_PICK {_tp_before}→{_tp_after} (ROUTE 거부권 제거)")
            else:
                log("🧠 [v32] 알파 미검증 → 레거시 폴백 게이트 (ROUTE 거부권 없이)")

            # [v59] 업종 결측 복구 — **켈리 사이징보다 먼저** 돌아야 한다.
            #   2026-08-07 배치 실측: 326종목 중 30종목(9.2%)이 업종_대분류
            #   결측이었고, 그 행들은 섹터/뉴스/전략 단계 이후에 합류해 18개
            #   컬럼(LDY_SCORE·NEWS_SCORE 등 포함)을 못 받은 상태였다.
            #   결과 ① 시장 탭 plotly 트리맵이 계층을 못 만들어 '로딩 실패'
            #        ② 켈리 섹터 모멘텀이 결측을 '?' 한 덩어리로 묶어 무관한
            #           30종목에 같은 배수를 줬다(→ 켈리_수량 오염)
            #   여기서 업종(상세)으로부터 같은 분류기로 복구하고, 복구 못 한
            #   행은 결측으로 남긴다(가짜 섹터 금지). 결측 사실은 로그로 남긴다 —
            #   조용히 채우면 '행이 늦게 합류하는' 근본 원인이 또 숨는다.
            try:
                from services.sector_repair import (
                    repair_sector, sector_repair_line, is_alarming)
                df_out, _srep = repair_sector(df_out)
                _sline = sector_repair_line(_srep)
                if _sline:
                    log(f"{'🚨' if is_alarming(_srep) else '🏷️'} [v59] {_sline}")
                    if is_alarming(_srep):
                        log("🚨 [v59] 업종 결측은 그 행이 섹터/뉴스/전략 단계 뒤에 "
                            "합류했다는 신호다 — LDY_SCORE·NEWS_SCORE 등도 비어 있다")
            except Exception as _se:
                log(f"⚠️ [v59] 업종 결측 복구 스킵: {_se}")

            # [v33.1] 켈리 사이징 배선 — collector 단계 켈리는 알파 주입 전에
            # 돌아 ALPHA_WIN_PROB를 못 봤다(7/19 첫 배치 실측: KELLY_P_SOURCE
            # 전부 레거시). 알파 게이트 직후 재계산해 사이징 축을 알파로 통일.
            try:
                from kelly_calibrator import resize_kelly_with_alpha
                df_out = resize_kelly_with_alpha(df_out, OUT_DIR, asof_ymd=trade_ymd)
                if "KELLY_P_SOURCE" in df_out.columns:
                    _n_alpha_p = int((df_out["KELLY_P_SOURCE"].astype(str) == "ALPHA_WIN_PROB").sum())
                    if _n_alpha_p:
                        log(f"💰 [v33.1] 켈리 알파 사이징 재계산 — ALPHA_WIN_PROB 적용 {_n_alpha_p}건")
            except Exception as _ke:
                logger.warning(f"⚠️ 켈리 알파 재계산 실패 (기존 사이징 유지): {_ke}")
        except Exception as _ae:
            logger.warning(f"⚠️ 알파 학습/점수/게이트 실패 (미사용 처리): {_ae}")
        _before_quality = int(((pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int) == 1)
                               & (pd.to_numeric(df_out.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0).astype(int) == 1)).sum())
        df_out = apply_recommendation_quality_guard(df_out)
        _after_quality = int(pd.to_numeric(df_out.get("PRODUCTION_BUY", 0), errors="coerce").fillna(0).astype(int).sum())

        # [v64] 세션 신선도 — 기준일이 실제 가격일과 다르면 **사실대로 적는다**.
        #   2026-08-17(광복절 대체공휴일) 배치가 8/14 가격으로 `기준일=20260817`,
        #   `RUN_STATUS=OK`를 찍고 TOP_PICK 12건·공식매수 1건을 냈다. 392종목
        #   종가가 8/14와 100% 동일했고 OHLCV에 8/17 거래일 자체가 없었다.
        #   원인은 collector.find_latest_valid_date의 4단계 폴백(IP차단 대비
        #   '최근 평일 강제 진행')이 공휴일에도 같은 경로를 타는 것이다.
        #   이력 전수: 배치 124일 중 비거래일 7일.
        #   실제 진입 왜곡은 이번엔 작았다(8/18 갭 중위 -0.05% · 손절터치 0/12)
        #   → 그래서 **픽을 죽이지 않고 표시만 정직하게** 한다(v45·v51·v61 원칙).
        try:
            from services.session_freshness import (annotate as _sf_annotate,
                                                    assess as _sf_assess,
                                                    is_alarming as _sf_alarm,
                                                    line as _sf_line)
            _sf_rep = _sf_assess(ctx.trade_ymd, OUT_DIR)
            df_out = _sf_annotate(df_out, _sf_rep)
            _sfl = _sf_line(_sf_rep)
            if _sfl:
                log(f"{'🚨' if _sf_alarm(_sf_rep) else '📅'} [v64] {_sfl}")
            if _sf_alarm(_sf_rep):
                log("   🚨 [v64] 이 배치의 가격·시장폭은 전 거래일 값이다 — "
                    "'익일 지정가'가 가격 기준일의 익일이 아니다")
            ctx.breadth["SESSION_STALE"] = bool(_sf_rep.get("stale"))
            ctx.breadth["PRICE_ASOF"] = _sf_rep.get("price_asof")
        except Exception as _se:
            logger.warning(f"⚠️ [v64] 세션 신선도 판정 실패 (표시만 영향): {_se}")

        # [v63] 공식 퍼널 라벨 재계산 — **품질게이트 뒤**가 맞는 자리다.
        #   v62에서 알파 게이트 직후에 뒀는데, 그 뒤에 품질게이트가 '당일 신규진입
        #   1종목 제한'으로 PRODUCTION_BUY를 잘라내므로 라벨이 다시 낡았다.
        #   v62는 한쪽 모순을 고치고 반대쪽을 열었다 — 2026-08-17 배치 실측:
        #     OFFICIAL_FUNNEL_STAGE="OFFICIAL_BUY"  ← 공식 매수라고 말하는데
        #     PRODUCTION_BUY=0 · ACTION_DECISION=WATCH
        #     OFFICIAL_BLOCK_REASON_2="TOP_PICK + BUY_NOW_ELIGIBLE"
        #       ← 근거로 든 BUY_NOW_ELIGIBLE이 정작 0이다
        #   같은 행이 '공식 매수'이면서 '매수 아님'이라고 말했다. 10건/1일.
        #   (v62 이전에는 반대 방향으로 낡아 'TOP_PICK=0'이라고 했다 — 11건.)
        #   품질게이트가 PRODUCTION_BUY·BUY_NOW_ELIGIBLE을 바꾸는 마지막 단계이므로
        #   라벨은 그 뒤에 굳어야 한다. 표시 컬럼만 갱신하고, 계약 컬럼이 바뀌면
        #   원복 + 에러 로그를 남긴다.
        try:
            _fn_cols = [c for c in ["TOP_PICK", "BUY_NOW_ELIGIBLE",
                                    "BUY_NOW_PASS", "BUY_NOW_GRADE",
                                    "PRODUCTION_BUY"]
                        if c in df_out.columns]
            _fn_before = df_out[_fn_cols].copy() if _fn_cols else None
            _stale = 0
            if "OFFICIAL_FUNNEL_STAGE" in df_out.columns:
                _pb_now = pd.to_numeric(df_out.get("PRODUCTION_BUY", 0),
                                        errors="coerce").fillna(0).astype(int)
                _tp_now = pd.to_numeric(df_out.get("TOP_PICK", 0),
                                        errors="coerce").fillna(0).astype(int)
                _stale = int(
                    ((df_out["OFFICIAL_FUNNEL_STAGE"].astype(str) == "OFFICIAL_BUY")
                     & (_pb_now == 0)).sum())
                _stale += int(((_tp_now == 1)
                               & df_out.get("OFFICIAL_BLOCK_REASON_1",
                                            pd.Series("", index=df_out.index))
                                 .astype(str).str.contains("TOP_PICK=0")).sum())
            df_out = add_official_buy_funnel_columns(
                df_out,
                macro_risk=ctx.macro_risk,
                market_breadth=ctx.breadth.get("ALL", np.nan),
                macro_msg=getattr(ctx, "macro_msg", ""),
            )
            if _fn_before is not None:
                for _c in _fn_cols:
                    if not df_out[_c].equals(_fn_before[_c]):
                        logger.error("[v63] 퍼널 재계산 중 %s 변경 감지 — 원복", _c)
                        df_out[_c] = _fn_before[_c]
            if _stale:
                log(f"🧭 [v63] 공식 퍼널 라벨 재계산 — 결정과 어긋난 라벨 {_stale}건 해소")
        except Exception as _fe:
            logger.warning(f"⚠️ [v63] 퍼널 라벨 재계산 실패 (표시만 영향): {_fe}")
        # [v28] "왜 이 종목인가" 근거 문장 — 실측 수치 기반, 리스크 병기
        try:
            from services.reco_evidence import add_evidence_columns
            df_out = add_evidence_columns(df_out)
        except Exception as _ee:
            logger.warning(f"⚠️ 근거 문장 생성 실패 (추천 유지): {_ee}")
        df_out = finalize_sort(df_out)
        df_out["LDY_RANK"] = np.arange(1, len(df_out) + 1)
        log(f"🧱 [loss-defense v1] 최종 품질게이트 — official {_before_quality}->{_after_quality} · 현금보유 허용")
        # [v55] #29 계측 — 차단 상태와 그날 픽 수량을 같은 줄에 남긴다.
        #   '왜 추천이 없냐'를 배치 로그만 보고 답할 수 있어야 한다. 실제로 v55에서
        #   원인 규명에 CSV를 직접 파야 했던 이유가 이 한 줄이 없었기 때문이다
        #   (최근 16영업일 중 15일이 risk_off 전 종목 차단이었다).
        try:
            from services.entry_block_status import compute_entry_block_status

            _blk = compute_entry_block_status(OUT_DIR)
            _nb = int(pd.to_numeric(df_out.get("NEW_ENTRY_BLOCKED", 0),
                                    errors="coerce").fillna(0).astype(bool).sum())
            _tp = int(pd.to_numeric(df_out.get("TOP_PICK", 0),
                                    errors="coerce").fillna(0).astype(int).sum())
            log(f"🚧 [v55] 진입차단 계측 — risk_off={_blk.get('risk_off')} "
                f"(이탈 {_blk.get('deviation_pct')}% · 해제까지 {_blk.get('unlock_gap_pct')}%p · "
                f"연속 {_blk.get('streak_days')}일 · 60일 차단율 {_blk.get('block_rate_60d_pct')}%) "
                f"· NEW_ENTRY_BLOCKED {_nb}/{len(df_out)} · TOP_PICK {_tp} "
                f"· PRODUCTION_BUY {_after_quality}")
        except Exception as _be:
            logger.warning(f"⚠️ [v55] 진입차단 계측 실패 (무해): {_be}")
    except Exception as e:
        logger.error(f"❌ 최종 품질게이트 실패 — 안전상 신규매수 전부 차단: {e}", exc_info=True)
        df_out["PRODUCTION_BUY"] = 0
        df_out["BUY_NOW_ELIGIBLE"] = 0
        df_out["ACTION_DECISION"] = "CASH"
        df_out["RECOMMENDED_WEIGHT_PCT"] = 0.0
        df_out["QUALITY_GUARD_REASON"] = "품질게이트 실행 실패"

    # ── CSV 저장 (분석 시점 불변 원본) ──
    ensure_dir(OUT_DIR)
    op_d = os.path.join(OUT_DIR, f"recommend_{trade_ymd}{f'_{ctx.tag}' if ctx.tag else ''}.csv")
    op_l = os.path.join(OUT_DIR, "recommend_latest.csv")
    # 종목명 오염 복구
    if "종목명" in df_out.columns and "종목코드" in df_out.columns:
        df_out["종목명"] = df_out["종목명"].astype(str)
        _cm2 = df_out["종목명"].str.match(r'^\d+$'); _cc = _cm2.sum()
        if _cc > 0:
            if ctx.name_map:
                df_out.loc[_cm2,"종목명"] = df_out.loc[_cm2,"종목코드"].astype(str).str.zfill(6).map(ctx.name_map).fillna(df_out.loc[_cm2,"종목명"])
            _sc2 = df_out["종목명"].str.match(r'^\d+$')
            if _sc2.sum() > 0:
                _sp = os.path.join(OUT_DIR, "price_snapshot_latest.csv")
                if os.path.exists(_sp):
                    try:
                        _sn = pd.read_csv(_sp, dtype={"종목코드":str}, usecols=["종목코드","종목명"])
                        _sm = dict(zip(_sn["종목코드"].str.zfill(6), _sn["종목명"]))
                        df_out.loc[_sc2,"종목명"] = df_out.loc[_sc2,"종목코드"].astype(str).str.zfill(6).map(_sm).fillna(df_out.loc[_sc2,"종목명"])
                        ctx.name_map.update({c:n for c,n in _sm.items() if c!=n})
                    except Exception as _e: log(f"⚠️ snapshot 폴백 실패: {_e}")
            _fc = df_out["종목명"].str.match(r'^\d+$').sum(); _fx = _cc - _fc
            if _fx > 0: log(f"🔧 종목명 복구: {_fx}/{_cc}건")
            if _fc > 0:
                _pat = r'^\d+$'
                _remain = df_out.loc[df_out['종목명'].str.match(_pat), '종목코드'].tolist()[:5]
                log(f"⚠️ 미복구 {_fc}건: {_remain}")
    df_out.to_csv(op_d, index=False, encoding=UTF8)
    df_out.to_csv(op_l, index=False, encoding=UTF8)
    log(f"💾 저장 완료 ({len(df_out)}건) → {op_d}")

    # ── [v20.6.3] run_meta JSON sidecar ──
    try:
        import json as _json
        _meta = {
            "trade_ymd": trade_ymd,
            "macro_risk": ctx.macro_risk,
            "macro_msg": ctx.macro_msg,
            "market_breadth": ctx.breadth.get("ALL", np.nan),
            "pass_ebs": ctx.pass_ebs,
            "rec_limit": ctx.rec_limit,
            "n_stocks": len(df_out),
            "run_status": _health.status if _health else "UNKNOWN",
            "confidence_score": _health.confidence_score if _health else 0.0,
            "max_allowed_route": _health.max_allowed_route if _health else "ATTACK",
            "scoring_axes": df_out["SCORING_AXES"].iloc[0] if "SCORING_AXES" in df_out.columns else "",
            "w_struct": float(df_out["W_STRUCT"].iloc[0]) if "W_STRUCT" in df_out.columns else 0.0,
            "w_timing": float(df_out["W_TIMING"].iloc[0]) if "W_TIMING" in df_out.columns else 0.0,
            "w_ai": float(df_out["W_AI"].iloc[0]) if "W_AI" in df_out.columns else 0.0,
        }
        _meta_d = os.path.join(OUT_DIR, f"run_meta_{trade_ymd}.json")
        _meta_l = os.path.join(OUT_DIR, "run_meta_latest.json")
        for _mp in [_meta_d, _meta_l]:
            with open(_mp, 'w', encoding='utf-8') as _mf:
                _json.dump(_meta, _mf, ensure_ascii=False, indent=2, default=str)
        log(f"📋 run_meta 저장 완료 → {_meta_d}")
    except Exception as _me:
        logger.warning(f"⚠️ run_meta 저장 실패 (무해): {_me}")

    # ══════════════════════════════════════════════════════════
    #  [v20.6.4] After-market → sidecar 파일로 분리
    #  recommend_latest.csv는 절대 수정하지 않음 (원본 보존)
    # ══════════════════════════════════════════════════════════
    try:
        from naver_aftermarket import fetch_after_market_prices_sidecar
        _snl = os.path.join(OUT_DIR, 'price_snapshot_latest.csv')
        _sidecar_path = os.path.join(OUT_DIR, 'aftermarket_prices_latest.csv')
        _ac = fetch_after_market_prices_sidecar(op_l, _sidecar_path, _snl)
        if _ac > 0:
            log(f'After-market sidecar: {_ac} stocks → {_sidecar_path}')
        else:
            log('After-market: no changes')
    except ImportError:
        # sidecar 함수 없으면 시간외 업데이트 스킵 (원본 보존 원칙)
        log('After-market: sidecar 함수 없음 — 스킵 (recommend 원본 보존)')
    except Exception as e:
        log(f'After-market sidecar failed: {e}')

    # 종목명 매핑
    try:
        _sp2 = os.path.join(OUT_DIR, "price_snapshot_latest.csv")
        if os.path.exists(_sp2):
            _nd = pd.read_csv(_sp2, dtype={"종목코드":str}, usecols=["종목코드","종목명"])
            _nd["종목코드"]=_nd["종목코드"].str.zfill(6); _nd=_nd.drop_duplicates("종목코드")
        else: _nd = df_out[["종목코드","종목명"]].drop_duplicates("종목코드")
        if ctx.name_map:
            _ex = [{"종목코드":c,"종목명":n} for c,n in ctx.name_map.items() if c not in _nd["종목코드"].values and c!=n and not n.isdigit()]
            if _ex: _nd = pd.concat([_nd, pd.DataFrame(_ex)], ignore_index=True)
        _nd = _nd[_nd["종목명"].astype(str)!=_nd["종목코드"].astype(str)]
        _np = os.path.join(OUT_DIR, "krx_names_latest.csv")
        _nd.to_csv(_np, index=False, encoding=UTF8)
        log(f"📋 종목명 매핑 저장: {len(_nd)}건 → {_np}")
    except Exception as e: log(f"⚠️ 종목명 매핑 실패: {e}")
    # DB
    try:
        from db_utils import get_db; get_db().save_recommendations(df_out, trade_ymd)
    except Exception as e: log(f"⚠️ DB 저장 실패: {e}")
    # Reality Check + Rank Validation
    run_reality_check(OUT_DIR, trade_ymd)
    make_rank_validation_report(OUT_DIR, asof_ymd=trade_ymd, methods=["ELITE_SCORE","DISPLAY_SCORE","FINAL_SCORE","AI_SCORE"])
    # [v22.3] monotonicity_report 인라인 생성 — daily_briefing.py 별도 실행 의존성 제거
    # 평가 피드백: "ZIP 기준 최신 검증 리포트가 항상 따라오는 구조" 보장
    try:
        from daily_briefing import generate_monotonicity_report
        _mono = generate_monotonicity_report(OUT_DIR, trade_ymd)
        _mono_status = _mono.get("ci_hard", [{}])[0].get("status", "?") if _mono.get("ci_hard") else "OK"
        log(f"📊 [v22.3] monotonicity_report → {trade_ymd} (status={_mono_status})")
    except ImportError:
        log("ℹ️ daily_briefing 모듈 없음 — monotonicity_report SKIP")
    except Exception as e:
        log(f"⚠️ monotonicity_report 생성 실패: {e}")
    # [v21.2+v22] TOP_PICK 검증 리포트 — 0건에도 latest 갱신 (CI 오독 차단)
    try:
        import json as _json2
        _tp_mask = df_out.get("TOP_PICK", pd.Series(0, index=df_out.index)).astype(int) == 1
        _tp_count = int(_tp_mask.sum())
        _tp_path = os.path.join(OUT_DIR, f"top_pick_validation_{trade_ymd}.json")
        _tp_latest = os.path.join(OUT_DIR, "top_pick_validation_latest.json")

        if _tp_count > 0:
            _tp_df = df_out[_tp_mask].copy()
            # [v22] AGGRESSIVE/STABLE 분리 집계
            _by_type = (_tp_df["TOP_PICK_TYPE"].value_counts().to_dict()
                        if "TOP_PICK_TYPE" in _tp_df.columns else {})
            _tp_summary = {
                "trade_ymd": trade_ymd,
                "top_pick_count": _tp_count,
                "top_pick_by_type": _by_type,
                "avg_elite": round(float(_tp_df["ELITE_SCORE"].mean()), 1),
                "avg_rr": round(float(_tp_df["RR_NOW_TP1"].mean()), 2),
                # [v22.3.1] 최소 RR + RR<1 카운트 — 평균은 위장 가능, 최소가 진실
                "min_rr": round(float(_tp_df["RR_NOW_TP1"].min()), 2),
                "rr_lt_1_count": int((pd.to_numeric(_tp_df["RR_NOW_TP1"], errors="coerce").fillna(0) < 1.0).sum()),
                "avg_balance": round(float(_tp_df["BALANCE_SCORE"].mean()), 1),
                "avg_win_rate": round(float(_tp_df["EST_WIN_RATE"].mean()), 3),
                "est_win_rate_method": (_tp_df["EST_WIN_RATE_METHOD"].iloc[0]
                                         if "EST_WIN_RATE_METHOD" in _tp_df.columns else "UNKNOWN"),
                "est_win_rate_mode": (_tp_df["EST_WIN_RATE_MODE"].iloc[0]
                                       if "EST_WIN_RATE_MODE" in _tp_df.columns else "UNKNOWN"),
                "est_win_rate_n": (int(_tp_df["EST_WIN_RATE_N"].iloc[0])
                                    if "EST_WIN_RATE_N" in _tp_df.columns else 0),
                "routes": _tp_df["ROUTE"].value_counts().to_dict(),
                "picks": _tp_df[[
                    c for c in [
                        "종목코드", "종목명",
                        "TOP_PICK_TYPE",
                        "ELITE_SCORE", "RR_NOW_TP1", "BALANCE_SCORE",
                        "ENTRY_GAP_PCT", "TP1_PCT",
                        "ROUTE",
                        "EST_WIN_RATE", "EST_WIN_RATE_METHOD",
                        "EST_WIN_RATE_MODE", "EST_WIN_RATE_N",
                    ] if c in _tp_df.columns
                ]].to_dict("records"),
            }
            _type_msg = (f" (AGGR={_by_type.get('AGGRESSIVE',0)}, "
                         f"STBL={_by_type.get('STABLE',0)})" if _by_type else "")
            log(f"🏆 TOP_PICK 검증: {_tp_count}종목{_type_msg} → {_tp_path}")
        else:
            # [v22] 0건 날에도 latest 갱신 — stale 방지
            _meta_method = "NONE"
            _meta_mode = "NONE"
            _meta_n = 0
            if "EST_WIN_RATE_METHOD" in df_out.columns and len(df_out) > 0:
                _meta_method = str(df_out["EST_WIN_RATE_METHOD"].iloc[0])
            if "EST_WIN_RATE_MODE" in df_out.columns and len(df_out) > 0:
                _meta_mode = str(df_out["EST_WIN_RATE_MODE"].iloc[0])
            if "EST_WIN_RATE_N" in df_out.columns and len(df_out) > 0:
                try:
                    _meta_n = int(df_out["EST_WIN_RATE_N"].iloc[0])
                except Exception:
                    _meta_n = 0
            _tp_summary = {
                "trade_ymd": trade_ymd,
                "top_pick_count": 0,
                "top_pick_by_type": {},
                "avg_elite": None,
                "avg_rr": None,
                # [v22.3.1] 0건 케이스에도 동일 필드 — null 안정성
                "min_rr": None,
                "rr_lt_1_count": 0,
                "avg_balance": None,
                "avg_win_rate": None,
                "est_win_rate_method": _meta_method,
                "est_win_rate_mode": _meta_mode,
                "est_win_rate_n": _meta_n,
                "routes": {},
                "picks": [],
            }
            log(f"🏆 TOP_PICK: 0종목 (게이트 미통과) — latest.json 갱신")

        for _p in [_tp_path, _tp_latest]:
            with open(_p, 'w', encoding='utf-8') as _f:
                _json2.dump(_tp_summary, _f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.warning(f"⚠️ TOP_PICK 검증 실패: {e}")
    # 텔레그램
    if ctx.enable_telegram:
        mkt = label_market_temp(ctx.breadth.get("ALL", np.nan))
        st = f"🌡 {mkt} (Breadth: {ctx.breadth.get('ALL',0)}%)"
        if ctx.macro_msg: st += f"\n{ctx.macro_msg}"
        if "SECTOR_RANK" in df_out.columns:
            ts2 = df_out.sort_values("SECTOR_RS",ascending=False)["업종_대분류"].unique()[:2]
            st += f"\n🚀 주도: {' '.join(ts2)}"
        send_telegram_auto(df_out, trade_ymd, market_summary=st, limit_count=ctx.rec_limit)
    else: log("✉️ 텔레그램 발송 생략")
    # 자동 캘리브레이션
    try:
        from auto_backtest import auto_calibrate
        cs = auto_calibrate(OUT_DIR, trade_ymd)
        log(f"📊 캘리브레이션: {cs.get('n_trades',0)}건, 승률={cs.get('overall_winrate',0):.1%}")
    except Exception as e: log(f"⚠️ 자동 캘리브레이션 스킵: {e}")
    # [v31.3] 알파 학습은 스코어링 직전(위)으로 이동 — 야간 별도 재학습 제거.
    # [v28] 점수 축 예측력 야간 감사 — 역주행(t<-2) 축 경고
    # AI 축만 신뢰도 게이트가 있고 룰 기반 축은 무검증이던 비대칭 해소.
    try:
        from services.axis_ic_report import save_axis_ic_report
        _icr = save_axis_ic_report(OUT_DIR, trade_ymd)
        if _icr.get("ok"):
            _warn_n = len(_icr.get("warnings", []))
            log(f"🧭 [v28] 축 IC 감사: {_icr.get('n_days_used',0)}일 기준"
                + (f" · ⚠️ 역주행 {_warn_n}축" if _warn_n else " · 이상 없음"))
    except Exception as e:
        log(f"⚠️ 축 IC 감사 스킵: {e}")
    # [v68] 선언 승률 vs 같은 점수 구간 실측 — 과신 방지 캡이 8월 내내
    #   조용히 미적용이었다. 원인 둘: (1) 픽이 사는 ELITE_SCORE [0,50) 구간의
    #   winrate_table 표본이 n_raw=2(폴백 p_win=0.5)라 신뢰 bin이 없었고,
    #   (2) compute_est_win_rate(pipeline_calibrate)가 auto_calibrate
    #   (이 파일 아래)보다 먼저 돌아 캡이 늘 전날 표를 읽는다.
    #   실측: 08-24 선언 45% vs 같은 구간 실측 19%(n=32) — 28/28종목 캡 초과.
    #   여기서는 **진단만** 한다 — EST_WIN_RATE·켈리 수량을 바꾸지 않는다.
    try:
        from services import winrate_truth as _wt
        _wtab = _wt.load_table(OUT_DIR, trade_ymd)
        if _wtab:
            df_out = _wt.annotate(df_out, _wtab)
            _tpm = pd.to_numeric(df_out.get("TOP_PICK"), errors="coerce").fillna(0) > 0
            _ws = _wt.summary(df_out, _wtab, mask=_tpm)
            if _ws.get("n"):
                log(f"🎲 [v68] 선언 승률 검증: {_ws['status_counts']}")
                if _ws.get("line"):
                    log(f"⚠️ [v68] {_ws['line']}")
            # 고장 ③ — 캡이 읽는 표와 리포트가 읽는 표가 어긋나면 알린다
            _wd = _wt.table_divergence(OUT_DIR)
            if _wd.get("diverged"):
                log(f"🚨 [v68] {_wd['line']}")
        else:
            log("⚠️ [v68] winrate_table 없음 — 선언 승률 검증 생략")
    except Exception as e:
        log(f"⚠️ 선언 승률 검증 스킵: {e}")

    # [v67] 보유 청산 규율 — 진입가 고정 손절선 기준으로 보유 상태를 기록한다.
    #   2026-08-24 전수: CARRY 58종목 중 55종목이 진입가 기준 -7% 손절선을 이미
    #   관통했고 평균 -30.4%였다(고정 손절 준수 시 -6.7%, 종목당 +23.8%p,
    #   페어드 t=9.33 p<1e-6). 이루온은 진입 다음날 관통 후 101일째였다.
    #   결정 컬럼은 건드리지 않는다 — 표시·경보 전용이다.
    try:
        from services.holding_exit import annotate as _he_annotate, summary as _he_sum
        df_out = _he_annotate(df_out)
        _hs = _he_sum(df_out)
        if _hs.get("n"):
            log(f"📌 [v67] 보유 {_hs['n']}종목 · 조치필요 {_hs['actionable']}건 "
                f"· 상태 {_hs['counts']}")
            if _hs.get("line"):
                log(f"⛔ [v67] {_hs['line']}")
    except Exception as e:
        log(f"⚠️ 보유 청산 규율 주석 스킵: {e}")

    # [v58] 알파 실전 성적 야간 누적 — 진입 SSOT도 매일 자기 성적을 남긴다.
    # v28이 룰 기반 축을 감사하게 만든 것과 같은 이유다. 이게 없어서
    # 2026-08-10에 "최근 구간은 아직 측정 불가"라고만 답할 수 있었다
    # (데이터는 있었고 누적된 성적표가 없었다).
    try:
        from services.alpha_live_report import save_alpha_live_report, alpha_live_line
        _alr = save_alpha_live_report(OUT_DIR, trade_ymd)
        _line = alpha_live_line(_alr)
        if _line:
            log(f"🎯 [v58] {_line}")
        elif not _alr.get("ok"):
            log(f"🎯 [v58] 알파 실전 성적: {_alr.get('reason', '집계 불가')}")
        for _w in (_alr.get("warnings") or []):
            log(f"🚨 [v58] {_w}")
        # [v65] 배포 확인용 — 유령 세션(휴장 복사본) 제거와 공식 픽 표본을
        #   로그에 남긴다. 이 두 줄이 없으면 성적표가 무엇을 셌는지 사후에
        #   알 수 없다(v63 리포트가 픽 없는 날 18/21일을 세고 있었던 이유).
        _s = _alr.get("sessions") or {}
        if _s.get("phantom_dropped"):
            log(f"🧾 [v65] 휴장 스냅샷 {_s['phantom_dropped']}일 제외 "
                f"(세션 {_s.get('real')}/{_s.get('snapshots')}) "
                f"— 최근: {', '.join(_s.get('phantom_days', [])[-3:])}")
        _ob = ((_alr.get("horizons") or {}).get("h5") or {}).get("official") or {}
        if _ob:
            log(f"🧾 [v65] 공식 매수 표본: 픽 {_ob.get('pick_days_declared', 0)}건"
                f"/{_ob.get('days_recorded', 0)}일"
                f" · 측정가능 {_ob.get('pick_days', 0)}건"
                f" · 미확정 {_ob.get('days_pick_unmeasured', 0)}건")
    except Exception as e:
        log(f"⚠️ 알파 실전 성적 리포트 스킵: {e}")
    # [v21.3] 조합 최적화
    try:
        from combo_optimizer import run_combo_optimization
        opt = run_combo_optimization(OUT_DIR, horizon=3, min_samples=10)
        if opt and opt.get("best"):
            b = opt["best"]
            log(f"🎯 최적 조합: S≥{b['S_min']} T≥{b['T_min']} AI≥{b['AI_min']} | 승률 {b['win_rate']}%")
    except Exception as e:
        log(f"⚠️ 조합 최적화 스킵: {e}")
    # 포지션
    try:
        from position_tracker import track_open_positions, register_from_recommendations
        register_from_recommendations(OUT_DIR, df_out, trade_ymd, top_n=ctx.rec_limit)
        tr = track_open_positions(OUT_DIR, trade_ymd)
        log(f"📍 포지션: 체크={tr.get('checked',0)}, 이벤트={tr.get('events',0)}, 청산={tr.get('closed',0)}")
    except Exception as e: log(f"⚠️ 포지션 트래킹 스킵: {e}")
    # 브리핑
    try:
        from daily_briefing import generate_daily_briefing
        br = generate_daily_briefing(OUT_DIR, trade_ymd, df_out)
        if br["count"]>0: log(f"📝 일일 브리핑: {br['count']}종목 [{', '.join(br.get('names',[]))}]")
        else: log("📝 일일 브리핑: 대상 없음")
    except Exception as e: log(f"⚠️ 일일 브리핑 스킵: {e}")
    ctx.df_out = df_out
