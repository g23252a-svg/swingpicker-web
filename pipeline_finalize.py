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
    
    close = pd.to_numeric(df.get("종가", 0), errors="coerce").fillna(0)
    entry = pd.to_numeric(df.get("추천매수가", 0), errors="coerce").fillna(0)
    # ATR_Pct(decimal) 우선, ATR_PCT(percentage)도 허용 — 내부 정규화
    atr = df.get("ATR_Pct", df.get("ATR_PCT", pd.Series(0.02, index=df.index)))
    mcap = pd.to_numeric(df.get("시가총액(억원)", 0), errors="coerce").fillna(0)
    
    return pd.Series(
        [_cine(c, e, a, m) for c, e, a, m in zip(close, entry, atr, mcap)],
        index=df.index,
        dtype=int,
    )


def finalize_sort(df: pd.DataFrame) -> pd.DataFrame:
    """[v22] SORT_SPEC — 8축 정렬 SSOT.
    
    정렬 우선순위 (내림차순 기준, 낮은 ROUTE_PRIORITY가 먼저):
      1. TOP_PICK (1 먼저)
      2. IS_NOW_ENTRY (1 먼저, adaptive 기반)
      3. ROUTE_PRIORITY (낮을수록 먼저: ATTACK=1 → CARRY=7)
      4. ELITE_SCORE (높을수록)
      5. RR_NOW_TP1 (높을수록)
      6. BALANCE_SCORE (높을수록)
      7. ENTRY_GAP_PCT (낮을수록)
      8. DISPLAY_SCORE (높을수록)
    
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
        ("TOP_PICK", 0), ("IS_NOW_ENTRY", 0),
        ("ELITE_SCORE", 0), ("RR_NOW_TP1", 0),
        ("BALANCE_SCORE", 0), ("ENTRY_GAP_PCT", 99),
        ("DISPLAY_SCORE", 0),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # SORT_SPEC 적용
    df = df.sort_values(
        by=["TOP_PICK", "IS_NOW_ENTRY", "_ROUTE_PRIORITY", "ELITE_SCORE",
            "RR_NOW_TP1", "BALANCE_SCORE", "ENTRY_GAP_PCT", "DISPLAY_SCORE"],
        ascending=[False, False, True, False, False, False, True, False],
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

    # [v3.9.26] Evidence-Gated No-Buy Breaker — 검증 통과 룰이 있을 때만 최대 1개 공식 승격
    try:
        _official_before = int(((pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int) == 1)
                                & (pd.to_numeric(df_out.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0).astype(int) == 1)).sum())
        df_out = apply_evidence_gated_no_buy_breaker(df_out, out_dir=OUT_DIR)
        _official_after = int(((pd.to_numeric(df_out.get("TOP_PICK", 0), errors="coerce").fillna(0).astype(int) == 1)
                               & (pd.to_numeric(df_out.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0).astype(int) == 1)).sum())
        if _official_after > _official_before:
            log(f"🧬 [v3.9.26] Evidence-Gated No-Buy Breaker 공식 후보 승격: {_official_after - _official_before}건")
        else:
            _dec = df_out.get("NO_BUY_BREAKER_DECISION", pd.Series("", index=df_out.index)).astype(str).value_counts().to_dict()
            log(f"🧬 [v3.9.26] No-Buy Breaker 비활성/보류 — {_dec}")
    except Exception as e:
        logger.warning(f"⚠️ apply_evidence_gated_no_buy_breaker 실패 (기존 공식추천 유지): {e}")

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
