"""
services/portfolio_swap.py
============================
[v3.9.21] 보유종목 vs 신규추천 교체 판단 — UI 비의존 로직 SSOT.

목표: 사용자가 매일 보는 보유종목에 대해 유지/감량/교체/정리 우선 판단.

질문 흐름 (검증 도구 5단계 다음):
  내 보유종목 계속 들고 갈까?
  오늘 추천 종목으로 갈아탈까?
  물타기/추매해도 되나?
  손절해야 하나?

설계:
- 입력: 보유종목 리스트 + 오늘 recommend CSV
- 보유종목 각각에 대해 현재 SwingPicker 지표 매칭 (recommend CSV에서 lookup)
- 신규추천 Top Pick과 비교
- 6단계 판정:
    🟢 유지              : 점수 양호 + ROUTE 정상 + 비중 적정
    🔵 유지+신규매수 금지 : 보유 OK but 신규 신호도 없음
    🟡 감량 검토         : 비중 과다 또는 점수 약함
    🟠 교체 후보         : 보유 약함 + 신규추천 강함 (anomaly 없음)
    🔴 정리 우선         : EBS 0 + ROUTE WAIT/EXIT + 손실
    ⚪ 데이터 부족       : 보유종목이 recommend에 없거나 정보 부족

UI 비의존:
- nicegui import 0
- services 외 의존 0 (pandas만)
- recommend_df는 호출자가 미리 로드
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

_logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# [v3.9.21] 교체 판단 임계값 — 모듈 상수
# ════════════════════════════════════════════════════════════════
# 보유종목 약함 기준
HOLD_WEAK_FINAL_SCORE = 60.0           # FINAL_SCORE < 60 → 약함
HOLD_WEAK_LOSS_PCT = -5.0              # 손익률 < -5% → 약함
HOLD_OVER_CONCENTRATION_PCT = 30.0     # 비중 > 30% → 과집중

# 신규추천 강함 기준
NEW_STRONG_FINAL_SCORE = 70.0          # FINAL_SCORE ≥ 70 → 강함
NEW_STRONG_RR = 1.5                    # RR_NOW_TP1 ≥ 1.5 → 강함

# 교체 추천을 위한 점수 차이
SWAP_MIN_SCORE_GAP = 10.0              # 신규 - 보유 ≥ 10점 차이 필요

# 신규추천 anomaly/과열 차단 기준 (교체 추천 금지 조건)
NEW_DANGER_ENTRY_GAP_PCT = 5.0         # ENTRY_GAP > 5% → 과열
NEW_DANGER_TOTAL_RET_ABS = 300.0       # 비현실 수익률 → anomaly
NEW_DANGER_TP_SATURATION = 80.0        # [v3.9.21b] TP 포화 ≥ 80% → 진입 신호 의심

# 정리 우선 ROUTE
LIQUIDATE_ROUTES = {"WAIT", "EXIT_WARNING", "EXIT"}
DANGER_ROUTES = {"OVERHEAT"}

# 회피 ROUTE — 신규추천이 이 ROUTE면 교체 추천 금지
NEW_ALLOWED_ROUTES = {"ATTACK", "ARMED", "NEUTRAL"}

# [v3.9.21c 평가 3] ROUTE 한글/대소문자 정규화
# 추천 CSV / UI 표시에서 다양한 표기 가능
ROUTE_ALIAS = {
    "ATTACK": "ATTACK", "적극매수": "ATTACK", "공격": "ATTACK",
    "ARMED": "ARMED", "진입대기": "ARMED", "관심": "ARMED",
    "NEUTRAL": "NEUTRAL", "중립": "NEUTRAL",
    "WAIT": "WAIT", "대기": "WAIT",
    "OVERHEAT": "OVERHEAT", "과열": "OVERHEAT",
    "EXIT": "EXIT", "EXIT_WARNING": "EXIT_WARNING", "이탈경고": "EXIT_WARNING",
}


def _normalize_route(raw: Optional[str]) -> Optional[str]:
    """[v3.9.21c 평가 3] ROUTE 정규화 — strip + upper + alias 매핑.

    Args:
        raw: 원본 ROUTE 문자열 ("attack", " ARMED ", "진입대기" 등)

    Returns:
        정규화된 ROUTE ("ATTACK"/"ARMED"/"NEUTRAL"/"WAIT"/"OVERHEAT"/"EXIT_WARNING"/"EXIT")
        또는 None (빈 문자열)
        매핑되지 않으면 upper만 적용해 반환 (알 수 없는 값은 그대로 유지)
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # 영문은 upper, 한글은 그대로
    key = s.upper()
    if key in ROUTE_ALIAS:
        return ROUTE_ALIAS[key]
    # 한글 키 시도 (upper로 바뀌지 않은 원본)
    if s in ROUTE_ALIAS:
        return ROUTE_ALIAS[s]
    return key  # 알 수 없는 ROUTE는 upper만


def analyze_portfolio_swap(
    holdings: list,
    recommend_df: pd.DataFrame,
    total_value: float = 0.0,
) -> dict:
    """[v3.9.21] 보유종목 vs 신규추천 교체 판단.

    Args:
        holdings: [{"name": str, "avg": int, "qty": int}, ...]
                  보유종목 (current price는 recommend_df의 종가에서 lookup)
        recommend_df: 오늘 recommend CSV (DISPLAY_SCORE, ROUTE, EBS, RR 등)
        total_value: 총 평가금액 (비중 계산용). 0이면 holdings에서 자동 계산

    Returns:
        dict {
            "summary": {"green": N, "blue": N, ..., "white": N},
            "top_pick": dict 또는 None (오늘 Top Pick — 교체 대상 후보 1순위),
            "new_recommend_safe": bool (신규추천이 anomaly/과열 없는지),
            "holdings_analysis": [
                {
                    "name": str,
                    "avg": int, "qty": int,
                    "current_price": int 또는 None,
                    "value": float,
                    "pnl_pct": float,
                    "weight_pct": float,
                    "matched_in_recommend": bool,
                    "final_score": float 또는 None,
                    "route": str 또는 None,
                    "ebs": int 또는 None,
                    "rr_now_tp1": float 또는 None,
                    "entry_gap_pct": float 또는 None,
                    "verdict": dict (icon/level/title/reasons/swap_candidate)
                },
                ...
            ]
        }
    """
    if recommend_df is None or recommend_df.empty:
        return {
            "summary": {},
            "top_pick": None,
            "new_recommend_safe": False,
            "holdings_analysis": [],
            "error": "recommend 데이터 없음",
        }
    if not holdings:
        return {
            "summary": {},
            "top_pick": None,
            "new_recommend_safe": False,
            "holdings_analysis": [],
            "error": "보유종목 없음",
        }

    # ─── 1. 신규추천 안전성 평가 (Top Pick anomaly 검사) ───
    top_pick = _select_top_pick(recommend_df)
    new_recommend_safe = _is_recommend_safe(top_pick)

    # ─── 2. 각 보유종목에 현재가 매칭 (1차 — value 산출) ───
    # [v3.9.21c 평가 4] 비중 2-pass 계산
    # 1차: 모든 보유종목에 recommend에서 현재가 매칭
    # 2차: 매칭된 current_price * qty 합계로 total_value 재계산
    #      (분자/분모 통일 — 분자는 현재가, 분모도 현재가 기준)
    # 3차: weight_pct 산출 후 verdict
    matched_picks = []  # [(hold, pick or None, current_price), ...]
    for hold in holdings:
        matched_df = _match_holding_to_recommend(hold, recommend_df)
        if matched_df.empty:
            pick = None
            current_price = None
        else:
            pick = _row_to_pick_dict(matched_df.iloc[0])
            current_price = pick.get("close_price")
            if current_price is None:
                current_price = hold["avg"]  # fallback
        matched_picks.append((hold, pick, current_price))

    # 2차: total_value 재계산
    # - 외부에서 명시적으로 주어진 total_value > 0이면 신뢰
    # - 아니면 current_price 기반 합계 (매칭 안 된 종목은 avg 사용)
    value_basis = "current_price"
    if total_value == 0:
        total_value = sum(
            (cp if cp is not None else h["avg"]) * h["qty"]
            for h, _, cp in matched_picks
        )
        if total_value == 0:
            total_value = 1  # 0 division 방지
        # 일부 종목이 매칭 안 됐으면 mixed basis
        unmatched = sum(1 for _, p, _ in matched_picks if p is None)
        if unmatched > 0:
            value_basis = "mixed_current_avg"

    # 3차: weight_pct 산출 + verdict
    analysis = []
    for hold, pick, current_price in matched_picks:
        item = _analyze_with_matched_pick(
            hold, pick, current_price, total_value,
            top_pick, new_recommend_safe,
        )
        analysis.append(item)

    # ─── 3. 요약 집계 ───
    summary = {"green": 0, "blue": 0, "yellow": 0, "orange": 0,
               "red": 0, "white": 0}
    for item in analysis:
        level = item["verdict"]["level"]
        if level in summary:
            summary[level] += 1

    return {
        "summary": summary,
        "top_pick": top_pick,
        "new_recommend_safe": new_recommend_safe,
        "holdings_analysis": analysis,
        "total_value": total_value,
        # [v3.9.21c 평가 4] 비중 계산 근거 — UI에서 사용자에게 표시
        "value_basis": value_basis,
    }


def _select_top_pick(recommend_df: pd.DataFrame) -> Optional[dict]:
    """오늘 Top Pick — TOP_PICK=1 또는 DISPLAY_SCORE 최고."""
    if recommend_df.empty:
        return None
    # TOP_PICK 컬럼이 있고 1인 행 우선
    if "TOP_PICK" in recommend_df.columns:
        tp_rows = recommend_df[
            pd.to_numeric(recommend_df["TOP_PICK"], errors="coerce") == 1
        ]
        if not tp_rows.empty:
            return _row_to_pick_dict(tp_rows.iloc[0])
    # fallback: DISPLAY_SCORE 최고
    score_col = None
    for c in ["DISPLAY_SCORE", "FINAL_SCORE"]:
        if c in recommend_df.columns:
            score_col = c
            break
    if score_col is None:
        return None
    top_idx = pd.to_numeric(
        recommend_df[score_col], errors="coerce"
    ).idxmax()
    return _row_to_pick_dict(recommend_df.loc[top_idx])


def _row_to_pick_dict(row: pd.Series) -> dict:
    """recommend 행 → pick dict."""
    def _get(field, default=None, num=True):
        v = row.get(field, default)
        if num and v is not None and not pd.isna(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return default
        return v if not pd.isna(v) else default

    def _get_first(fields, default=None, num=True):
        """[v3.9.21d 평가 3] alias 컬럼 순회 — 첫 번째 유효 값 반환.

        예: ENTRY_GAP_PCT 없으면 GAP_PCT → entry_gap → ENTRY_GAP 순으로 탐색.
        """
        for f in fields:
            v = _get(f, default=None, num=num)
            if v is not None:
                return v
        return default

    # [v3.9.21b] anomaly 정보도 함께 추출
    # recommend CSV에 IS_ANOMALY / ANOMALY_FLAG / TP_SATURATION 컬럼이 있을 수 있음
    # [v3.9.21c 평가 2] list/tuple 판단을 pd.isna보다 먼저 — array truth ambiguity 방지
    is_anomaly = False
    for col in ("IS_ANOMALY", "ANOMALY_FLAG", "anomaly_flags"):
        v = row.get(col)
        if v is None:
            continue
        # list/tuple는 pd.isna() 호출 전에 처리 (배열 반환 → truth ambiguity)
        if isinstance(v, (list, tuple)):
            if len(v) > 0:
                is_anomaly = True
                break
            continue
        # scalar에서만 pd.isna 호출
        try:
            if pd.isna(v):
                continue
        except Exception as e:
            _logger.debug(f"[swap] pd.isna 검사 실패: {e}")
            pass
        if isinstance(v, str):
            if v.strip() and v.strip().lower() not in ("0", "false", "none", "[]"):
                is_anomaly = True
                break
        else:
            try:
                if bool(v):
                    is_anomaly = True
                    break
            except Exception as e:
                _logger.debug(f"[swap] anomaly bool 변환 실패: {e}")
                pass

    # [v3.9.21d 평가 3] alias fallback — 다양한 컬럼명 안전 처리
    tp_saturation = _get_first(("TP_SATURATION", "TP_포화율", "tp_saturation"))

    # [v3.9.21b 평가 2] EBS 파서 — 다양한 형식 안전 처리
    ebs_val = _parse_ebs(row)

    return {
        "name": str(row.get("종목명", "")),
        "code": str(row.get("종목코드", "")),
        "final_score": _get_first(("FINAL_SCORE", "DISPLAY_SCORE", "final_score")),
        "display_score": _get("DISPLAY_SCORE"),
        "elite_score": _get_first(("ELITE_SCORE", "elite_score")),
        "route": _normalize_route(row.get("ROUTE")),
        "ebs": ebs_val,
        "rr_now_tp1": _get_first(("RR_NOW_TP1", "RR_TP1", "rr_now_tp1", "rr")),
        # [v3.9.21d 평가 3] entry_gap alias 확장
        "entry_gap_pct": _get_first((
            "ENTRY_GAP_PCT", "GAP_PCT", "ENTRY_GAP",
            "ENTRY_GAP_TO_BUY", "entry_gap_pct", "gap_pct",
        )),
        "close_price": _get_first(("종가", "close", "close_price")),
        "macro_risk": str(row.get("MACRO_RISK", "")) or None,
        # [v3.9.21b 평가 1] anomaly 정보
        "is_anomaly": is_anomaly,
        "tp_saturation": tp_saturation,
        # 비현실 수익률도 anomaly로 분류 (있으면)
        "total_return_abs": _get_first((
            "TOTAL_RETURN", "total_return", "TOTAL_RET", "total_ret",
        )),
    }


def _parse_ebs(row) -> int:
    """[v3.9.21b 평가 2] EBS 다양한 형식 안전 파싱.

    지원 형식:
    - 숫자: 0, 1, 8 → 그대로
    - 문자열: "PASS"/"FAIL", "8/8 (PASS)", "8/8" → 변환
    - bool: True/False → 1/0
    - EBS_PASS 컬럼: 별도 boolean 컬럼

    Returns:
        int — EBS 점수 (0=FAIL, ≥1=PASS)
    """
    # EBS_PASS boolean 컬럼 먼저 확인
    if "EBS_PASS" in row:
        v = row["EBS_PASS"]
        if v is not None and not pd.isna(v):
            # [v3.9.21d 평가 1] 문자열 "False"/"0"/"FAIL"이 bool("False")=True로
            # 오해석되는 버그 차단 — 명시 화이트리스트 우선
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("1", "true", "yes", "y", "pass", "t"):
                    return 1
                if s in ("0", "false", "no", "n", "fail", "none", "", "f"):
                    return 0
                # 알 수 없는 문자열은 fallthrough — bool 처리에 맡김
                # 단 빈 문자열이 아니라면 bool로는 True가 됨 — 보수적으로 0
                _logger.debug(
                    f"[swap] EBS_PASS 문자열 인식 실패: {v!r} → 0 fallback"
                )
                return 0
            # 비문자열 (bool/int/float) 처리
            try:
                return 1 if bool(v) else 0
            except Exception as e:
                _logger.debug(f"[swap] EBS_PASS bool 변환 실패: {e}")
                pass

    # EBS / EBS_SCORE / EBS_COUNT 순서로 탐색
    for col in ("EBS", "EBS_SCORE", "EBS_COUNT"):
        if col not in row:
            continue
        v = row.get(col)
        if v is None or pd.isna(v):
            continue

        # 숫자형 직접 변환 가능
        try:
            num = float(v)
            return int(num)
        except (ValueError, TypeError) as e:
            _logger.debug(f"[swap] EBS 숫자 변환 실패 (문자열로 시도): {e}")
            pass

        # 문자열 파싱
        s = str(v).upper().strip()
        if "PASS" in s:
            return 1
        if "FAIL" in s:
            return 0
        if "/" in s:
            # "8/8" 또는 "8/8 (PASS)"
            head = s.split("/")[0].strip()
            try:
                return int(float(head))
            except (ValueError, TypeError) as e:
                _logger.debug(f"[swap] EBS fraction 변환 실패: {e}")
                pass

    return 0


def _is_recommend_safe(top_pick: Optional[dict]) -> bool:
    """[v3.9.21b 평가 1] 신규추천이 anomaly/과열 아닌지 — 교체 추천 가능 여부.

    Returns False if:
    - top_pick이 None
    - is_anomaly=True (anomaly_flags 존재)
    - tp_saturation ≥ 80% (TP 포화 과다)
    - total_return ≥ 300% (비현실 수익률 — 백테스트 anomaly와 동기)
    - ROUTE가 OVERHEAT
    - ROUTE가 NEW_ALLOWED_ROUTES 밖
    - ENTRY_GAP > 5% (과열)
    """
    if top_pick is None:
        return False

    # [v3.9.21b 평가 1] 진짜 anomaly 차단 — IS_ANOMALY/anomaly_flags 존재
    if top_pick.get("is_anomaly"):
        return False

    # TP 포화 과다 — 80% 이상이면 신규 진입 신호 의심
    tp_sat = top_pick.get("tp_saturation")
    if tp_sat is not None and tp_sat >= NEW_DANGER_TP_SATURATION:
        return False

    # 비현실 수익률 차단 (300% 초과 — backtest_policy의 ANOMALY_TOTAL_RET_ABS와 동기)
    total_ret = top_pick.get("total_return_abs")
    if total_ret is not None and abs(total_ret) > NEW_DANGER_TOTAL_RET_ABS:
        return False

    route = top_pick.get("route", "")
    if route in DANGER_ROUTES:
        return False
    if route and route not in NEW_ALLOWED_ROUTES and route not in DANGER_ROUTES:
        # 알 수 없는 ROUTE는 보수적으로 False
        # 단 빈 문자열/None은 허용 (정보 부족이지 위험은 아님)
        return False

    entry_gap = top_pick.get("entry_gap_pct")
    if entry_gap is not None and entry_gap > NEW_DANGER_ENTRY_GAP_PCT:
        return False

    return True


def _analyze_with_matched_pick(
    hold: dict,
    pick: Optional[dict],
    current_price: Optional[float],
    total_value: float,
    top_pick: Optional[dict],
    new_recommend_safe: bool,
) -> dict:
    """[v3.9.21c] 이미 매칭된 pick + current_price로 단일 보유종목 분석.

    2-pass 비중 계산을 위해 _analyze_single_holding을 분해:
    - 매칭 로직은 analyze_portfolio_swap 안에서 1차에 끝남
    - 이 함수는 매칭 결과 받아서 pnl/weight/verdict만 산출
    """
    avg = int(hold["avg"])
    qty = int(hold["qty"])

    if pick is None or current_price is None:
        # 보유종목이 오늘 추천 목록에 없음
        return _build_holding_item(
            hold, current_price=None, total_value=total_value,
            pick=None, verdict=_verdict_white(
                "오늘 추천에 없음 — 현재 SwingPicker 신호 산출 안 됨"
            ),
        )

    value = current_price * qty
    pnl_pct = ((current_price - avg) / avg * 100) if avg > 0 else 0.0
    weight_pct = (value / total_value * 100) if total_value > 0 else 0.0

    verdict = _derive_holding_verdict(
        pick=pick,
        pnl_pct=pnl_pct,
        weight_pct=weight_pct,
        top_pick=top_pick,
        new_recommend_safe=new_recommend_safe,
    )

    return _build_holding_item(
        hold, current_price, total_value, pick, verdict,
        pnl_pct=pnl_pct, weight_pct=weight_pct,
    )


def _analyze_single_holding(
    hold: dict,
    recommend_df: pd.DataFrame,
    top_pick: Optional[dict],
    total_value: float,
    new_recommend_safe: bool,
) -> dict:
    """단일 보유종목 분석 (단독 호출용 — analyze_portfolio_swap는 2-pass 사용).

    backward compat 유지 — 외부에서 직접 호출하는 경우.
    """
    matched = _match_holding_to_recommend(hold, recommend_df)
    if matched.empty:
        pick = None
        current_price = None
    else:
        pick = _row_to_pick_dict(matched.iloc[0])
        current_price = pick.get("close_price")
        if current_price is None:
            current_price = hold["avg"]
    return _analyze_with_matched_pick(
        hold, pick, current_price, total_value,
        top_pick, new_recommend_safe,
    )


def _match_holding_to_recommend(
    hold: dict,
    recommend_df: pd.DataFrame,
) -> pd.DataFrame:
    """[v3.9.21b 평가 3] 보유종목 → recommend DataFrame row 매칭.

    매칭 우선순위:
    1. 종목코드 (6자리 zero-padded) — 가장 안전
    2. 종목명 strip (공백 제거 비교)

    Args:
        hold: {"name": str, "code": str (optional), "avg": int, "qty": int}
        recommend_df: 종목코드 / 종목명 컬럼 보유

    Returns:
        매칭된 row(들) DataFrame (있으면 1개, 없으면 empty)
    """
    # 1순위: 종목코드 매칭 (6자리)
    code = str(hold.get("code", "") or "").strip()
    if code and "종목코드" in recommend_df.columns:
        code_padded = code.zfill(6)
        matched = recommend_df[
            recommend_df["종목코드"].astype(str).str.strip().str.zfill(6)
            == code_padded
        ]
        if not matched.empty:
            return matched

    # 2순위: 종목명 strip 비교
    name = str(hold.get("name", "") or "").strip()
    if not name or "종목명" not in recommend_df.columns:
        return recommend_df.iloc[0:0]  # empty

    return recommend_df[
        recommend_df["종목명"].astype(str).str.strip() == name
    ]


def _build_holding_item(
    hold: dict,
    current_price: Optional[float],
    total_value: float,
    pick: Optional[dict],
    verdict: dict,
    pnl_pct: float = 0.0,
    weight_pct: float = 0.0,
) -> dict:
    """holding 분석 결과 dict 빌드."""
    value = (current_price * hold["qty"]) if current_price else 0.0
    item = {
        "name": hold["name"],
        "avg": hold["avg"],
        "qty": hold["qty"],
        "current_price": current_price,
        "value": value,
        "pnl_pct": pnl_pct,
        "weight_pct": weight_pct,
        "matched_in_recommend": pick is not None,
        "verdict": verdict,
    }
    if pick is not None:
        item.update({
            "final_score": pick.get("final_score"),
            "display_score": pick.get("display_score"),
            "route": pick.get("route"),
            "ebs": pick.get("ebs"),
            "rr_now_tp1": pick.get("rr_now_tp1"),
            "entry_gap_pct": pick.get("entry_gap_pct"),
        })
    else:
        item.update({
            "final_score": None, "display_score": None, "route": None,
            "ebs": None, "rr_now_tp1": None, "entry_gap_pct": None,
        })
    return item


def derive_holding_verdict(
    pick: Optional[dict],
    pnl_pct: float,
    weight_pct: float,
    top_pick: Optional[dict],
    new_recommend_safe: bool,
) -> dict:
    """[v3.9.21] 보유종목 1개 판정 — 6단계.

    공개 API (테스트용 — 외부에서 직접 호출 가능).

    🔴 정리 우선 : EBS 0/FAIL + ROUTE WAIT/EXIT + 손실
    🟠 교체 후보 : 보유 약함 AND 신규추천 강함 (안전한 경우)
    🟡 감량 검토 : 비중 과다 OR 보유 약함 (신규 부재)
    🔵 유지 + 신규금지 : 보유 OK but 비중/신호 모호
    🟢 유지     : 점수+ROUTE+비중 모두 양호
    ⚪ 데이터 부족: pick 없음
    """
    return _derive_holding_verdict(
        pick, pnl_pct, weight_pct, top_pick, new_recommend_safe
    )


def _derive_holding_verdict(
    pick: Optional[dict],
    pnl_pct: float,
    weight_pct: float,
    top_pick: Optional[dict],
    new_recommend_safe: bool,
) -> dict:
    """보유종목 판정 — 우선순위 기반 6단계."""
    if pick is None:
        return _verdict_white("recommend 데이터에 없음")

    final = pick.get("final_score") or 0
    route = pick.get("route", "") or ""
    ebs = int(pick.get("ebs") or 0)
    rr = pick.get("rr_now_tp1")
    entry_gap = pick.get("entry_gap_pct")

    # ─── 🔴 정리 우선 (최우선 차단) ───
    # EBS 0/FAIL + ROUTE WAIT/EXIT + 손실 (≥ 2개 조건)
    danger_signals = []
    if ebs == 0:
        danger_signals.append("EBS 0")
    if route in LIQUIDATE_ROUTES:
        danger_signals.append(f"ROUTE {route}")
    if pnl_pct <= HOLD_WEAK_LOSS_PCT:
        danger_signals.append(f"손익 {pnl_pct:+.1f}%")

    if len(danger_signals) >= 2:
        return {
            "icon": "🔴",
            "level": "red",
            "title": "정리 우선",
            "color_class": "text-red-400",
            "reasons": danger_signals,
            "swap_candidate": new_recommend_safe and top_pick is not None,
            "body": (
                f"위험 신호 {len(danger_signals)}개 동시 발생: "
                f"{', '.join(danger_signals)}. 손실 확대 전 정리 검토 권장."
            ),
        }

    # ─── 🟠 교체 후보 ───
    # 조건: 보유 약함 AND 신규추천 강함 AND 안전
    is_hold_weak = (
        final < HOLD_WEAK_FINAL_SCORE
        or ebs == 0
        or route in LIQUIDATE_ROUTES
    )
    is_new_strong = (
        top_pick is not None
        and (top_pick.get("final_score") or 0) >= NEW_STRONG_FINAL_SCORE
        and int(top_pick.get("ebs") or 0) >= 1
        and (top_pick.get("rr_now_tp1") or 0) >= NEW_STRONG_RR
    )
    # 신규추천이 anomaly/과열이 아니어야 교체 추천 가능
    if is_hold_weak and is_new_strong and new_recommend_safe:
        score_gap = (top_pick.get("final_score") or 0) - final
        if score_gap >= SWAP_MIN_SCORE_GAP:
            reasons = [
                f"보유 FINAL {final:.0f} (기준 {HOLD_WEAK_FINAL_SCORE:.0f})",
                f"신규 {top_pick['name']} FINAL {top_pick['final_score']:.0f}",
                f"점수차 {score_gap:+.0f}",
            ]
            return {
                "icon": "🟠",
                "level": "orange",
                "title": "교체 후보",
                "color_class": "text-orange-400",
                "reasons": reasons,
                "swap_candidate": True,
                "body": (
                    f"보유종목 약함 + 신규추천 {top_pick['name']}이(가) 강함 "
                    f"(FINAL {top_pick['final_score']:.0f}, ROUTE "
                    f"{top_pick.get('route', '?')}). 교체 가능성 검토. "
                    "(자동 실행 아님 — 사용자 최종 확인 필수)"
                ),
            }

    # ─── 🟡 감량 검토 ───
    # 비중 과다 OR 보유 약함 (교체 후보 조건 미충족)
    if weight_pct >= HOLD_OVER_CONCENTRATION_PCT:
        return {
            "icon": "🟡",
            "level": "yellow",
            "title": "감량 검토",
            "color_class": "text-yellow-400",
            "reasons": [f"비중 {weight_pct:.0f}% (기준 {HOLD_OVER_CONCENTRATION_PCT:.0f}%)"],
            "swap_candidate": False,
            "body": (
                f"단일 종목 비중이 {weight_pct:.0f}%로 과집중. "
                "수익 중이어도 분산 차원에서 일부 감량 검토 권장."
            ),
        }
    if is_hold_weak:
        reasons = []
        if final < HOLD_WEAK_FINAL_SCORE:
            reasons.append(f"FINAL {final:.0f}")
        if ebs == 0:
            reasons.append("EBS 0")
        if route in LIQUIDATE_ROUTES:
            reasons.append(f"ROUTE {route}")
        return {
            "icon": "🟡",
            "level": "yellow",
            "title": "감량 검토",
            "color_class": "text-yellow-400",
            "reasons": reasons,
            "swap_candidate": False,
            "body": (
                f"보유종목 신호 약화 ({', '.join(reasons)}) — 단 신규추천도 "
                "강하지 않아 바로 교체하기보다 감량 후 관찰 권장."
            ),
        }

    # ─── 🔵 유지 + 신규매수 금지 ───
    # 비중 적정 + 보유 OK but 추매 신호 약함
    if route not in ("ATTACK", "ARMED") and rr is not None and rr < 1.2:
        return {
            "icon": "🔵",
            "level": "blue",
            "title": "유지 · 신규매수 금지",
            "color_class": "text-blue-400",
            "reasons": [
                f"ROUTE {route or '?'}", f"RR {rr:.2f}" if rr else "RR ?"
            ],
            "swap_candidate": False,
            "body": (
                f"보유는 유지 가능. 다만 ROUTE/RR이 추매 신호 부족 — "
                "현재 비중 유지하고 추가 매수는 자제 권장."
            ),
        }

    # ─── 🟢 유지 (모든 조건 양호) ───
    reasons = [
        f"FINAL {final:.0f}",
        f"ROUTE {route or '?'}",
        f"손익 {pnl_pct:+.1f}%",
    ]
    return {
        "icon": "🟢",
        "level": "green",
        "title": "유지",
        "color_class": "text-emerald-400",
        "reasons": reasons,
        "swap_candidate": False,
        "body": (
            "보유 지속 권장. SwingPicker 신호와 비중 모두 안정 — "
            "현재 포지션 유지."
        ),
    }


def _verdict_white(reason: str) -> dict:
    """⚪ 데이터 부족 verdict."""
    return {
        "icon": "⚪",
        "level": "white",
        "title": "데이터 부족",
        "color_class": "text-gray-400",
        "reasons": [reason],
        "swap_candidate": False,
        "body": reason,
    }
