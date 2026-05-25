# -*- coding: utf-8 -*-
"""
daily_briefing.py — 매일 자동 브리핑 생성기
═══════════════════════════════════════════════
[Rule]
  1. collector Action 완료 후 실행
  2. ATTACK / ARMED 종목만 필터
  3. 목표가 달성(CLOSED_TP) 종목 제외
  4. DISPLAY_SCORE 상위 3종목 선정
  5. 토스/블로그/텔레그램 배포 가능한 마크다운 생성

출력:
  - data/briefing_{YYYYMMDD}.md   (일자별 아카이브)
  - data/briefing_latest.md       (최신 고정)
  - data/briefing_{YYYYMMDD}.json (구조화 데이터 — API/웹용)
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
#  1. 목표가 달성 종목 필터
# ═══════════════════════════════════════════════════

def _load_closed_tp_codes(out_dir: str) -> set:
    """positions.json에서 목표가 도달(CLOSED_TP) 종목코드 세트 반환"""
    pos_path = os.path.join(out_dir, "positions.json")
    if not os.path.exists(pos_path):
        return set()
    try:
        with open(pos_path, "r", encoding="utf-8") as f:
            positions = json.load(f)
        # 최근 30일 내 CLOSED_TP 종목만
        codes = set()
        for p in positions:
            if p.get("status") == "CLOSED_TP":
                codes.add(str(p.get("code", "")).zfill(6))
        return codes
    except Exception as e:
        logger.warning(f"positions.json 파싱 실패: {e}")
        return set()


def _load_closed_tp_from_log(out_dir: str, lookback_days: int = 14) -> set:
    """per_trade_log.csv에서 최근 N일 내 익절 종목 제외"""
    log_path = os.path.join(out_dir, "per_trade_log.csv")
    if not os.path.exists(log_path):
        return set()
    try:
        df = pd.read_csv(log_path, dtype={"code": str})
        if "exit_type" in df.columns and "exit_ymd" in df.columns:
            df["exit_ymd"] = pd.to_datetime(df["exit_ymd"], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
            recent_tp = df[(df["exit_type"] == "TP") & (df["exit_ymd"] >= cutoff)]
            return set(recent_tp["code"].astype(str).str.zfill(6))
    except Exception as e:
        logger.warning(f"per_trade_log 파싱 실패: {e}")
    return set()


# ═══════════════════════════════════════════════════
#  2. 상위 3종목 선정
# ═══════════════════════════════════════════════════

def select_top3(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    ATTACK/ARMED 종목 중 목표가 미달성 상위 3종목 선정

    Returns: DataFrame (최대 3행)
    """
    # 1) ATTACK / ARMED만
    df = df.copy()
    df["ROUTE"] = df["ROUTE"].astype(str).str.strip().str.upper()
    active = df[df["ROUTE"].isin(["ATTACK", "ARMED"])].copy()

    if active.empty:
        logger.info("📝 브리핑: ATTACK/ARMED 종목 없음")
        return pd.DataFrame()

    # 2) 목표가 달성 종목 제외
    tp_codes = _load_closed_tp_codes(out_dir) | _load_closed_tp_from_log(out_dir)
    if tp_codes:
        active["종목코드"] = active["종목코드"].astype(str).str.zfill(6)
        before = len(active)
        active = active[~active["종목코드"].isin(tp_codes)]
        excluded = before - len(active)
        if excluded > 0:
            logger.info(f"📝 브리핑: 목표가 달성 {excluded}건 제외")

    if active.empty:
        logger.info("📝 브리핑: 목표가 제외 후 ATTACK/ARMED 종목 없음")
        return pd.DataFrame()

    # 3) DISPLAY_SCORE 상위 3종목
    active["DISPLAY_SCORE"] = pd.to_numeric(active["DISPLAY_SCORE"], errors="coerce").fillna(0)
    top3 = active.nlargest(3, "DISPLAY_SCORE")

    return top3


# ═══════════════════════════════════════════════════
#  3. 마크다운 생성 (토스/블로그 배포용)
# ═══════════════════════════════════════════════════

def _safe_int(val) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _safe_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _route_emoji(route: str) -> str:
    return {"ATTACK": "🚀", "ARMED": "🔫"}.get(route, "👀")


def _route_kr(route: str) -> str:
    return {"ATTACK": "매수 돌입", "ARMED": "매수 대기"}.get(route, route)


def generate_briefing_md(top3: pd.DataFrame, trade_ymd: str, site_url: str = "https://ldyprotrader.com") -> str:
    """토스/블로그 배포용 마크다운 생성"""

    date_display = f"{trade_ymd[:4]}.{trade_ymd[4:6]}.{trade_ymd[6:]}"
    lines = []

    # 헤더
    lines.append(f"🎯 SwingPicker AI 오늘의 Top 3 ({date_display})")
    lines.append("")
    lines.append("AI가 107종목을 분석해서 뽑은 오늘의 핵심 종목입니다.")
    lines.append("")

    # 각 종목
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        code = str(row.get("종목코드", "")).zfill(6)
        name = str(row.get("종목명", code))
        route = str(row.get("ROUTE", "")).upper()
        score = _safe_float(row.get("DISPLAY_SCORE", 0))
        close = _safe_int(row.get("종가", 0))
        entry = _safe_int(row.get("추천매수가", 0))
        stop = _safe_int(row.get("손절가", 0))
        t1 = _safe_int(row.get("추천매도가1", 0))
        est_wr = _safe_float(row.get("EST_WIN_RATE", 0))

        emoji = _route_emoji(route)
        wr_pct = est_wr * 100 if est_wr <= 1 else est_wr

        # 손절/익절 퍼센트
        stop_pct = (stop / entry - 1) * 100 if entry > 0 and stop > 0 else 0
        t1_pct = (t1 / entry - 1) * 100 if entry > 0 and t1 > 0 else 0
        risk = entry - stop if entry > 0 and stop > 0 else 1
        rr = (t1 - entry) / risk if risk > 0 and t1 > 0 else 0

        lines.append(f"{'─' * 30}")
        lines.append(f"{emoji} #{rank}. {name} ({code})")
        lines.append(f"AI 점수: {score:.0f}점 | 신호: {_route_kr(route)} | 승률: {wr_pct:.0f}%")
        lines.append("")

        if entry > 0:
            lines.append(f"  현재가: {close:,}원")
            lines.append(f"  매수가: {entry:,}원")
            lines.append(f"  손절가: {stop:,}원 ({stop_pct:+.1f}%)")
            if t1 > 0:
                lines.append(f"  목표가: {t1:,}원 ({t1_pct:+.1f}%) → 손익비 {rr:.1f}:1")
            lines.append("")

        # 분석 링크
        lines.append(f"  📊 상세 분석 → {site_url}/stock/{code}")
        lines.append("")

    # 푸터
    lines.append(f"{'─' * 30}")
    lines.append("")
    lines.append(f"🔗 전체 107종목 분석: {site_url}")
    lines.append("")
    lines.append("⚠️ 본 자료는 AI 분석 참고 자료이며 투자 권유가 아닙니다.")
    lines.append("투자 판단은 본인 책임이며, 손실이 발생할 수 있습니다.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════
#  4. JSON 구조화 데이터 (웹/API용)
# ═══════════════════════════════════════════════════

def generate_briefing_json(top3: pd.DataFrame, trade_ymd: str, site_url: str = "https://ldyprotrader.com") -> dict:
    """웹 표시/API 응답용 구조화 데이터"""
    stocks = []
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        code = str(row.get("종목코드", "")).zfill(6)
        entry = _safe_int(row.get("추천매수가", 0))
        stop = _safe_int(row.get("손절가", 0))
        t1 = _safe_int(row.get("추천매도가1", 0))
        risk = entry - stop if entry > 0 and stop > 0 else 1

        stocks.append({
            "rank": rank,
            "code": code,
            "name": str(row.get("종목명", code)),
            "route": str(row.get("ROUTE", "")),
            "score": round(_safe_float(row.get("DISPLAY_SCORE", 0)), 1),
            "close": _safe_int(row.get("종가", 0)),
            "entry": entry,
            "stop": stop,
            "target1": t1,
            "target2": _safe_int(row.get("추천매도가2", 0)),
            "est_win_rate": round(_safe_float(row.get("EST_WIN_RATE", 0)), 3),
            "rr": round((t1 - entry) / risk, 1) if risk > 0 and t1 > 0 else 0,
            "sector": str(row.get("업종_대분류", "")),
            "market": str(row.get("시장", "")),
            "url": f"{site_url}/stock/{code}",
        })

    return {
        "trade_date": trade_ymd,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(stocks),
        "stocks": stocks,
    }


# ═══════════════════════════════════════════════════
#  5. 메인 실행 (collector Step 12에서 호출)
# ═══════════════════════════════════════════════════

def generate_daily_briefing(
    out_dir: str,
    trade_ymd: str,
    df: Optional[pd.DataFrame] = None,
    site_url: str = "https://ldyprotrader.com",
) -> Dict:
    """
    매일 자동 브리핑 생성 — collector 파이프라인 Step 12

    Args:
        out_dir: data/ 디렉토리
        trade_ymd: 거래 기준일 (YYYYMMDD)
        df: recommend DataFrame (None이면 CSV에서 로드)
        site_url: 사이트 URL

    Returns:
        {"count": int, "codes": list, "md_path": str, "json_path": str}
    """
    # 데이터 로드
    if df is None:
        csv_path = os.path.join(out_dir, "recommend_latest.csv")
        if not os.path.exists(csv_path):
            logger.warning("❌ recommend_latest.csv 없음 — 브리핑 스킵")
            return {"count": 0, "codes": [], "md_path": "", "json_path": ""}
        df = pd.read_csv(csv_path, dtype={"종목코드": str})

    # 상위 3종목 선정
    top3 = select_top3(df, out_dir)
    if top3.empty:
        logger.info("📝 브리핑 대상 없음 (ATTACK/ARMED 0건)")
        return {"count": 0, "codes": [], "md_path": "", "json_path": ""}

    codes = top3["종목코드"].astype(str).str.zfill(6).tolist()
    names = top3["종목명"].tolist()

    # 마크다운 생성
    md_content = generate_briefing_md(top3, trade_ymd, site_url)
    md_dated = os.path.join(out_dir, f"briefing_{trade_ymd}.md")
    md_latest = os.path.join(out_dir, "briefing_latest.md")

    with open(md_dated, "w", encoding="utf-8") as f:
        f.write(md_content)
    with open(md_latest, "w", encoding="utf-8") as f:
        f.write(md_content)

    # JSON 생성
    json_data = generate_briefing_json(top3, trade_ymd, site_url)
    json_dated = os.path.join(out_dir, f"briefing_{trade_ymd}.json")
    json_latest = os.path.join(out_dir, "briefing_latest.json")

    for p in [json_dated, json_latest]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"📝 일일 브리핑 생성: {len(top3)}종목 [{', '.join(names)}]")
    logger.info(f"   → {md_dated}")

    return {
        "count": len(top3),
        "codes": codes,
        "names": names,
        "md_path": md_dated,
        "json_path": json_dated,
    }


# ═══════════════════════════════════════════════════
#  [v22] monotonicity_report + CI HARD/SOFT Gate
#  ─ 참조: SwingPicker_v22_Final_Consolidated.md §3
# ═══════════════════════════════════════════════════

def _load_json_safe(path: str) -> Optional[dict]:
    """JSON 파일 안전 로드 — 미존재/파싱 실패 시 None"""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _compute_monotonicity(winrate_table: list) -> dict:
    """winrate_table bin별 p_win / avg_ret_excess 단조성 판정.
    
    단조성: 점수 ↑ → 성과 ↑ 이상적. Wilson LCB로 표본 흔들림 방어.
    """
    try:
        from shared_utils import wilson_lcb
    except ImportError:
        return {"valid": False, "reason": "wilson_lcb 미탑재"}
    
    if not winrate_table or len(winrate_table) < 2:
        return {"valid": False, "reason": "bin 2개 미만"}
    
    # score_lo 오름차순 정렬 보장
    bins = sorted(winrate_table, key=lambda r: r.get("score_lo", 0))
    
    p_wins = []
    p_wins_lcb = []
    avg_rets = []
    avg_excess = []
    for b in bins:
        n = int(b.get("n_raw", 0))
        p = b.get("p_win")
        if n > 0 and p is not None:
            # Wilson LCB — raw p_win이 아닌 신뢰하한으로 단조성 판정
            wins = int(round(float(p) * n))
            p_wins_lcb.append(round(wilson_lcb(wins, n), 4))
            p_wins.append(round(float(p), 4))
            avg_rets.append(b.get("avg_ret_net_pct"))
            avg_excess.append(b.get("avg_ret_excess_pct"))
        else:
            p_wins_lcb.append(None)
            p_wins.append(None)
            avg_rets.append(None)
            avg_excess.append(None)
    
    # 단조성 체크 — None 제외, 유효값만 비교
    def _is_monotone_nondec(seq):
        vals = [v for v in seq if v is not None]
        if len(vals) < 2:
            return None   # 판정 불가
        return all(vals[i] <= vals[i+1] + 0.05 for i in range(len(vals)-1))
    
    return {
        "valid": True,
        "bins": [
            {
                "score_lo": b.get("score_lo"),
                "score_hi": b.get("score_hi"),
                "n_raw": int(b.get("n_raw", 0)),
                "p_win": p,
                "p_win_lcb": lcb,
                "avg_ret_net_pct": r,
                "avg_ret_excess_pct": e,
            }
            for b, p, lcb, r, e in zip(bins, p_wins, p_wins_lcb, avg_rets, avg_excess)
        ],
        "monotone_p_win_lcb": _is_monotone_nondec(p_wins_lcb),
        "monotone_avg_ret": _is_monotone_nondec(avg_rets),
        "monotone_avg_ret_excess": _is_monotone_nondec(avg_excess),
    }


def _avg_stable_7day(out_dir: str, asof_ymd: str) -> Optional[float]:
    """최근 7영업일 STABLE TOP_PICK 평균 개수 — top_pick_validation_*.json 집계.
    
    휴장일 파일 없으면 skip, 3개 이상 파일 있으면 평균, 그 미만은 None.
    """
    from glob import glob
    files = sorted(glob(os.path.join(out_dir, "top_pick_validation_2*.json")), reverse=True)[:7]
    if len(files) < 3:
        return None
    
    stable_counts = []
    for f in files:
        data = _load_json_safe(f)
        if not data:
            continue
        by_type = data.get("top_pick_by_type", {})
        if isinstance(by_type, dict):
            stable_counts.append(int(by_type.get("STABLE", 0)))
    
    if len(stable_counts) < 3:
        return None
    return round(sum(stable_counts) / len(stable_counts), 2)


def _realized_wr_for_score_range(
    wt_table: list,
    lo: float,
    hi: float,
) -> tuple:
    """[v22.3.2] 특정 ELITE_SCORE 범위의 historical realized win rate.

    declared와 같은 모집단(점수 범위)에서 realized를 계산해야 fair 비교.
    예: TOP_PICK이 ELITE_SCORE 79.4~79.7 → [70-80) bin만 사용
        (전체 bin 가중평균을 쓰던 기존 방식은 영구 양수 gap 유발)

    sufficient=false bin은 제외 — winrate_table.meta.min_n 정책 존중.

    Args:
        wt_table: winrate_table["table"] (list of bin dicts)
        lo: 점수 하한 (포함)
        hi: 점수 상한 (미포함)

    Returns:
        (realized_wr, n_total)
        - realized_wr: 가중평균 승률 (전체 bin sufficient=false면 None)
        - n_total: 매칭된 effective n (HARD 게이트의 표본 가드용)
    """
    total_n = 0
    w_sum = 0.0
    try:
        for b in wt_table:
            b_lo = float(b.get("score_lo", -1))
            b_hi = float(b.get("score_hi", -1))
            # bin이 [lo, hi) 와 겹치는지
            if b_hi <= lo or b_lo >= hi:
                continue
            n = int(b.get("n_raw", 0))
            p = b.get("p_win")
            if n > 0 and p is not None and b.get("sufficient"):
                total_n += n
                w_sum += n * float(p)
    except Exception:
        return None, 0
    if total_n > 0:
        return round(w_sum / total_n, 4), total_n
    return None, 0


def generate_monotonicity_report(out_dir: str, asof_ymd: str) -> dict:
    """[v22] 일일 단조성/커버리지/갭 리포트.
    
    출력:
      - data/monotonicity_report_{YYYYMMDD}.json
      - data/monotonicity_report_latest.json
    
    포함 필드:
      - market_map_coverage (+ benchmark_mapping_warning)
      - ELITE_SCORE bin별 Wilson LCB + 단조성 판정
      - avg_ret_excess_pct 집계
      - declared_vs_realized_gap (선언 vs 실현 승률 차이)
      - top_pick_funnel (오늘)
      - top_pick_stable_7day_avg
      - ci_checks (HARD/SOFT 게이트 결과)
    """
    try:
        from shared_utils import compute_market_map_coverage
    except ImportError:
        compute_market_map_coverage = None
    
    report = {
        "asof_ymd": asof_ymd,
        "generated_at": datetime.now().isoformat(),
    }
    
    # 1. recommend_latest → market map coverage + 선언 승률
    rec_path = os.path.join(out_dir, "recommend_latest.csv")
    declared_wr_active = None
    declared_wr_top_pick = None
    tp1_neg_count = 0
    top_pick_wait_count = 0
    top_pick_count = 0
    # [v22.3.1] RR<1.0 검증 변수 초기화
    top_pick_rr_lt1_count = 0
    top_pick_min_rr = None
    
    if os.path.exists(rec_path):
        try:
            rec = pd.read_csv(rec_path, dtype={"종목코드": str})
            # market coverage
            if compute_market_map_coverage is not None:
                codes = rec["종목코드"] if "종목코드" in rec.columns else []
                coverage = compute_market_map_coverage(codes)
                report.update(coverage)
            
            # 선언 승률 — Active(ATTACK/ARMED) 평균
            if "EST_WIN_RATE" in rec.columns and "ROUTE" in rec.columns:
                active = rec[rec["ROUTE"].astype(str).isin(["ATTACK", "ARMED"])]
                if len(active) > 0:
                    _wr_a = active["EST_WIN_RATE"].dropna()
                    if len(_wr_a) > 0:
                        declared_wr_active = round(float(_wr_a.mean()), 4)
            
            # [v22 v4] 선언 승률 — TOP_PICK 평균 (HARD gate에 더 적합)
            if "EST_WIN_RATE" in rec.columns and "TOP_PICK" in rec.columns:
                tp_for_wr = rec[rec["TOP_PICK"].astype(int) == 1]
                if len(tp_for_wr) > 0:
                    _wr_t = tp_for_wr["EST_WIN_RATE"].dropna()
                    if len(_wr_t) > 0:
                        declared_wr_top_pick = round(float(_wr_t.mean()), 4)
            
            # TOP_PICK HARD 검증
            if "TOP_PICK" in rec.columns:
                tp = rec[rec["TOP_PICK"].astype(int) == 1]
                top_pick_count = len(tp)
                if "ROUTE" in tp.columns:
                    top_pick_wait_count = int(
                        (~tp["ROUTE"].astype(str).isin(["ATTACK", "ARMED"])).sum()
                    )
                if "TP1_PCT" in tp.columns:
                    tp1_neg_count = int(
                        (pd.to_numeric(tp["TP1_PCT"], errors="coerce") <= 0).sum()
                    )
                # [v22.3.1] RR_NOW_TP1 < 1.0 hard 검증 — scoring_engine v22.3과 일관성
                # 평가 피드백: "로직은 막았는데 리포트가 그 조건을 감시하지 않음"
                if "RR_NOW_TP1" in tp.columns:
                    top_pick_rr_lt1_count = int(
                        (pd.to_numeric(tp["RR_NOW_TP1"], errors="coerce").fillna(0) < 1.0).sum()
                    )
                    top_pick_min_rr = float(
                        pd.to_numeric(tp["RR_NOW_TP1"], errors="coerce").fillna(0).min()
                    ) if len(tp) > 0 else None
                else:
                    top_pick_rr_lt1_count = 0
                    top_pick_min_rr = None
        except Exception as e:
            report["recommend_parse_error"] = str(e)
    
    # 두 기준 모두 report에 기록
    report["declared_wr_active"] = declared_wr_active
    report["declared_wr_top_pick"] = declared_wr_top_pick
    
    # 2. winrate_table_by_ELITE_SCORE_latest → bin 단조성
    wt_elite = _load_json_safe(os.path.join(out_dir, "winrate_table_by_ELITE_SCORE_latest.json"))
    if wt_elite and "table" in wt_elite:
        mono = _compute_monotonicity(wt_elite["table"])
        report["elite_monotonicity"] = mono
        
        # ─────────────────────────────────────────────────────────
        # [v22.3.2] 모집단 일치 realized — declared와 같은 점수 범위에서만 계산
        # 기존 버그: declared_top_pick(상위 점수 평균)과 realized(전체 bin 가중)
        #          비교 → 모델이 정확해도 영구 양수 gap → HARD FAIL
        # 수정: TOP_PICK / active의 ELITE_SCORE 범위로 매칭 bin만 가중평균
        # ─────────────────────────────────────────────────────────
        table = wt_elite["table"]
        realized_wr_top_pick = None
        realized_wr_top_pick_n = 0
        realized_wr_active = None
        realized_wr_active_n = 0
        try:
            if os.path.exists(rec_path):
                rec_for_match = pd.read_csv(rec_path, dtype={"종목코드": str})
                if "ELITE_SCORE" in rec_for_match.columns:
                    if "TOP_PICK" in rec_for_match.columns:
                        tp_for_range = rec_for_match[
                            rec_for_match["TOP_PICK"].astype(int) == 1
                        ]
                        if len(tp_for_range) > 0:
                            tp_lo = float(tp_for_range["ELITE_SCORE"].min())
                            tp_hi = float(tp_for_range["ELITE_SCORE"].max()) + 0.01
                            realized_wr_top_pick, realized_wr_top_pick_n = (
                                _realized_wr_for_score_range(table, tp_lo, tp_hi)
                            )
                    if "ROUTE" in rec_for_match.columns:
                        act_for_range = rec_for_match[
                            rec_for_match["ROUTE"].astype(str).isin(["ATTACK", "ARMED"])
                        ]
                        if len(act_for_range) > 0:
                            a_lo = float(act_for_range["ELITE_SCORE"].min())
                            a_hi = float(act_for_range["ELITE_SCORE"].max()) + 0.01
                            realized_wr_active, realized_wr_active_n = (
                                _realized_wr_for_score_range(table, a_lo, a_hi)
                            )
        except Exception as e:
            report["matched_realized_calc_error"] = str(e)
        
        # 호환용 — 전체 bin 가중평균 (기존 키 유지)
        realized_wr = None
        total_n = 0
        w_sum = 0.0
        for b in wt_elite["table"]:
            n = int(b.get("n_raw", 0))
            p = b.get("p_win")
            if n > 0 and p is not None and b.get("sufficient"):
                total_n += n
                w_sum += n * float(p)
        if total_n > 0:
            realized_wr = round(w_sum / total_n, 4)
        report["realized_wr"] = realized_wr

        # [v22.3.10b] 레거시/테스트 CSV 호환:
        # recommend_latest.csv에 ELITE_SCORE가 없으면 같은 점수구간 매칭이 불가능하다.
        # 이 경우 기존 호환 방식인 전체 sufficient bin 가중 realized_wr로 fallback한다.
        # 실데이터에 ELITE_SCORE가 있으면 기존 matched-population 로직이 그대로 우선된다.
        if realized_wr_top_pick is None and declared_wr_top_pick is not None and realized_wr is not None:
            realized_wr_top_pick = realized_wr
            realized_wr_top_pick_n = total_n
            report["realized_wr_top_pick_fallback"] = "overall_no_elite_score"
        if realized_wr_active is None and declared_wr_active is not None and realized_wr is not None:
            realized_wr_active = realized_wr
            realized_wr_active_n = total_n
            report["realized_wr_active_fallback"] = "overall_no_elite_score"
        
        # [v22.3.2] 모집단 일치 realized 신규 키
        report["realized_wr_top_pick"] = realized_wr_top_pick
        report["realized_wr_top_pick_n"] = realized_wr_top_pick_n
        report["realized_wr_active"] = realized_wr_active
        report["realized_wr_active_n"] = realized_wr_active_n
        
        # 3. 선언 vs 실현 갭 — [v22.3.2] 같은 모집단끼리만 비교
        gap_active = None
        gap_top_pick = None
        if realized_wr_top_pick is not None and declared_wr_top_pick is not None:
            gap_top_pick = round(declared_wr_top_pick - realized_wr_top_pick, 4)
        if realized_wr_active is not None and declared_wr_active is not None:
            gap_active = round(declared_wr_active - realized_wr_active, 4)
        report["declared_vs_realized_gap_active"] = gap_active
        report["declared_vs_realized_gap_top_pick"] = gap_top_pick
        # 호환용 alias — TOP_PICK 우선, 없으면 active
        report["declared_vs_realized_gap"] = (
            gap_top_pick if gap_top_pick is not None else gap_active
        )
        
        # avg_ret_excess 집계 (전체 bin 가중평균) — 기존 그대로
        total_n2 = 0
        e_sum = 0.0
        for b in wt_elite["table"]:
            n = int(b.get("n_raw", 0))
            e = b.get("avg_ret_excess_pct")
            if n > 0 and e is not None:
                total_n2 += n
                e_sum += n * float(e)
        if total_n2 > 0:
            report["avg_ret_excess_pct"] = round(e_sum / total_n2, 4)
    else:
        report["elite_monotonicity"] = {"valid": False, "reason": "winrate_table_by_ELITE_SCORE 없음"}
    
    # 4. top_pick_funnel (오늘)
    funnel = _load_json_safe(os.path.join(out_dir, f"top_pick_funnel_{asof_ymd}.json"))
    if funnel:
        report["top_pick_funnel"] = funnel
    
    # 5. STABLE 7일 평균
    report["top_pick_stable_7day_avg"] = _avg_stable_7day(out_dir, asof_ymd)
    
    # 6. CI Gate 검증
    ci_hard = []
    ci_soft = []
    
    # HARD 1: TOP_PICK에 WAIT/NEUTRAL 없음
    if top_pick_wait_count > 0:
        ci_hard.append({
            "gate": "top_pick_route_positive",
            "status": "FAIL",
            "detail": f"TOP_PICK {top_pick_wait_count}건이 ATTACK/ARMED 아님",
        })
    else:
        ci_hard.append({"gate": "top_pick_route_positive", "status": "PASS"})
    
    # HARD 2: TOP_PICK인데 TP1_PCT <= 0 없음
    if tp1_neg_count > 0:
        ci_hard.append({
            "gate": "top_pick_tp1_positive",
            "status": "FAIL",
            "detail": f"TOP_PICK {tp1_neg_count}건이 TP1_PCT <= 0",
        })
    else:
        ci_hard.append({"gate": "top_pick_tp1_positive", "status": "PASS"})

    # [v22.3.1] HARD 2.5: TOP_PICK RR_NOW_TP1 >= 1.0
    # scoring_engine v22.3 hard_gate와 일관성. 운영 리포트 차원에서 다시 한번 차단.
    if top_pick_rr_lt1_count > 0:
        ci_hard.append({
            "gate": "top_pick_rr_now_tp1_1",
            "status": "FAIL",
            "detail": f"TOP_PICK {top_pick_rr_lt1_count}건이 RR_NOW_TP1 < 1.0 "
                      f"(min={top_pick_min_rr:.2f})" if top_pick_min_rr is not None
                      else f"TOP_PICK {top_pick_rr_lt1_count}건이 RR_NOW_TP1 < 1.0",
        })
    else:
        _detail = f"min_rr={top_pick_min_rr:.2f}" if top_pick_min_rr is not None else "TOP_PICK 0건"
        ci_hard.append({
            "gate": "top_pick_rr_now_tp1_1",
            "status": "PASS",
            "detail": _detail,
        })
    
    # HARD 3: 선언-실현 갭 15%p 이하 — [v22.3.2-A] TOP_PICK 전용
    # ─────────────────────────────────────────────────────────
    # active fallback 제거 이유:
    #   active 모집단(ATTACK/ARMED ~수십~수백 종목)은 EST_WIN_RATE가 거의 fallback
    #   상수(~0.539)에 가까워서, 매칭 realized와의 gap이 "모델 over-confidence"가
    #   아니라 "fallback 정책의 보정값"을 측정하게 됨. HARD gate 본래 의도(개별
    #   예측 calibration 검증)와 어긋나므로 active는 SOFT WARN 모니터링으로 강등.
    # ─────────────────────────────────────────────────────────
    MIN_N_FOR_HARD_GAP = 30
    gap_top_pick = report.get("declared_vs_realized_gap_top_pick")
    gap_top_pick_n = report.get("realized_wr_top_pick_n", 0) or 0

    if gap_top_pick is None:
        ci_hard.append({"gate": "declared_vs_realized_gap_15pp", "status": "SKIP",
                        "detail": "TOP_PICK 매칭 sufficient bin 없음"})
    elif gap_top_pick_n < MIN_N_FOR_HARD_GAP:
        ci_hard.append({"gate": "declared_vs_realized_gap_15pp", "status": "SKIP",
                        "detail": f"TOP_PICK matched n={gap_top_pick_n} < {MIN_N_FOR_HARD_GAP}"})
        if abs(gap_top_pick) > 0.15:
            ci_soft.append({
                "gate": "declared_vs_realized_gap_top_pick_small_n",
                "status": "WARN",
                "detail": f"gap_top_pick={gap_top_pick:.1%} (n={gap_top_pick_n}, 표본 부족 → 추세 모니터링)",
            })
    elif gap_top_pick > 0.15:
        ci_hard.append({
            "gate": "declared_vs_realized_gap_15pp",
            "status": "FAIL",
            "detail": f"gap_top_pick={gap_top_pick:.1%} > 15%p (n={gap_top_pick_n})",
        })
    else:
        ci_hard.append({"gate": "declared_vs_realized_gap_15pp", "status": "PASS",
                        "detail": f"gap_top_pick={gap_top_pick:.1%} (n={gap_top_pick_n})"})

    # SOFT (신규) — active gap은 모니터링 신호로 분리
    gap_active = report.get("declared_vs_realized_gap_active")
    gap_active_n = report.get("realized_wr_active_n", 0) or 0
    if gap_active is None:
        ci_soft.append({"gate": "declared_vs_realized_gap_active", "status": "OK",
                        "detail": "active 매칭 sufficient bin 없음"})
    elif gap_active_n < MIN_N_FOR_HARD_GAP:
        ci_soft.append({"gate": "declared_vs_realized_gap_active", "status": "OK",
                        "detail": f"n={gap_active_n} 표본 부족"})
    elif abs(gap_active) > 0.15:
        ci_soft.append({
            "gate": "declared_vs_realized_gap_active",
            "status": "WARN",
            "detail": f"gap_active={gap_active:.1%} (n={gap_active_n}) — fallback 보정 또는 calibration drift 모니터링",
        })
    else:
        ci_soft.append({"gate": "declared_vs_realized_gap_active", "status": "OK",
                        "detail": f"gap_active={gap_active:.1%} (n={gap_active_n})"})
    
    # SOFT 1: Wilson LCB 단조성
    mono = report.get("elite_monotonicity", {})
    if mono.get("valid") and mono.get("monotone_p_win_lcb") is False:
        ci_soft.append({"gate": "wilson_monotonicity",
                        "status": "WARN",
                        "detail": "ELITE_SCORE bin별 p_win LCB 단조성 깨짐"})
    else:
        ci_soft.append({"gate": "wilson_monotonicity", "status": "OK"})
    
    # SOFT 2: avg_ret_excess 음수
    excess = report.get("avg_ret_excess_pct")
    if excess is not None and excess < 0:
        ci_soft.append({"gate": "avg_ret_excess_positive",
                        "status": "WARN",
                        "detail": f"avg_ret_excess={excess:.2f}%"})
    elif excess is not None:
        ci_soft.append({"gate": "avg_ret_excess_positive", "status": "OK",
                        "detail": f"{excess:.2f}%"})
    
    # SOFT 3: market_map_coverage 낮음
    cov = report.get("market_map_coverage", 1.0)
    if cov < 0.95:
        ci_soft.append({"gate": "market_map_coverage_95",
                        "status": "WARN",
                        "detail": f"coverage={cov:.1%}"})
    else:
        ci_soft.append({"gate": "market_map_coverage_95", "status": "OK"})
    
    # SOFT 4: STABLE 7일 평균 부족
    stable_avg = report.get("top_pick_stable_7day_avg")
    if stable_avg is not None and stable_avg < 1.0:
        ci_soft.append({"gate": "stable_7day_avg_1",
                        "status": "WARN",
                        "detail": f"평균 {stable_avg}개"})
    elif stable_avg is not None:
        ci_soft.append({"gate": "stable_7day_avg_1", "status": "OK",
                        "detail": f"평균 {stable_avg}개"})
    
    report["ci_hard"] = ci_hard
    report["ci_soft"] = ci_soft
    report["ci_hard_all_pass"] = all(c["status"] in ("PASS", "SKIP") for c in ci_hard)
    report["ci_soft_any_warn"] = any(c["status"] == "WARN" for c in ci_soft)
    report["top_pick_count"] = top_pick_count
    # [v22.3.1] RR 검증 메트릭 — 평가 피드백: "min_rr 추가"
    report["top_pick_rr_lt1_count"] = top_pick_rr_lt1_count
    report["top_pick_min_rr"] = round(top_pick_min_rr, 2) if top_pick_min_rr is not None else None
    
    # 저장
    dated = os.path.join(out_dir, f"monotonicity_report_{asof_ymd}.json")
    latest = os.path.join(out_dir, "monotonicity_report_latest.json")
    for p in [dated, latest]:
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"monotonicity_report 저장 실패: {e}")
    
    logger.info(
        f"📊 [v22] monotonicity_report: "
        f"HARD={'PASS' if report['ci_hard_all_pass'] else 'FAIL'}, "
        f"SOFT={'WARN' if report['ci_soft_any_warn'] else 'OK'}, "
        f"gap={report.get('declared_vs_realized_gap')}, "
        f"excess={excess}, coverage={cov}"
    )
    
    return report


# ═══════════════════════════════════════════════════
#  standalone 실행
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    _dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    _ymd = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y%m%d")
    logging.basicConfig(level=logging.INFO)
    result = generate_daily_briefing(_dir, _ymd)
    if result["count"] > 0:
        print(f"\n✅ 브리핑 생성 완료: {result['count']}종목")
        print(f"   📄 {result['md_path']}")
        with open(result["md_path"], "r", encoding="utf-8") as f:
            print(f"\n{f.read()}")
    else:
        print("❌ 브리핑 대상 종목 없음")
    
    # [v22] monotonicity_report 별도 생성 (브리핑 대상 유무와 무관)
    print("\n" + "=" * 50)
    print("📊 [v22] monotonicity_report 생성 중...")
    mono = generate_monotonicity_report(_dir, _ymd)
    print(f"   HARD: {'✅ PASS' if mono['ci_hard_all_pass'] else '❌ FAIL'}")
    print(f"   SOFT: {'⚠️ WARN' if mono['ci_soft_any_warn'] else '✅ OK'}")
    if mono.get("declared_vs_realized_gap") is not None:
        print(f"   선언-실현 갭: {mono['declared_vs_realized_gap']:.1%}")
    if mono.get("avg_ret_excess_pct") is not None:
        print(f"   avg_ret_excess: {mono['avg_ret_excess_pct']:+.2f}%")
    print(f"   → {os.path.join(_dir, f'monotonicity_report_{_ymd}.json')}")
