# -*- coding: utf-8 -*-
"""
run_health.py — 파이프라인 실행 건강 상태 진단
═══════════════════════════════════════════════════
[v19.2] Degraded Run을 숨기지 않고 명시적으로 표시
[v20.6.3] macro_risk/market_breadth 직접 저장 → Dashboard 추론 불필요

사용법:
    from run_health import RunHealth, check_run_health

    health = check_run_health(df_out, mcap_map, bench_map, inv_maps)
    # health.status = "OK" | "DEGRADED" | "CRITICAL"
    # health.reasons = ["MCAP_EMPTY", "BENCH_FAIL", ...]
    # health.inject_columns(df_out)  → CSV에 RUN_STATUS 등 주입
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RunHealth:
    """파이프라인 실행 건강 상태"""
    status: str = "OK"                        # OK / DEGRADED / CRITICAL
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_freshness_ok: bool = True
    checks: Dict[str, bool] = field(default_factory=dict)
    confidence_score: float = 100.0           # [v20.0] 0~100 신뢰도
    # [v20.6] macro 직접 저장 — Dashboard fallback 추론 불필요
    macro_risk: str = "NORMAL"
    market_breadth: float = 50.0
    # [v20.7] 축별 상태 메타 — silent fallback 가시화
    axis_status: Dict[str, str] = field(default_factory=lambda: {
        "MCAP": "UNKNOWN", "BENCH": "UNKNOWN", "FLOW": "UNKNOWN",
        "NEWS": "UNKNOWN", "SECTOR": "UNKNOWN", "ML": "UNKNOWN",
        "TRIGGER": "UNKNOWN",
    })
    # 축 상태 값: OK | PARTIAL | FALLBACK_USED | DISABLED | FAILED_ZERO_FILLED | UNKNOWN
    fallback_count: int = 0

    # 축별 신뢰 가중치 (결손 시 감점)
    _AXIS_WEIGHTS = {
        "MCAP_EMPTY": 20, "MCAP_ALL_ZERO": 20,
        "BENCH_FAIL": 15, "BENCH_NAN": 15,
        "FLOW_ZERO": 15,      # 수급 전무 (외인+기관+개인 모두 0)
        "FLOW_PARTIAL": 8,    # 수급 부분 결손 (외인/기관 없고 개인만 있음)
        "NEWS_OFF": 10,
        "SECTOR_FAIL": 10,
    }

    def add_issue(self, code: str, severity: str = "DEGRADED"):
        """이슈 등록. severity = DEGRADED | CRITICAL"""
        self.reasons.append(code)
        if severity == "CRITICAL":
            self.status = "CRITICAL"
        elif self.status != "CRITICAL":
            self.status = "DEGRADED"
        self.checks[code] = False
        # [v20.0] 신뢰도 감점
        self.confidence_score = max(0, self.confidence_score - self._AXIS_WEIGHTS.get(code, 5))

    def add_ok(self, code: str):
        """정상 확인 등록"""
        self.checks[code] = True

    @property
    def max_allowed_route(self) -> str:
        """[v20.0.2] RUN_STATUS + 신뢰도 기반 최대 허용 ROUTE

        핵심 규칙:
          - CRITICAL → 무조건 WAIT
          - DEGRADED → 최대 ARMED (ATTACK 금지)
          - OK + 신뢰도 70+ → ATTACK 허용
          - OK + 신뢰도 40~69 → ARMED까지
          - OK + 신뢰도 <40 → WAIT만
        """
        if self.status == "CRITICAL":
            return "WAIT"
        if self.status == "DEGRADED":
            # 결손 축 3개 이상이면 WAIT까지만
            if len(self.reasons) >= 3 and self.confidence_score < 50:
                return "WAIT"
            return "ARMED"       # DEGRADED면 ATTACK 절대 금지
        # OK 상태
        if self.confidence_score >= 70:
            return "ATTACK"
        elif self.confidence_score >= 40:
            return "ARMED"
        else:
            return "WAIT"

    def set_axis(self, axis: str, status: str):
        """[v20.7] 축별 상태 설정."""
        self.axis_status[axis] = status
        if status in ("FALLBACK_USED", "FAILED_ZERO_FILLED"):
            self.fallback_count += 1

    def inject_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """CSV에 건강 상태 컬럼 주입"""
        df["RUN_STATUS"] = self.status
        df["DEGRADED_REASONS"] = "|".join(self.reasons) if self.reasons else ""
        df["DATA_FRESHNESS_OK"] = self.data_freshness_ok
        df["CONFIDENCE_SCORE"] = self.confidence_score
        df["MAX_ALLOWED_ROUTE"] = self.max_allowed_route

        # [v20.7] 축별 상태
        for axis, st in self.axis_status.items():
            df[f"AXIS_{axis}"] = st
        df["FALLBACK_COUNT"] = self.fallback_count
        df["AXIS_QUALITY"] = sum(1 for v in self.axis_status.values() if v == "OK")

        # 데이터 유무 플래그
        df["HAS_NEWS"] = df.get("NEWS_SCORE", pd.Series(0, index=df.index)).fillna(0).astype(float).ne(0).any()
        df["HAS_FLOW"] = (
            df.get("외인순매수", pd.Series(0, index=df.index)).fillna(0).astype(float).ne(0).any()
            or df.get("기관순매수", pd.Series(0, index=df.index)).fillna(0).astype(float).ne(0).any()
        )
        df["HAS_SECTOR"] = df.get("SECTOR_RANK", pd.Series(dtype=float)).notna().any()

        return df

    def summary(self) -> str:
        """사람이 읽기 좋은 요약"""
        emoji = {"OK": "🟢", "DEGRADED": "🟡", "CRITICAL": "🔴"}.get(self.status, "⚪")
        lines = [f"{emoji} Run Status: {self.status} (신뢰도: {self.confidence_score:.0f}/100, 최대허용: {self.max_allowed_route})"]
        if self.reasons:
            lines.append(f"   Issues: {', '.join(self.reasons)}")
        if self.warnings:
            lines.append(f"   Warnings: {', '.join(self.warnings)}")
        for k, v in self.checks.items():
            mark = "✅" if v else "❌"
            lines.append(f"   {mark} {k}")
        return "\n".join(lines)


def check_run_health(
    df: pd.DataFrame,
    mcap_map: Optional[Dict] = None,
    bench_map: Optional[Dict] = None,
    inv_maps: Optional[Dict] = None,
    trade_ymd: str = "",
) -> RunHealth:
    """
    파이프라인 결과물 건강 진단

    Args:
        df: 최종 추천 DataFrame
        mcap_map: 시가총액 맵
        bench_map: 벤치마크 수익률 맵
        inv_maps: 수급 데이터 맵
        trade_ymd: 기준일

    Returns:
        RunHealth 인스턴스
    """
    h = RunHealth()

    # ── 1. 시가총액 ──
    if mcap_map is None or len(mcap_map) == 0:
        h.add_issue("MCAP_EMPTY")
        h.set_axis("MCAP", "FAILED_ZERO_FILLED")
        logger.warning("⚠️ [Health] 시가총액 맵 비어있음 → 시총 기반 손절 캡 비활성")
    elif "시가총액(억원)" in df.columns:
        mcap_col = pd.to_numeric(df["시가총액(억원)"], errors="coerce").fillna(0)
        if (mcap_col == 0).all():
            h.add_issue("MCAP_ALL_ZERO")
            h.set_axis("MCAP", "FAILED_ZERO_FILLED")
            logger.warning("⚠️ [Health] 시가총액 전행 0 → 폴백 필요")
        else:
            h.add_ok("MCAP")
            h.set_axis("MCAP", "OK")

    # ── 2. 벤치마크 ──
    if bench_map is None or len(bench_map) == 0:
        h.add_issue("BENCH_FAIL")
        h.set_axis("BENCH", "FAILED_ZERO_FILLED")
        logger.warning("⚠️ [Health] 벤치마크 맵 비어있음 → 상대강도 계산 불가")
    else:
        if "rel_60d_%" in df.columns and df["rel_60d_%"].isna().all():
            h.add_issue("BENCH_NAN")
            h.set_axis("BENCH", "PARTIAL")
        else:
            h.add_ok("BENCH")
            h.set_axis("BENCH", "OK")

    # ── 3. 수급 ──
    if inv_maps is None:
        h.add_issue("FLOW_ZERO")
        h.set_axis("FLOW", "FAILED_ZERO_FILLED")
    else:
        frg = inv_maps.get("frg", {})
        inst = inv_maps.get("inst", {})
        ant = inv_maps.get("ant", {})
        major_ok = len(frg) > 0 or len(inst) > 0
        if major_ok:
            h.add_ok("FLOW")
            h.set_axis("FLOW", "OK")
        elif len(ant) > 0:
            h.add_issue("FLOW_PARTIAL")
            h.set_axis("FLOW", "PARTIAL")
            logger.warning("⚠️ [Health] 외인/기관 수급 0건, 개인 수급만 있음 → 부분 보정")
        else:
            h.add_issue("FLOW_ZERO")
            h.set_axis("FLOW", "FAILED_ZERO_FILLED")
            logger.warning("⚠️ [Health] 수급 데이터 0건 → 수급 보정 비활성")

    # ── 4. 뉴스 ──
    if "NEWS_SCORE" in df.columns:
        ns = pd.to_numeric(df["NEWS_SCORE"], errors="coerce").fillna(0)
        if (ns == 0).all():
            h.add_issue("NEWS_OFF")
            h.set_axis("NEWS", "DISABLED")
        else:
            h.add_ok("NEWS")
            h.set_axis("NEWS", "OK")
    else:
        h.add_issue("NEWS_OFF")
        h.set_axis("NEWS", "DISABLED")

    # ── 5. 섹터 ──
    if "SECTOR_RANK" in df.columns:
        if df["SECTOR_RANK"].isna().all():
            h.add_issue("SECTOR_FAIL")
            h.set_axis("SECTOR", "FAILED_ZERO_FILLED")
        else:
            h.add_ok("SECTOR")
            h.set_axis("SECTOR", "OK")
    else:
        h.add_issue("SECTOR_FAIL")
        h.set_axis("SECTOR", "FAILED_ZERO_FILLED")

    # ── 5b. ML 축 상태 (ML_STATUS + AI_SCORE 통합 판정) ──
    if "ML_STATUS" in df.columns:
        # [v20.8] ML_STATUS가 있으면 직접 참조
        ml_st = df["ML_STATUS"].astype(str).str.strip()
        ml_ok_count = (ml_st == "OK").sum()
        ml_fail_statuses = ml_st[~ml_st.isin(["OK", "NO_DATA", "nan", ""])].unique()
        if ml_ok_count > 0:
            h.set_axis("ML", "OK")
        elif len(ml_fail_statuses) > 0:
            # 구체적 실패 사유 기록
            _ml_reason = str(ml_fail_statuses[0])
            h.set_axis("ML", f"FAILED:{_ml_reason}")
            h.warnings.append(f"ML_FAIL:{_ml_reason}")
        else:
            h.set_axis("ML", "DISABLED")
    elif "AI_SCORE" in df.columns:
        ai = pd.to_numeric(df["AI_SCORE"], errors="coerce").fillna(0)
        if (ai == 0).all():
            h.set_axis("ML", "DISABLED")
        else:
            h.set_axis("ML", "OK")
    else:
        h.set_axis("ML", "DISABLED")

    # ── 5c. Trigger 축 상태 ──
    if "TIMING_SCORE" in df.columns:
        ts = pd.to_numeric(df["TIMING_SCORE"], errors="coerce").fillna(0)
        if (ts == 0).all():
            h.set_axis("TRIGGER", "FAILED_ZERO_FILLED")
        else:
            h.set_axis("TRIGGER", "OK")

    # ── [v20.7] fallback 2개+ 시 MAX_ROUTE 자동 하향 ──
    _fb = sum(1 for v in h.axis_status.values() if v in ("FALLBACK_USED", "FAILED_ZERO_FILLED"))
    if _fb >= 3 and h.status == "OK":
        h.status = "DEGRADED"
        h.warnings.append(f"AUTO_DEGRADE: {_fb} axes failed/fallback")

    # ── 6. TP 단조성 검증 ──
    if "추천매도가1" in df.columns and "추천매도가2" in df.columns:
        tp1 = pd.to_numeric(df["추천매도가1"], errors="coerce").fillna(0)
        tp2 = pd.to_numeric(df["추천매도가2"], errors="coerce").fillna(0)
        violations = ((tp2 > 0) & (tp2 <= tp1)).sum()
        if violations > 0:
            h.add_issue(f"TP_MONO_FAIL_{violations}")
            logger.warning(f"⚠️ [Health] TP2 ≤ TP1 위반: {violations}건")
        else:
            h.add_ok("TP_MONOTONIC")

    # ── 7. 데이터 신선도 ──
    if trade_ymd and "기준일" in df.columns:
        basis = df["기준일"].dropna().unique()
        if len(basis) > 0:
            latest = str(max(basis))
            if latest < trade_ymd:
                h.data_freshness_ok = False
                h.warnings.append(f"DATA_STALE({latest}<{trade_ymd})")

    return h


def save_health(health: RunHealth, out_dir: str, trade_ymd: str) -> str:
    """건강 상태를 JSON으로 저장 (dated + latest 동시 저장)"""
    import json
    import os

    path = os.path.join(out_dir, f"run_health_{trade_ymd}.json")
    latest_path = os.path.join(out_dir, "run_health_latest.json")  # [v22.3] latest 추가
    data = {
        "trade_ymd": trade_ymd,  # [v22.3] latest 파일에서 어느 날짜인지 식별용
        "status": health.status,
        "reasons": health.reasons,
        "warnings": health.warnings,
        "data_freshness_ok": health.data_freshness_ok,
        "checks": health.checks,
        "confidence_score": health.confidence_score,
        "max_allowed_route": health.max_allowed_route,
        "macro_risk": health.macro_risk,
        "market_breadth": health.market_breadth,
        "axis_status": health.axis_status,
        "fallback_count": health.fallback_count,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"📊 [Health] {health.status} → {path}")
        # [v22.3] latest 갱신 — 운영 모니터링 안정성 (날짜 추측 불필요)
        try:
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ [Health] latest 저장 실패: {e}")
    except Exception as e:
        logger.warning(f"⚠️ [Health] 저장 실패: {e}")
    return path
