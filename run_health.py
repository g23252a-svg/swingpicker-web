# -*- coding: utf-8 -*-
"""
run_health.py — 파이프라인 실행 건강 상태 진단
═══════════════════════════════════════════════════
[v19.2] Degraded Run을 숨기지 않고 명시적으로 표시

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

    # 축별 신뢰 가중치 (결손 시 감점)
    _AXIS_WEIGHTS = {
        "MCAP_EMPTY": 20, "MCAP_ALL_ZERO": 20,
        "BENCH_FAIL": 15, "BENCH_NAN": 15,
        "FLOW_ZERO": 15,
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

    def inject_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """CSV에 건강 상태 컬럼 주입"""
        df["RUN_STATUS"] = self.status
        df["DEGRADED_REASONS"] = "|".join(self.reasons) if self.reasons else ""
        df["DATA_FRESHNESS_OK"] = self.data_freshness_ok
        df["CONFIDENCE_SCORE"] = self.confidence_score
        df["MAX_ALLOWED_ROUTE"] = self.max_allowed_route

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
        logger.warning("⚠️ [Health] 시가총액 맵 비어있음 → 시총 기반 손절 캡 비활성")
    elif "시가총액(억원)" in df.columns:
        mcap_col = pd.to_numeric(df["시가총액(억원)"], errors="coerce").fillna(0)
        if (mcap_col == 0).all():
            h.add_issue("MCAP_ALL_ZERO")
            logger.warning("⚠️ [Health] 시가총액 전행 0 → 폴백 필요")
        else:
            h.add_ok("MCAP")

    # ── 2. 벤치마크 ──
    if bench_map is None or len(bench_map) == 0:
        h.add_issue("BENCH_FAIL")
        logger.warning("⚠️ [Health] 벤치마크 맵 비어있음 → 상대강도 계산 불가")
    else:
        # rel_60d_% 전부 NaN인지 체크
        if "rel_60d_%" in df.columns and df["rel_60d_%"].isna().all():
            h.add_issue("BENCH_NAN")
        else:
            h.add_ok("BENCH")

    # ── 3. 수급 ──
    if inv_maps is None:
        h.add_issue("FLOW_ZERO")
    else:
        frg = inv_maps.get("frg", {})
        inst = inv_maps.get("inst", {})
        if len(frg) == 0 and len(inst) == 0:
            h.add_issue("FLOW_ZERO")
            logger.warning("⚠️ [Health] 수급 데이터 0건 → 수급 보정 비활성")
        else:
            h.add_ok("FLOW")

    # ── 4. 뉴스 ──
    if "NEWS_SCORE" in df.columns:
        ns = pd.to_numeric(df["NEWS_SCORE"], errors="coerce").fillna(0)
        if (ns == 0).all():
            h.add_issue("NEWS_OFF")
        else:
            h.add_ok("NEWS")
    else:
        h.add_issue("NEWS_OFF")

    # ── 5. 섹터 ──
    if "SECTOR_RANK" in df.columns:
        if df["SECTOR_RANK"].isna().all():
            h.add_issue("SECTOR_FAIL")
        else:
            h.add_ok("SECTOR")
    else:
        h.add_issue("SECTOR_FAIL")

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
    """건강 상태를 JSON으로 저장"""
    import json
    import os

    path = os.path.join(out_dir, f"run_health_{trade_ymd}.json")
    data = {
        "status": health.status,
        "reasons": health.reasons,
        "warnings": health.warnings,
        "data_freshness_ok": health.data_freshness_ok,
        "checks": health.checks,
        "confidence_score": health.confidence_score,
        "max_allowed_route": health.max_allowed_route,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"📊 [Health] {health.status} → {path}")
    except Exception as e:
        logger.warning(f"⚠️ [Health] 저장 실패: {e}")
    return path
