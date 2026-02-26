# -*- coding: utf-8 -*-
"""
portfolio_service.py — 포트폴리오 분석 서비스
═══════════════════════════════════════════════════
UI 의존성 0. 순수 데이터 계산 + Gist I/O만 담당.

사용처:
    from data.services.portfolio_service import PortfolioService
    svc = PortfolioService()
    holdings = svc.load_holdings()
    result = svc.analyze(holdings, scored_df)
"""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from config import settings
from data_source import KRXDataSource, get_data_source

_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  데이터 모델
# ═══════════════════════════════════════════════════

@dataclass
class Holding:
    """보유 종목 1건"""
    name: str
    code: str  # 6자리 종목코드 (현금이면 "CASH")
    avg_price: float  # 평단가
    qty: int  # 수량
    memo: str = ""


@dataclass
class HoldingResult:
    """분석 완료된 보유 종목"""
    name: str
    code: str
    avg_price: float
    qty: int
    current_price: int = 0
    buy_amount: float = 0.0
    eval_amount: float = 0.0
    pnl: float = 0.0  # 평가손익
    pnl_pct: float = 0.0  # 수익률 (%)
    weight_pct: float = 0.0  # 비중 (%)
    score: float = 0.0  # AI 종합 점수
    advice: str = "관망"
    advice_color: str = "gray"
    is_cash: bool = False


@dataclass
class PortfolioSummary:
    """포트폴리오 전체 요약"""
    total_eval: float = 0.0
    total_invest: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    cash_amount: float = 0.0
    cash_pct: float = 0.0
    stock_count: int = 0
    holdings: List[HoldingResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════
#  서비스
# ═══════════════════════════════════════════════════

class PortfolioService:
    """포트폴리오 분석 — UI 미포함, 순수 로직"""

    GIST_FILENAME = "my_portfolio.json"

    def __init__(self, data_source: Optional[KRXDataSource] = None):
        self._gist_token = settings.LDY_GIST_TOKEN or ""
        self._gist_id = settings.LDY_GIST_ID or ""
        self._local_file = "my_portfolio.json"
        self._ds = data_source or get_data_source()

    # ─── Gist / Local I/O ───

    def load_holdings_text(self) -> str:
        """포트폴리오 텍스트 로드 (Gist → Local fallback)"""
        if self._gist_token and self._gist_id:
            try:
                headers = {"Authorization": f"token {self._gist_token}"}
                r = requests.get(
                    f"https://api.github.com/gists/{self._gist_id}",
                    headers=headers, timeout=5,
                )
                if r.status_code == 200:
                    files = r.json().get("files", {})
                    if self.GIST_FILENAME in files:
                        content = files[self.GIST_FILENAME]["content"]
                        return json.loads(content).get("data", "")
            except Exception as e:
                _logger.warning(f"Gist 포트폴리오 로드 실패: {e}")

        if os.path.exists(self._local_file):
            try:
                with open(self._local_file, "r", encoding="utf-8") as f:
                    return json.load(f).get("data", "")
            except Exception:
                pass
        return ""

    def save_holdings_text(self, text: str) -> bool:
        """포트폴리오 텍스트 저장 — Gist 업로드는 Fire & Forget"""
        json_content = json.dumps({"data": text}, ensure_ascii=False)

        # 로컬 즉시 저장
        try:
            with open(self._local_file, "w", encoding="utf-8") as f:
                f.write(json_content)
        except Exception as e:
            _logger.warning(f"로컬 포트폴리오 저장 실패: {e}")

        # Gist 비동기 업로드
        if self._gist_token and self._gist_id:
            def _upload():
                try:
                    headers = {"Authorization": f"token {self._gist_token}"}
                    payload = {"files": {self.GIST_FILENAME: {"content": json_content}}}
                    requests.patch(
                        f"https://api.github.com/gists/{self._gist_id}",
                        headers=headers, json=payload, timeout=10,
                    )
                except Exception as e:
                    _logger.warning(f"Gist 포트폴리오 업로드 실패: {e}")

            threading.Thread(target=_upload, daemon=True, name="pf-gist-upload").start()

        return True

    # ─── 텍스트 파싱 ───

    def parse_holdings_text(self, text: str, code_map: Dict[str, str] = None) -> Tuple[List[Holding], float]:
        """'종목명:평단가:수량' 텍스트 → Holding 리스트 + 현금

        code_map: {종목명: 종목코드} 매핑 딕셔너리
        """
        holdings: List[Holding] = []
        cash = 0.0
        if not text:
            return holdings, cash

        if code_map is None:
            code_map = {}

        for line in text.strip().split("\n"):
            if ":" not in line:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            try:
                nm = parts[0].strip()
                price = int(float(parts[1].replace(",", "").strip()))
                qty = int(float(parts[2].replace(",", "").strip()))
            except (ValueError, TypeError):
                continue

            if nm.upper() == "CASH" or "현금" in nm:
                cash += price * qty
            else:
                code = code_map.get(nm, nm)
                holdings.append(Holding(name=nm, code=str(code).zfill(6), avg_price=price, qty=qty))

        return holdings, cash

    def holdings_to_text(self, rows: List[dict]) -> str:
        """UI 에디터 rows → 저장용 텍스트"""
        lines = []
        for r in rows:
            nm = str(r.get("종목명", "")).strip()
            if not nm:
                continue
            p = int(float(r.get("평단가", 0)))
            q = int(float(r.get("수량", 0)))
            lines.append(f"{nm}:{p}:{q}")
        return "\n".join(lines)

    # ─── 현재가 조회 (병렬) ───

    def fetch_current_price(self, code: str, name: str) -> Tuple[str, str, int]:
        """현재가 조회 — KRXDataSource 경유 (pykrx/FDR fallback 자동)"""
        price = 0
        try:
            end_ymd = datetime.now().strftime("%Y%m%d")
            start_ymd = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
            df = self._ds.get_ohlcv(str(code).zfill(6), start_ymd, end_ymd)
            if df is not None and not df.empty:
                col = "종가" if "종가" in df.columns else "Close"
                if col in df.columns:
                    price = int(df.iloc[-1][col])
        except Exception as e:
            _logger.debug(f"현재가 조회 실패 {code}: {e}")
        return code, name, price

    def fetch_prices_parallel(self, holdings: List[Holding], max_workers: int = 10) -> Dict[str, int]:
        """보유 종목 현재가 병렬 조회"""
        price_map: Dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(self.fetch_current_price, h.code, h.name): h
                for h in holdings if not (h.name.upper() == "CASH" or "현금" in h.name)
            }
            for fut in as_completed(futs):
                try:
                    code, name, price = fut.result()
                    price_map[code] = price
                except Exception:
                    pass
        return price_map

    # ─── 핵심: 포트폴리오 분석 ───

    def analyze(
        self,
        holdings: List[Holding],
        cash: float,
        price_map: Dict[str, int],
        scored_df: Optional[pd.DataFrame] = None,
    ) -> PortfolioSummary:
        """보유 종목 + 현재가 + AI 스코어 → 포트폴리오 진단 결과"""
        summary = PortfolioSummary()
        summary.cash_amount = cash

        if scored_df is None:
            scored_df = pd.DataFrame()

        results: List[HoldingResult] = []
        total_eval = 0.0
        total_buy = 0.0

        for h in holdings:
            curr = price_map.get(h.code, 0)
            buy_amt = h.avg_price * h.qty
            eval_amt = curr * h.qty
            total_eval += eval_amt
            total_buy += buy_amt

            pct = (curr - h.avg_price) / h.avg_price * 100 if h.avg_price > 0 and curr > 0 else 0.0

            # AI 점수 매핑
            score = self._match_score(h.code, h.name, scored_df)
            advice, advice_color = self._generate_advice(score, pct)

            results.append(HoldingResult(
                name=h.name,
                code=h.code,
                avg_price=h.avg_price,
                qty=h.qty,
                current_price=curr,
                buy_amount=buy_amt,
                eval_amount=eval_amt,
                pnl=eval_amt - buy_amt,
                pnl_pct=pct,
                score=score,
                advice=advice,
                advice_color=advice_color,
            ))

        # 전체 요약
        total_asset = total_eval + cash
        total_invest = total_buy + cash
        summary.total_eval = total_asset
        summary.total_invest = total_invest
        summary.total_pnl = total_asset - total_invest
        summary.total_pnl_pct = (total_asset - total_invest) / total_invest * 100 if total_invest > 0 else 0.0
        summary.cash_pct = cash / total_asset * 100 if total_asset > 0 else 0.0
        summary.stock_count = len(results)

        # 비중 계산
        for r in results:
            r.weight_pct = r.eval_amount / total_asset * 100 if total_asset > 0 else 0.0

        # 경고 생성
        summary.warnings = self._generate_warnings(results)
        summary.holdings = results
        return summary

    def _match_score(self, code: str, name: str, scored_df: pd.DataFrame) -> float:
        """scored_df에서 종목 점수 매칭 (코드 → 이름 fallback)"""
        if scored_df.empty:
            return 0.0

        code_6 = str(code).zfill(6)

        # 1차: 코드 매칭
        if "종목코드" in scored_df.columns:
            match = scored_df[scored_df["종목코드"].astype(str).str.zfill(6) == code_6]
            if not match.empty:
                return self._calc_final_score(match.iloc[0])

        # 2차: 이름 매칭
        if "종목명" in scored_df.columns:
            match = scored_df[scored_df["종목명"] == name]
            if not match.empty:
                return self._calc_final_score(match.iloc[0])

        return 0.0

    @staticmethod
    def _calc_final_score(row: pd.Series) -> float:
        """AI(60%) + 퀀트(40%) 종합 점수"""
        ml = float(row.get("ML_SCORE", 0) or 0)
        rank = float(row.get("RANK_SCORE", 0) or 0)
        display = float(row.get("DISPLAY_SCORE", 0) or 0)

        if ml > 0:
            return ml * 0.6 + rank * 0.4
        if display > 0:
            return display
        return rank

    @staticmethod
    def _generate_advice(score: float, pnl_pct: float) -> Tuple[str, str]:
        """점수 + 수익률 기반 AI 조언"""
        if score >= 80:
            return "💪강력홀딩/추매", "green"
        elif score >= 60:
            return "👌보유(양호)", "blue"
        elif score == 0:
            return "❓정보없음", "gray"
        elif score <= 40:
            return "⚠️교체권장/매도", "red"
        else:
            return "👀관망(중립)", "orange"

    @staticmethod
    def _generate_warnings(results: List[HoldingResult]) -> List[str]:
        """리스크 경고 메시지 생성"""
        warnings = []
        if not results:
            return warnings

        # 집중도 리스크
        max_r = max(results, key=lambda r: r.weight_pct)
        if max_r.weight_pct >= 40:
            warnings.append(
                f"⚠️ 집중 리스크: {max_r.name}이(가) 포트폴리오의 {max_r.weight_pct:.1f}%를 차지합니다."
            )

        # 손절 점검
        loss_names = [r.name for r in results if r.pnl_pct < -10]
        if loss_names:
            warnings.append(
                f"🔴 손절 점검 필요: {', '.join(loss_names)} (수익률 -10% 이하)"
            )

        # 교체 권장 종목
        bad_names = [r.name for r in results if 0 < r.score <= 40]
        if bad_names:
            warnings.append(
                f"🔄 교체 검토: {', '.join(bad_names)} (AI 점수 40점 이하)"
            )

        return warnings

    # ─── 교체 추천 ───

    @staticmethod
    def get_replacement_suggestion(
        scored_df: pd.DataFrame, exclude_codes: List[str] = None
    ) -> Optional[dict]:
        """교체 매매 추천 (scored_df 1위 종목)"""
        if scored_df.empty:
            return None
        if exclude_codes is None:
            exclude_codes = []

        # 보유 종목 제외
        if "종목코드" in scored_df.columns:
            mask = ~scored_df["종목코드"].astype(str).str.zfill(6).isin(exclude_codes)
            candidates = scored_df[mask]
        else:
            candidates = scored_df

        if candidates.empty:
            return None

        top = candidates.iloc[0]
        return {
            "name": top.get("종목명", "추천주"),
            "code": str(top.get("종목코드", "")).zfill(6),
            "score": float(top.get("DISPLAY_SCORE", top.get("TOTAL_SCORE", 0)) or 0),
        }
