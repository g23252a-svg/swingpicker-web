# -*- coding: utf-8 -*-
"""
tab_portfolio.py — 💼 내 자산 AI 진단 (NiceGUI 컴포넌트)
═══════════════════════════════════════════════════
[v1.0] Streamlit → NiceGUI 이식

핵심 차이점:
  - Streamlit: 버튼 클릭 → 전체 스크립트 재실행 → 화면 깜빡임
  - NiceGUI:   버튼 클릭 → async io_bound → 결과 영역만 갱신 (반짝임 0)

구조:
  ┌─ 에디터 영역 (보유 종목 입력)
  ├─ [분석 시작] 버튼 → async 현재가 조회
  ├─ 요약 메트릭 카드 (4열)
  ├─ 종목별 손익 테이블
  ├─ AI 진단 카드 (점수 낮은 순)
  └─ 자산 구성 파이차트
"""
import logging
from typing import List, Optional

import pandas as pd
from nicegui import ui, run

from state import global_data
from data.services.portfolio_service import (
    PortfolioService, Holding, HoldingResult, PortfolioSummary,
)

_logger = logging.getLogger(__name__)
_pf_svc = PortfolioService()


# ═══════════════════════════════════════════════════
#  메인 렌더 함수 (탭에서 호출)
# ═══════════════════════════════════════════════════

def render_tab_portfolio(auth_status: str):
    """Tab 3: 내 자산 AI 진단

    Args:
        auth_status: "guest" | "free" | "prime" | "pro" | "admin"
    """

    # ── 권한 체크 ──
    if auth_status in ("guest", "free"):
        with ui.card().classes("w-full p-8 text-center"):
            ui.icon("lock", size="xl").classes("text-gray-400")
            ui.label("🔒 내 자산 분석은 Pro 등급부터 이용 가능합니다.").classes(
                "text-lg text-gray-600 mt-4"
            )
            ui.label("체험권이 필요하시면 '문의 게시판'에 신청해주세요.").classes(
                "text-sm text-gray-400 mt-2"
            )
        return

    ui.label("💼 내 자산: AI 리밸런싱 & 진단").classes("text-2xl font-bold mb-4")

    # ── 상태 ──
    editor_rows: List[dict] = []
    result_container = ui.column().classes("w-full")

    # ── 1) 저장된 포트폴리오 로드 ──
    saved_text = _pf_svc.load_holdings_text()
    initial_rows = _parse_saved_text(saved_text)
    if not initial_rows:
        initial_rows = [{"종목명": "", "평단가": 0, "수량": 0}]

    # ── 2) 에디터 영역 ──
    ui.label("👇 현재 보유 중인 종목을 입력하세요. AI가 점수를 분석해 조언을 드립니다.").classes(
        "text-sm text-gray-500 mb-2"
    )

    columns = [
        {"name": "종목명", "label": "종목명", "field": "종목명", "align": "left", "sortable": True},
        {"name": "평단가", "label": "평단가(원)", "field": "평단가", "align": "right", "sortable": True},
        {"name": "수량", "label": "수량(주)", "field": "수량", "align": "right", "sortable": True},
    ]

    table = ui.table(
        columns=columns,
        rows=initial_rows,
        row_key="종목명",
        selection="none",
    ).classes("w-full")

    # ── 종목 추가/삭제 UI ──
    with ui.row().classes("gap-2 items-end mt-2"):
        inp_name = ui.input("종목명").classes("w-40")
        inp_price = ui.number("평단가", value=0, min=0, format="%.0f").classes("w-32")
        inp_qty = ui.number("수량", value=0, min=0, format="%.0f").classes("w-24")

        def add_row():
            name = inp_name.value.strip() if inp_name.value else ""
            if not name:
                ui.notify("종목명을 입력하세요", type="warning")
                return
            new_row = {"종목명": name, "평단가": int(inp_price.value or 0), "수량": int(inp_qty.value or 0)}
            table.rows.append(new_row)
            table.update()
            inp_name.value = ""
            inp_price.value = 0
            inp_qty.value = 0

        ui.button("➕ 종목 추가", on_click=add_row).props("dense color=primary")

        def remove_last():
            if table.rows:
                table.rows.pop()
                table.update()

        ui.button("➖ 마지막 삭제", on_click=remove_last).props("dense color=grey outline")

    ui.separator().classes("my-4")

    # ── 3) 분석 시작 버튼 ──
    analyze_btn = ui.button("🚀 AI 진단 시작", color="primary").classes("text-lg px-6 py-2")
    spinner_holder = ui.row().classes("items-center gap-2")

    async def on_analyze():
        """비동기 분석 — UI 블로킹 없음"""

        # 버튼 비활성화 + 스피너
        analyze_btn.disable()
        analyze_btn.text = "⏳ 분석 중..."
        with spinner_holder:
            spin = ui.spinner("dots", size="lg", color="primary")

        try:
            rows = table.rows
            if not rows or all(not r.get("종목명", "").strip() for r in rows):
                ui.notify("보유 종목을 입력해주세요", type="warning")
                return

            # 저장
            save_text = _pf_svc.holdings_to_text(rows)
            _pf_svc.save_holdings_text(save_text)

            # 파싱
            code_map = _get_code_map()
            holdings, cash = _pf_svc.parse_holdings_text(save_text, code_map)

            if not holdings and cash <= 0:
                ui.notify("분석할 종목이 없습니다", type="info")
                return

            # 현금만 있고 종목이 없는 경우 — 시세 조회 스킵, 요약만 렌더
            if not holdings and cash > 0:
                summary = _pf_svc.analyze([], cash, {}, scored_df)
                result_container.clear()
                with result_container:
                    _render_summary_metrics(summary)
                    ui.label("💰 현금만 보유 중입니다. 종목을 추가하면 AI 진단이 가능합니다.").classes(
                        "text-gray-500 mt-4"
                    )
                ui.notify("현금 포트폴리오 확인 완료", type="info")
                return

            # 현재가 병렬 조회 (I/O bound → 별도 스레드)
            price_map = await run.io_bound(_pf_svc.fetch_prices_parallel, holdings)

            # 점수 데이터
            scored_df = global_data.scored

            # 분석 실행
            summary = _pf_svc.analyze(holdings, cash, price_map, scored_df)

            # 결과 렌더링 (Reactive: 기존 내용 비우고 새로 그리기)
            result_container.clear()
            with result_container:
                _render_summary_metrics(summary)
                _render_holdings_table(summary)
                _render_ai_diagnosis_cards(summary, scored_df)
                _render_pie_chart(summary)

            ui.notify(f"✅ {summary.stock_count}개 종목 분석 완료!", type="positive")

        except Exception as e:
            _logger.error(f"포트폴리오 분석 실패: {e}")
            ui.notify(f"분석 중 오류: {e}", type="negative")

        finally:
            analyze_btn.enable()
            analyze_btn.text = "🚀 AI 진단 시작"
            spinner_holder.clear()

    analyze_btn.on_click(on_analyze)


# ═══════════════════════════════════════════════════
#  결과 렌더링 (Reactive 컴포넌트들)
# ═══════════════════════════════════════════════════

def _render_summary_metrics(s: PortfolioSummary):
    """전체 자산 요약 메트릭 카드 (4열)"""
    ui.separator()

    with ui.row().classes("w-full gap-4 flex-wrap mt-4"):
        _metric_card(
            "총 평가금액",
            f"{int(s.total_eval):,}원",
            icon="account_balance_wallet",
            color="blue",
        )
        _metric_card(
            "총 매입금액",
            f"{int(s.total_invest):,}원",
            icon="shopping_cart",
            color="gray",
        )

        pnl_color = "red" if s.total_pnl >= 0 else "blue"
        _metric_card(
            "총 평가손익",
            f"{int(s.total_pnl):+,}원",
            subtitle=f"{s.total_pnl_pct:+.2f}%",
            icon="trending_up" if s.total_pnl >= 0 else "trending_down",
            color=pnl_color,
        )

        if s.cash_amount > 0:
            _metric_card(
                "현금 비중",
                f"{s.cash_pct:.1f}%",
                subtitle=f"{int(s.cash_amount):,}원",
                icon="savings",
                color="green",
            )
        else:
            _metric_card(
                "종목 수",
                f"{s.stock_count}개",
                icon="list_alt",
                color="purple",
            )

    # 경고 메시지
    for w in s.warnings:
        if "집중" in w or "교체" in w:
            ui.label(w).classes("text-amber-600 font-medium mt-2 px-2")
        elif "손절" in w:
            ui.label(w).classes("text-red-600 font-bold mt-2 px-2")


def _render_holdings_table(s: PortfolioSummary):
    """종목별 손익 상세 테이블"""
    if not s.holdings:
        return

    with ui.expansion("📋 종목별 손익 상세", value=True).classes("w-full mt-4"):
        columns = [
            {"name": "name", "label": "종목명", "field": "name", "align": "left", "sortable": True},
            {"name": "curr", "label": "현재가", "field": "curr", "align": "right", "sortable": True},
            {"name": "avg", "label": "평단가", "field": "avg", "align": "right"},
            {"name": "qty", "label": "수량", "field": "qty", "align": "right"},
            {"name": "buy", "label": "매입금", "field": "buy", "align": "right"},
            {"name": "eval", "label": "평가금", "field": "eval", "align": "right"},
            {"name": "pnl", "label": "평가손익", "field": "pnl", "align": "right", "sortable": True},
            {"name": "pnl_pct", "label": "수익률", "field": "pnl_pct", "align": "right", "sortable": True},
            {"name": "weight", "label": "비중(%)", "field": "weight", "align": "right"},
            {"name": "score", "label": "AI점수", "field": "score", "align": "right", "sortable": True},
            {"name": "advice", "label": "AI조언", "field": "advice", "align": "left"},
        ]

        rows = []
        for h in s.holdings:
            rows.append({
                "name": h.name,
                "curr": f"{h.current_price:,}",
                "avg": f"{int(h.avg_price):,}",
                "qty": f"{h.qty:,}",
                "buy": f"{int(h.buy_amount):,}",
                "eval": f"{int(h.eval_amount):,}",
                "pnl": f"{int(h.pnl):+,}",
                "pnl_pct": f"{h.pnl_pct:+.2f}%",
                "weight": f"{h.weight_pct:.1f}",
                "score": f"{h.score:.0f}" if h.score > 0 else "-",
                "advice": h.advice,
            })

        ui.table(columns=columns, rows=rows, row_key="name").classes("w-full").props("dense flat")


def _render_ai_diagnosis_cards(s: PortfolioSummary, scored_df: pd.DataFrame):
    """AI 포트폴리오 진단 카드 (점수 낮은 순)"""
    ui.label("🩺 AI 포트폴리오 진단 결과").classes("text-xl font-bold mt-6 mb-2")

    if not s.holdings:
        ui.label("보유 종목 정보가 없습니다.").classes("text-gray-500")
        return

    # 점수 낮은 순 정렬 (위험 종목 먼저)
    sorted_holdings = sorted(s.holdings, key=lambda h: h.score)

    # 교체 추천용 보유 코드 수집
    held_codes = [h.code for h in s.holdings]

    for h in sorted_holdings:
        color_map = {
            "green": "bg-green-50 border-green-300",
            "blue": "bg-blue-50 border-blue-300",
            "red": "bg-red-50 border-red-300",
            "orange": "bg-amber-50 border-amber-300",
            "gray": "bg-gray-50 border-gray-300",
        }
        card_class = color_map.get(h.advice_color, "bg-gray-50")

        with ui.card().classes(f"w-full {card_class} border-l-4 p-4 mb-2"):
            with ui.row().classes("w-full items-center justify-between flex-wrap gap-2"):
                # 왼쪽: 종목 정보
                with ui.column().classes("min-w-[180px]"):
                    ui.label(h.name).classes("text-lg font-bold")
                    pct_color = "text-red-600" if h.pnl_pct >= 0 else "text-blue-600"
                    ui.label(f"{h.pnl_pct:+.2f}%").classes(f"text-xl font-bold {pct_color}")
                    ui.label(f"평가금: {int(h.eval_amount):,}원 | 비중: {h.weight_pct:.1f}%").classes(
                        "text-xs text-gray-500"
                    )

                # 가운데: AI 점수 프로그레스
                with ui.column().classes("min-w-[150px] items-center"):
                    ui.label("🤖 AI Score").classes("text-sm font-medium")
                    if h.score > 0:
                        score_color = (
                            "green" if h.score >= 70 else "orange" if h.score >= 40 else "red"
                        )
                        ui.linear_progress(
                            value=min(h.score / 100, 1.0),
                            color=score_color,
                            size="12px",
                        ).classes("w-32")
                        ui.label(f"{h.score:.0f}점").classes("text-xs text-gray-600")
                    else:
                        ui.label("분석불가").classes("text-xs text-gray-400")

                # 오른쪽: AI 조언
                with ui.column().classes("min-w-[200px]"):
                    ui.label(f"AI 의견: {h.advice}").classes("font-medium")

                    # 교체 추천 (40점 이하)
                    if 0 < h.score <= 40 and not scored_df.empty:
                        rec = _pf_svc.get_replacement_suggestion(scored_df, held_codes)
                        if rec:
                            ui.label(
                                f"💡 교체 추천: {rec['name']} ({rec['score']:.0f}점)"
                            ).classes("text-sm text-blue-600 mt-1")

                    elif h.score >= 80:
                        ui.label("🚀 추세가 강력합니다!").classes("text-sm text-green-600 mt-1")


def _render_pie_chart(s: PortfolioSummary):
    """자산 구성 파이 차트 (Plotly) — 종목 1개 이상일 때만 렌더"""
    if not s.holdings:
        return

    # 실제 종목(현금 제외)이 0개면 파이 차트 스킵
    real_holdings = [h for h in s.holdings if not h.is_cash]
    if not real_holdings:
        return

    labels = [h.name for h in s.holdings]
    values = [h.eval_amount for h in s.holdings]

    if s.cash_amount > 0:
        labels.append("현금 (CASH)")
        values.append(s.cash_amount)

    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo="label+percent",
            textposition="outside",
            marker=dict(line=dict(color="#ffffff", width=2)),
        )])
        fig.update_layout(
            title="📊 자산 구성",
            height=350,
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        ui.plotly(fig).classes("w-full mt-4")
    except ImportError:
        ui.label("📊 Plotly 미설치 — 파이 차트 생략").classes("text-gray-400")


# ═══════════════════════════════════════════════════
#  유틸 (내부)
# ═══════════════════════════════════════════════════

def _metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    icon: str = "info",
    color: str = "blue",
):
    """메트릭 카드 컴포넌트"""
    color_map = {
        "blue": "bg-blue-50 text-blue-700",
        "red": "bg-red-50 text-red-700",
        "green": "bg-green-50 text-green-700",
        "gray": "bg-gray-50 text-gray-700",
        "purple": "bg-purple-50 text-purple-700",
    }
    classes = color_map.get(color, "bg-gray-50 text-gray-700")

    with ui.card().classes(f"min-w-[200px] flex-1 {classes} p-4"):
        with ui.row().classes("items-center gap-2"):
            ui.icon(icon).classes("text-2xl opacity-60")
            ui.label(title).classes("text-sm opacity-80")
        ui.label(value).classes("text-2xl font-bold mt-1")
        if subtitle:
            ui.label(subtitle).classes("text-sm opacity-70")


def _parse_saved_text(text: str) -> List[dict]:
    """저장된 텍스트 → 테이블 rows"""
    rows = []
    if not text:
        return rows
    for line in text.strip().split("\n"):
        if ":" not in line:
            continue
        parts = line.split(":")
        if len(parts) < 3:
            continue
        try:
            nm = parts[0].strip()
            p = int(float(parts[1].replace(",", "").strip()))
            q = int(float(parts[2].replace(",", "").strip()))
            rows.append({"종목명": nm, "평단가": p, "수량": q})
        except (ValueError, TypeError):
            continue
    return rows


def _get_code_map() -> dict:
    """global_data.scored에서 {종목명: 종목코드} 매핑 추출"""
    df = global_data.scored
    if df.empty:
        return {}
    if "종목명" in df.columns and "종목코드" in df.columns:
        return dict(zip(df["종목명"], df["종목코드"].astype(str).str.zfill(6)))
    return {}
