# -*- coding: utf-8 -*-
"""
trade_journal_tab.py — 매매 복기(Trading Journal) 탭 (v1.0)
═══════════════════════════════════════════════════════════
퀀트 모델 추천가 vs 실제 체결가 차이(슬리피지) 기록 및 시각화.
손실 패턴 분석 → 파라미터 역튜닝 피드백 루프 구축.

저장소: SQLite ldy_trader.db (trades 테이블)
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

logger = logging.getLogger("trade_journal")

# ─────────────────────────────────────────────
#  DB 헬퍼 (인라인 SQLite — db_utils와 독립)
# ─────────────────────────────────────────────
import sqlite3

# ✅ [Fix #1] Railway/Docker 재배포 시에도 data/ 볼륨에 유지됨
# Railway volume mount: 프로젝트 루트의 data/ 폴더를 영구 볼륨으로 마운트해야 함
# railway.toml 예시:  [[mounts]]  source = "data"  target = "/app/data"
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(_DATA_DIR, exist_ok=True)          # 첫 배포 시 data/ 없으면 자동 생성
_DB_PATH = os.path.join(_DATA_DIR, "ldy_trader.db")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS trade_journal (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    stock_name   TEXT NOT NULL,
    stock_code   TEXT,
    direction    TEXT DEFAULT 'LONG',
    recommend_price  REAL,
    actual_price     REAL,
    stop_price       REAL,
    target_price     REAL,
    exit_price       REAL,
    qty              INTEGER DEFAULT 0,
    slippage_pct     REAL,
    outcome          TEXT,    -- WIN / LOSS / OPEN
    profit_pct       REAL,
    route            TEXT,    -- ATTACK / ARMED / WAIT
    score            REAL,
    notes            TEXT,
    tags             TEXT     -- JSON array string
);
"""

def _get_conn():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_SQL)
    conn.commit()
    return conn


def save_trade(entry: dict) -> int:
    """신규 거래 기록 저장. 생성된 id 반환."""
    conn = _get_conn()
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        rec_p  = float(entry.get("recommend_price", 0) or 0)
        act_p  = float(entry.get("actual_price", 0) or 0)
        exit_p = float(entry.get("exit_price", 0) or 0)

        slip = (act_p - rec_p) / rec_p * 100 if rec_p > 0 and act_p > 0 else None

        outcome = entry.get("outcome", "OPEN")
        profit  = None
        if exit_p > 0 and act_p > 0:
            profit = (exit_p - act_p) / act_p * 100
            if profit > 0:  outcome = "WIN"
            elif profit < 0: outcome = "LOSS"

        tags = json.dumps(entry.get("tags", []), ensure_ascii=False)

        cur = conn.execute("""
            INSERT INTO trade_journal
            (created_at, stock_name, stock_code, direction,
             recommend_price, actual_price, stop_price, target_price, exit_price,
             qty, slippage_pct, outcome, profit_pct, route, score, notes, tags)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            entry.get("created_at", now),
            entry.get("stock_name", ""),
            entry.get("stock_code", ""),
            entry.get("direction", "LONG"),
            rec_p, act_p,
            float(entry.get("stop_price", 0) or 0),
            float(entry.get("target_price", 0) or 0),
            exit_p,
            int(entry.get("qty", 0) or 0),
            slip, outcome, profit,
            entry.get("route", ""),
            float(entry.get("score", 0) or 0),
            entry.get("notes", ""),
            tags,
        ))
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        logger.error(f"trade_journal 저장 실패: {e}")
        return -1
    finally:
        conn.close()


def load_trades(limit: int = 200) -> List[Dict]:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT * FROM trade_journal ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def delete_trade(trade_id: int) -> bool:
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM trade_journal WHERE id=?", (trade_id,))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def update_exit(trade_id: int, exit_price: float, notes: str = "") -> bool:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT actual_price FROM trade_journal WHERE id=?", (trade_id,)
        ).fetchone()
        if not row:
            return False
        act_p = row["actual_price"] or 0
        profit = (exit_price - act_p) / act_p * 100 if act_p > 0 else 0
        outcome = "WIN" if profit > 0 else "LOSS" if profit < 0 else "OPEN"
        conn.execute("""
            UPDATE trade_journal
            SET exit_price=?, profit_pct=?, outcome=?, notes=?
            WHERE id=?
        """, (exit_price, profit, outcome, notes, trade_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"exit 업데이트 실패: {e}")
        return False
    finally:
        conn.close()


# ─────────────────────────────────────────────
#  분석 함수들
# ─────────────────────────────────────────────
def compute_stats(trades: List[Dict]) -> dict:
    closed = [t for t in trades if t["outcome"] in ("WIN", "LOSS")]
    wins   = [t for t in closed if t["outcome"] == "WIN"]
    losses = [t for t in closed if t["outcome"] == "LOSS"]

    if not closed:
        return {"total": len(trades), "closed": 0, "win_rate": 0,
                "avg_profit": 0, "avg_slip": 0, "expectancy": 0}

    wr = len(wins) / len(closed) * 100
    profits = [t["profit_pct"] for t in closed if t["profit_pct"] is not None]
    slips   = [t["slippage_pct"] for t in trades if t["slippage_pct"] is not None]
    avg_w   = sum(t["profit_pct"] for t in wins if t["profit_pct"]) / max(len(wins),1)
    avg_l   = sum(abs(t["profit_pct"]) for t in losses if t["profit_pct"]) / max(len(losses),1)
    expct   = (wr/100 * avg_w) - ((1-wr/100) * avg_l)

    return {
        "total":       len(trades),
        "closed":      len(closed),
        "open":        len(trades) - len(closed),
        "wins":        len(wins),
        "losses":      len(losses),
        "win_rate":    round(wr, 1),
        "avg_profit":  round(sum(profits)/len(profits), 2) if profits else 0,
        "avg_slip":    round(sum(slips)/len(slips), 2) if slips else 0,
        "expectancy":  round(expct, 2),
    }


# ─────────────────────────────────────────────
#  NiceGUI 탭 렌더러
# ─────────────────────────────────────────────
def render_trade_journal_tab(df_scored=None):
    """Tab 9 전체 UI 렌더"""
    try:
        from nicegui import ui
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
    except ImportError:
        return

    # ── 입력 폼 ──────────────────────────────
    with ui.expansion("✏️ 새 매매 기록 추가", value=True).classes("w-full mb-4"):
        with ui.row().classes("w-full gap-3 flex-wrap"):
            f_name  = ui.input("종목명").classes("flex-1 min-w-[130px]")
            f_code  = ui.input("종목코드").classes("min-w-[100px]")
            f_route = ui.select(["ATTACK","ARMED","WAIT","NEUTRAL"], value="ATTACK",
                                label="상태").classes("min-w-[100px]")
            f_score = ui.number("시스템 점수", value=0, min=0, max=100).classes("min-w-[100px]")

        with ui.row().classes("w-full gap-3 flex-wrap"):
            f_rec   = ui.number("추천 매수가",  value=0, min=0, format="%,.0f").classes("flex-1")
            f_act   = ui.number("실제 체결가",  value=0, min=0, format="%,.0f").classes("flex-1")
            f_stop  = ui.number("손절가",       value=0, min=0, format="%,.0f").classes("flex-1")
            f_tgt   = ui.number("목표가 (T1)",  value=0, min=0, format="%,.0f").classes("flex-1")
            f_qty   = ui.number("수량",         value=0, min=0).classes("min-w-[80px]")

        f_notes = ui.input("메모", placeholder="진입 근거, 특이사항 등").classes("w-full")
        save_msg = ui.label("").classes("text-sm mt-1")

        def _save():
            if not f_name.value.strip():
                save_msg.set_text("⚠️ 종목명 필수"); return
            tid = save_trade({
                "stock_name":      f_name.value.strip(),
                "stock_code":      f_code.value.strip(),
                "route":           f_route.value,
                "score":           f_score.value,
                "recommend_price": f_rec.value,
                "actual_price":    f_act.value,
                "stop_price":      f_stop.value,
                "target_price":    f_tgt.value,
                "qty":             f_qty.value,
                "notes":           f_notes.value,
            })
            if tid > 0:
                save_msg.set_text(f"✅ 저장 완료 (id={tid})")
                save_msg.classes(replace="text-sm mt-1 text-green-400")
                _refresh()
            else:
                save_msg.set_text("❌ 저장 실패")
                save_msg.classes(replace="text-sm mt-1 text-red-400")

        ui.button("💾 기록 저장", on_click=_save).props("color=primary").classes("mt-2")

    # ── 통계 요약 ──────────────────────────────
    stats_area = ui.row().classes("w-full gap-3 flex-wrap")
    # ── 차트 영역 ──────────────────────────────
    chart_area = ui.column().classes("w-full mt-4")
    # ── 목록 테이블 ────────────────────────────
    table_area = ui.column().classes("w-full mt-4")

    def _refresh():
        trades = load_trades(200)
        stats  = compute_stats(trades)

        # 통계 카드
        stats_area.clear()
        with stats_area:
            _sc("총 기록",    f'{stats["total"]}건')
            _sc("승률",       f'{stats["win_rate"]}%',
                color="#10B981" if stats["win_rate"] >= 50 else "#EF4444")
            _sc("평균 손익",  f'{stats["avg_profit"]:+.2f}%',
                color="#10B981" if stats["avg_profit"] >= 0 else "#EF4444")
            _sc("평균 슬리피지", f'{stats["avg_slip"]:+.2f}%',
                color="#F59E0B")
            _sc("기대수익 (Exp.)", f'{stats["expectancy"]:+.2f}%',
                color="#10B981" if stats["expectancy"] >= 0 else "#EF4444")

        # 차트
        chart_area.clear()
        if trades:
            _draw_charts(trades, chart_area)

        # 목록 테이블
        table_area.clear()
        _draw_table(trades, table_area)

    def _sc(title, val, color="white"):
        with stats_area:
            with ui.card().classes("p-3 min-w-[120px] bg-[#1a1a2e] border border-gray-700 rounded-xl"):
                ui.label(title).classes("text-xs text-gray-400")
                ui.label(val).classes("text-base font-bold").style(f"color:{color}")

    def _draw_charts(trades, container):
        import pandas as pd
        df = pd.DataFrame(trades)
        df = df[df["outcome"].isin(["WIN","LOSS"])]
        if df.empty:
            return

        with container:
            with ui.row().classes("w-full gap-4 flex-wrap"):
                # 1. 슬리피지 히스토그램
                with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
                    slip_data = df["slippage_pct"].dropna()
                    if not slip_data.empty:
                        fig_s = go.Figure()
                        fig_s.add_trace(go.Histogram(
                            x=slip_data, nbinsx=20,
                            marker_color="#F59E0B", opacity=0.7, name="슬리피지"
                        ))
                        fig_s.update_layout(
                            title="📊 슬리피지 분포 (%)",
                            height=250, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                            margin=dict(t=40,b=20,l=20,r=20),
                        )
                        ui.plotly(fig_s).classes("w-full")

                # 2. 누적 손익 곡선
                with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
                    df_s = df.sort_values("created_at").dropna(subset=["profit_pct"])
                    if not df_s.empty:
                        df_s["cumulative"] = df_s["profit_pct"].cumsum()
                        fig_c = go.Figure()
                        colors = ["#10B981" if v >= 0 else "#EF4444" for v in df_s["cumulative"]]
                        fig_c.add_trace(go.Scatter(
                            x=list(range(len(df_s))), y=df_s["cumulative"],
                            mode="lines+markers", fill="tozeroy",
                            fillcolor="rgba(16,185,129,0.15)",
                            line=dict(color="#10B981", width=2),
                        ))
                        fig_c.update_layout(
                            title="📈 누적 손익 곡선 (%)",
                            height=250, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                            margin=dict(t=40,b=20,l=20,r=20),
                        )
                        ui.plotly(fig_c).classes("w-full")

            # 3. 패턴별 손실 분석 (ROUTE × WIN/LOSS)
            if "route" in df.columns and not df["route"].isna().all():
                with ui.card().classes("w-full p-2 bg-[#1a1a2e] mt-2"):
                    grp = (
                        df.groupby(["route", "outcome"])
                        .size().reset_index(name="count")
                    )
                    if not grp.empty:
                        fig_g = px.bar(
                            grp, x="route", y="count", color="outcome",
                            color_discrete_map={"WIN":"#10B981","LOSS":"#EF4444"},
                            title="🔍 상태별 승/패 패턴",
                            barmode="group",
                        )
                        fig_g.update_layout(
                            height=250, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                            margin=dict(t=40,b=20,l=20,r=20),
                        )
                        ui.plotly(fig_g).classes("w-full")

    def _draw_table(trades, container):
        with container:
            ui.label(f"📂 매매 기록 ({len(trades)}건)").classes("text-white font-bold mb-2")
            if not trades:
                ui.label("기록이 없습니다.").classes("text-gray-400")
                return

            cols = [
                {"name":"date",  "label":"일자",    "field":"date",   "align":"left"},
                {"name":"name",  "label":"종목명",  "field":"name",   "align":"left"},
                {"name":"rec",   "label":"추천가",  "field":"rec",    "align":"right"},
                {"name":"act",   "label":"체결가",  "field":"act",    "align":"right"},
                {"name":"slip",  "label":"슬리피지","field":"slip",   "align":"right"},
                {"name":"exit",  "label":"청산가",  "field":"exit",   "align":"right"},
                {"name":"pnl",   "label":"손익%",   "field":"pnl",    "align":"right"},
                {"name":"out",   "label":"결과",    "field":"out",    "align":"center"},
                {"name":"route", "label":"상태",    "field":"route",  "align":"center"},
            ]
            rows = []
            for t in trades:
                slip = t.get("slippage_pct")
                pnl  = t.get("profit_pct")
                rows.append({
                    "id":    t["id"],
                    "date":  str(t.get("created_at",""))[:10],
                    "name":  t.get("stock_name",""),
                    "rec":   f'{int(t.get("recommend_price") or 0):,}',
                    "act":   f'{int(t.get("actual_price") or 0):,}',
                    "slip":  f'{slip:+.2f}%' if slip is not None else "-",
                    "exit":  f'{int(t.get("exit_price") or 0):,}' if t.get("exit_price") else "-",
                    "pnl":   f'{pnl:+.2f}%' if pnl is not None else "-",
                    "out":   {"WIN":"✅","LOSS":"❌","OPEN":"⏳"}.get(t.get("outcome",""),"?"),
                    "route": t.get("route",""),
                })

            exit_id   = ui.number("청산 기록 ID", value=0, min=1).classes("min-w-[100px]")
            exit_px   = ui.number("청산가", value=0, min=0, format="%,.0f").classes("min-w-[120px]")
            exit_note = ui.input("메모").classes("flex-1")

            def _do_exit():
                tid = int(exit_id.value or 0)
                ep  = float(exit_px.value or 0)
                if tid > 0 and ep > 0:
                    ok = update_exit(tid, ep, exit_note.value)
                    ui.notify("✅ 청산 기록 완료" if ok else "❌ 실패")
                    _refresh()

            with ui.row().classes("w-full gap-3 items-end mb-3 flex-wrap"):
                exit_id; exit_px; exit_note
                ui.button("📝 청산 기록", on_click=_do_exit).props("dense flat").classes(
                    "text-green-400 border border-green-700"
                )

            tbl = ui.table(
                columns=cols, rows=rows, row_key="id", pagination={"rowsPerPage": 15}
            ).classes("w-full").props("dense dark flat bordered")

    _refresh()
