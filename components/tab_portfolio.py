# -*- coding: utf-8 -*-
"""
tab_portfolio.py — 💼 내 자산: AI 리밸런싱 & 진단 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
포트폴리오 입력, 비동기 시세 조회, AI 진단, Kelly 사이저
"""
import asyncio
import glob
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
from nicegui import ui, app

from shared_utils import nz_num, safe_float

try:
    from async_helpers import run_sync, _io_pool
except ImportError:
    async def run_sync(fn, *a, **kw):
        return fn(*a, **kw)
    _io_pool = None

FDR_OK = False
fdr = None
try:
    import FinanceDataReader as _fdr
    fdr = _fdr
    FDR_OK = True
except ImportError:
    pass

try:
    from price_cache import fetch_with_cache, fetch_prices_async
    PRICE_CACHE_OK = True
except ImportError:
    PRICE_CACHE_OK = False

try:
    from kelly_widget import render_kelly_calculator, render_portfolio_kelly_summary
    KELLY_OK = True
except ImportError:
    KELLY_OK = False

_logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# 이용권 가격
try:
    from version_info import PRICE_PRIME
except ImportError:
    PRICE_PRIME = 19_900
    PRICE_PRIME = 19_900


# ══════════════════════════════════════════════════════
#  UI 유틸
# ══════════════════════════════════════════════════════

def _section_title(text):
    ui.label(text).classes("text-lg font-bold text-white mt-6 mb-2 border-b border-gray-700 pb-2")


def _metric_card(title, value, delta="", positive=True):
    with ui.card().classes("p-4 min-w-[140px] bg-[#1a1a2e] border border-gray-700 rounded-xl"):
        ui.label(title).classes("text-xs text-gray-400 uppercase tracking-wide")
        ui.label(str(value)).classes("text-xl font-bold text-white mt-1")
        if delta:
            color = "text-green-400" if positive else "text-red-400"
            ui.label(str(delta)).classes(f"text-sm {color} mt-0.5")


def _plotly_dark(fig, height=300):
    if fig:
        fig.update_layout(
            height=height, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
            margin=dict(t=30, b=10, l=10, r=10),
        )
    return fig


# ══════════════════════════════════════════════════════
#  데이터 유틸 (main.py에서 이식)
# ══════════════════════════════════════════════════════

# KRX 전체 종목 캐시
_KRX_NAME_MAP = {}

def _ensure_krx_map():
    global _KRX_NAME_MAP
    if _KRX_NAME_MAP:
        return
    if FDR_OK:
        try:
            listing = fdr.StockListing("KRX")
            if listing is not None and not listing.empty:
                for _, r in listing.iterrows():
                    code = str(r.get("Code", "")).zfill(6)
                    name = str(r.get("Name", ""))
                    if code and name:
                        _KRX_NAME_MAP[name] = code
        except Exception:
            pass
    # CSV 폴백
    if not _KRX_NAME_MAP:
        csv_path = os.path.join(DATA_DIR, "krx_names_latest.csv")
        if os.path.exists(csv_path):
            try:
                kdf = pd.read_csv(csv_path, dtype=str)
                if "종목코드" in kdf.columns and "종목명" in kdf.columns:
                    _KRX_NAME_MAP.update(dict(zip(kdf["종목명"], kdf["종목코드"].str.zfill(6))))
            except Exception:
                pass


def _get_code_map(df):
    if df.empty or "종목코드" not in df.columns or "종목명" not in df.columns:
        return {}
    return dict(zip(df["종목명"], df["종목코드"].astype(str).str.zfill(6)))


def _find_code_by_name(name, code_map):
    if name in code_map: return code_map[name]
    for k, v in code_map.items():
        if name in k or k in name: return v
    _ensure_krx_map()
    if name in _KRX_NAME_MAP:
        return _KRX_NAME_MAP[name]
    for k, v in _KRX_NAME_MAP.items():
        if name in k or k in name:
            return v
    return name


def _fetch_current_price(code, name):
    """현재가 조회 — 캐시 → Circuit Breaker → FDR 순"""
    code_str = str(code).zfill(6) if str(code).isdigit() else ""

    def _fdr_fetch(c):
        if not FDR_OK or not c: return 0
        try:
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            d = fdr.DataReader(c, start)
            if d is not None and not d.empty:
                return int(d.iloc[-1]["Close"])
        except Exception:
            pass
        return 0

    if PRICE_CACHE_OK and code_str:
        c, n, p = fetch_with_cache(code_str, name, _fdr_fetch)
        if p > 0: return c, n, p

    if FDR_OK and not code_str:
        _ensure_krx_map()
        found = _KRX_NAME_MAP.get(name)
        if not found:
            for k, v in _KRX_NAME_MAP.items():
                if name in k or k in name:
                    found = v; break
        if found:
            if PRICE_CACHE_OK:
                c, n, p = fetch_with_cache(found, name, _fdr_fetch)
                if p > 0: return found, name, p
            else:
                p = _fdr_fetch(found)
                if p > 0: return found, name, p

    if FDR_OK and code_str:
        p = _fdr_fetch(code_str)
        if p > 0: return code, name, p

    return code, name, 0


# Portfolio Gist I/O
def _load_portfolio_file():
    token = os.environ.get("LDY_GIST_TOKEN", "")
    gist_id = os.environ.get("LDY_GIST_ID", "")
    if not token or not gist_id: return ""
    try:
        import requests
        r = requests.get(f"https://api.github.com/gists/{gist_id}",
                        headers={"Authorization": f"token {token}"}, timeout=10)
        if r.ok:
            files = r.json().get("files", {})
            if "portfolio.txt" in files:
                return files["portfolio.txt"]["content"]
    except Exception:
        pass
    return ""


def _save_portfolio_file(text_data):
    token = os.environ.get("LDY_GIST_TOKEN", "")
    gist_id = os.environ.get("LDY_GIST_ID", "")
    if not token or not gist_id: return False
    try:
        import requests
        r = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {token}"},
            json={"files": {"portfolio.txt": {"content": text_data}}},
            timeout=10,
        )
        return r.ok
    except Exception:
        return False


# ── 과거 추천 캐시 ──
_hist_recommend_cache: dict = {}
_hist_cache_loaded = False


def _ensure_hist_cache():
    global _hist_recommend_cache, _hist_cache_loaded
    if _hist_cache_loaded:
        return
    _hist_cache_loaded = True

    pattern = os.path.join(DATA_DIR, "recommend_*.csv")
    files = sorted(glob.glob(pattern), reverse=True)

    for fpath in files[:7]:
        if "latest" in fpath: continue
        try:
            hdf = pd.read_csv(fpath, dtype={"종목코드": str, "종목명": str})
            for _, r in hdf.iterrows():
                code = str(r.get("종목코드", "")).zfill(6)
                if code and code not in _hist_recommend_cache:
                    _hist_recommend_cache[code] = {
                        "종목명": str(r.get("종목명", "")),
                        "DISPLAY_SCORE": safe_float(r.get("DISPLAY_SCORE", r.get("FINAL_SCORE", 0))),
                        "ROUTE": str(r.get("ROUTE", r.get("상태", ""))),
                        "추천매수가": nz_num(r.get("추천매수가", 0)),
                        "손절가": nz_num(r.get("손절가", 0)),
                        "추천매도가1": nz_num(r.get("추천매도가1", 0)),
                        "종가": nz_num(r.get("종가", 0)),
                        "_source_file": os.path.basename(fpath),
                    }
        except Exception as e:
            _logger.debug(f"과거 추천 캐시 로드 실패 ({fpath}): {e}")

    if _hist_recommend_cache:
        _logger.info(f"📦 과거 추천 캐시: {len(_hist_recommend_cache)}종목")


def _lookup_stock_info(code, name, df):
    code6 = str(code).zfill(6)
    if not df.empty and "종목코드" in df.columns:
        match = df[df["종목코드"].astype(str).str.zfill(6) == code6]
        if match.empty and "종목명" in df.columns:
            match = df[df["종목명"] == name]
        if not match.empty:
            r = match.iloc[0]
            return (safe_float(r.get("DISPLAY_SCORE", 0)), str(r.get("ROUTE", "")), "금일추천")
    _ensure_hist_cache()
    hist = _hist_recommend_cache.get(code6)
    if hist:
        return (hist["DISPLAY_SCORE"], hist["ROUTE"], f"전일추천({hist.get('_source_file', '')[10:18]})")
    return (0, "", "미추천")


# ══════════════════════════════════════════════════════
#  메인 렌더
# ══════════════════════════════════════════════════════

def render_tab_portfolio(df, auth):
    """Tab 3: 내 자산 (포트폴리오 AI 진단)

    Args:
        df: 추천 종목 DataFrame
        auth: "guest" | "free" | "pro" | "prime" | "admin"
    """
    if auth in ("guest", "free"):
        with ui.card().classes("w-full p-8 bg-[#1a1a2e] border border-gray-700 rounded-xl text-center"):
            ui.label("🔒 내 자산 분석").classes("text-2xl font-bold text-white mb-2")
            ui.label("Prime 회원 전용 기능입니다").classes("text-gray-400 mb-2")
            ui.label(f"👑 Prime ({PRICE_PRIME:,}원/월) · 신규 가입 시 14일 무료체험!").classes("text-gray-400 text-sm mb-4")
            with ui.row().classes("justify-center mt-2 gap-4"):
                ui.html("""
                <div style="text-align:center; padding:16px; border:1px solid #374151; border-radius:12px; min-width:100px;">
                    <div style="font-size:24px;">🤖</div>
                    <div style="color:#9CA3AF; font-size:13px; margin-top:4px;">AI 리밸런싱</div>
                </div>
                <div style="text-align:center; padding:16px; border:1px solid #374151; border-radius:12px; min-width:100px;">
                    <div style="font-size:24px;">📊</div>
                    <div style="color:#9CA3AF; font-size:13px; margin-top:4px;">섹터 분석</div>
                </div>
                <div style="text-align:center; padding:16px; border:1px solid #374151; border-radius:12px; min-width:100px;">
                    <div style="font-size:24px;">⚡</div>
                    <div style="color:#9CA3AF; font-size:13px; margin-top:4px;">실시간 진단</div>
                </div>
                """)
            ui.button(
                "💎 멤버십 업그레이드 알아보기",
                on_click=lambda: ui.run_javascript(
                    "document.querySelector('[role=tab]:nth-child(4)')?.click()"
                ),
            ).classes("mt-4").props("color=primary rounded size=lg")
        return

    _section_title("💼 내 자산: AI 리밸런싱 & 진단")

    # [v21.3] 종목 검색 기반 입력 UI
    saved_local = app.storage.user.get("portfolio_text", "")
    saved_gist = _load_portfolio_file() if not saved_local else ""
    saved = saved_local or saved_gist or ""

    # 종목명 목록 (추천 CSV + KRX)
    code_map = _get_code_map(df)
    stock_names = sorted(code_map.keys()) if code_map else []

    ui.label("📌 보유 종목 추가").classes("text-sm font-bold text-white mb-2")

    with ui.row().classes("w-full gap-3 items-end flex-wrap"):
        stock_select = ui.select(
            stock_names, with_input=True, label="종목명 검색",
            value=None,
        ).classes("min-w-[200px] flex-1").props("clearable use-input")

        avg_price_input = ui.number(
            "평단가 (원)", value=None, min=0, step=100, format="%.0f"
        ).classes("min-w-[130px]")

        qty_input = ui.number(
            "수량 (주)", value=None, min=1, step=1, format="%.0f"
        ).classes("min-w-[100px]")

        def _add_stock():
            name = stock_select.value
            avg = avg_price_input.value
            qty = qty_input.value
            if not name:
                ui.notify("종목명을 선택하세요", type="warning"); return
            if not avg or avg <= 0:
                ui.notify("평단가를 입력하세요", type="warning"); return
            if not qty or qty <= 0:
                ui.notify("수량을 입력하세요", type="warning"); return

            new_line = f"{name}:{int(avg)}:{int(qty)}"
            current = pf_input.value.strip()
            # 중복 체크
            existing_names = [l.split(":")[0] for l in current.split("\n") if ":" in l]
            if name in existing_names:
                # 기존 종목 업데이트
                lines = current.split("\n")
                updated = []
                for l in lines:
                    if l.startswith(f"{name}:"):
                        updated.append(new_line)
                    else:
                        updated.append(l)
                pf_input.value = "\n".join(updated)
                ui.notify(f"✏️ {name} 업데이트 완료", type="positive")
            else:
                pf_input.value = f"{current}\n{new_line}" if current else new_line
                ui.notify(f"✅ {name} 추가 완료", type="positive")

            app.storage.user["portfolio_text"] = pf_input.value
            # 입력 초기화
            stock_select.value = None
            avg_price_input.value = None
            qty_input.value = None

        ui.button("➕ 추가", on_click=_add_stock).props("color=primary dense").classes("h-10")

    # 현재 보유 목록 미니 테이블
    holding_area = ui.column().classes("w-full mt-2 mb-2")

    def _refresh_holdings():
        holding_area.clear()
        text = pf_input.value.strip()
        if not text:
            return
        items = []
        for line in text.split("\n"):
            if ":" not in line:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            try:
                items.append({"name": parts[0].strip(), "avg": int(parts[1]), "qty": int(parts[2])})
            except (ValueError, IndexError):
                pass

        if not items:
            return

        with holding_area:
            with ui.row().classes("w-full gap-2 flex-wrap"):
                for item in items:
                    with ui.card().classes("p-2 bg-[#0d0d1a] border border-gray-700 rounded-lg"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(f"{item['name']}").classes("text-white text-sm font-bold")
                            ui.label(f"{item['avg']:,}원 × {item['qty']}주").classes("text-xs text-gray-400")
                            val = item['avg'] * item['qty']
                            ui.label(f"= {val:,}원").classes("text-xs text-cyan-400")

                            def _remove(n=item['name']):
                                lines = [l for l in pf_input.value.strip().split("\n")
                                         if not l.startswith(f"{n}:")]
                                pf_input.value = "\n".join(lines)
                                app.storage.user["portfolio_text"] = pf_input.value
                                ui.notify(f"🗑️ {n} 제거", type="info")
                                _refresh_holdings()

                            ui.button("✕", on_click=_remove).props("flat dense size=xs color=red")

    # 기존 textarea (숨김 — 데이터 저장용)
    with ui.expansion("📋 텍스트 직접 편집 (고급)", value=False).classes("w-full text-xs text-gray-500"):
        pf_input = ui.textarea("포트폴리오 데이터", value=saved,
                               placeholder="종목명:평단가:수량 (줄바꿈 구분)\n예) 에코프로머티:67341:60").classes("w-full").props("rows=4")

    result_area = ui.column().classes("w-full mt-4")

    def _auto_save():
        app.storage.user["portfolio_text"] = pf_input.value
        _refresh_holdings()

    pf_input.on("blur", lambda _: _auto_save())
    _refresh_holdings()

    async def analyze():
        result_area.clear()
        text = pf_input.value.strip()
        if not text: return

        app.storage.user["portfolio_text"] = text
        await run_sync(_save_portfolio_file, text)
        ui.notify("💾 포트폴리오 저장됨", type="positive")

        code_map = _get_code_map(df)
        targets = []
        cash_amt = 0.0

        for line in text.split("\n"):
            if ":" not in line: continue
            parts = line.split(":")
            if len(parts) < 3: continue
            try:
                nm = parts[0].strip()
                price = int(float(parts[1].replace(",", "").strip()))
                qty = int(float(parts[2].replace(",", "").strip()))
            except (ValueError, TypeError):
                continue
            if nm.upper() == "CASH" or "현금" in nm:
                cash_amt += price * qty
            else:
                real_code = _find_code_by_name(nm, code_map) or nm
                targets.append((real_code, nm, price, qty))

        if not targets and cash_amt <= 0:
            with result_area:
                ui.label("입력된 종목이 없습니다.").classes("text-gray-400")
            return

        with result_area:
            ui.label("⚡ 시세 조회 중...").classes("text-gray-400")

        # 비동기 현재가 조회
        price_map = {}
        if PRICE_CACHE_OK and FDR_OK:
            try:
                price_results = await fetch_prices_async(
                    [(t[0], t[1]) for t in targets], fdr
                )
                price_map = price_results
            except Exception as _ae:
                _logger.warning(f"async 조회 실패, ThreadPool fallback: {_ae}")

        if not price_map:
            if _io_pool:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(_io_pool, _fetch_current_price, t[0], t[1])
                    for t in targets
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, tuple) and len(res) == 3:
                        c, n, p = res
                        price_map[c] = p

        total_eval = total_buy = 0.0
        pf_rows = []
        for code, name, avg, qty in targets:
            curr = price_map.get(code, 0)

            # 폴백 1: scored 종가
            if curr == 0 and not df.empty and '종가' in df.columns:
                match_p = df[df['종목코드'] == str(code).zfill(6)] if '종목코드' in df.columns else pd.DataFrame()
                if match_p.empty and '종목명' in df.columns:
                    match_p = df[df['종목명'] == name]
                if not match_p.empty:
                    curr = int(nz_num(match_p.iloc[0].get('종가', 0)))

            # 폴백 2: 과거 추천 캐시 종가
            if curr == 0:
                _ensure_hist_cache()
                hist = _hist_recommend_cache.get(str(code).zfill(6))
                if hist and hist.get("종가", 0) > 0:
                    curr = int(hist["종가"])

            # 폴백 3: price_snapshots
            if curr == 0:
                for _snap_name in ["price_snapshot_latest.csv", "price_snapshot.csv"]:
                    _snap_path = os.path.join(DATA_DIR, _snap_name)
                    if os.path.exists(_snap_path):
                        try:
                            _snap = pd.read_csv(_snap_path, dtype={"종목코드": str})
                            _sm = _snap[_snap["종목코드"].astype(str).str.zfill(6) == str(code).zfill(6)]
                            if _sm.empty and "종목명" in _snap.columns:
                                _sm = _snap[_snap["종목명"] == name]
                            if not _sm.empty and "종가" in _snap.columns:
                                _p = int(nz_num(_sm.iloc[0]["종가"]))
                                if _p > 0:
                                    curr = _p; break
                        except Exception:
                            pass

            # 폴백 4: 평단가
            if curr == 0 and avg > 0:
                curr = avg

            _price_src = ""
            if curr == avg and curr > 0:
                _price_src = " (평단가)"
            elif curr > 0 and price_map.get(code, 0) == 0:
                _price_src = " (전일종가)"

            eval_amt = curr * qty
            buy_amt = avg * qty
            total_eval += eval_amt
            total_buy += buy_amt
            pct = (curr - avg) / avg * 100 if avg > 0 and curr > 0 else 0

            score, route, source = _lookup_stock_info(code, name, df)

            if source == "금일추천":
                if score >= 80: advice, acolor = "💪강력홀딩", "#10B981"
                elif score >= 60: advice, acolor = "👌보유(양호)", "#3B82F6"
                elif score <= 40 and score > 0: advice, acolor = "⚠️교체권장", "#EF4444"
                else: advice, acolor = "👀관망", "#F59E0B"
            elif source.startswith("전일추천"):
                if score >= 70: advice, acolor = f"📤금일 제외 (전일 {score:.0f}점) — 홀딩 검토", "#F59E0B"
                elif score >= 50: advice, acolor = f"📤금일 제외 (전일 {score:.0f}점) — 모니터링", "#F59E0B"
                else: advice, acolor = f"📤금일 제외 (전일 {score:.0f}점) — 손절 검토", "#EF4444"
            else:
                if curr == 0: advice, acolor = "❓시세조회 실패", "#EF4444"
                else: advice, acolor = "ℹ️시스템 외 종목", "#9CA3AF"

            pf_rows.append({"종목명": name, "현재가": curr, "평단가": avg, "수량": qty,
                            "매입금": buy_amt, "평가금": eval_amt, "수익률": pct,
                            "점수": score, "상태": route, "소스": source,
                            "가격소스": _price_src,
                            "AI조언": advice, "색상": acolor, "code": code})

        result_area.clear()
        with result_area:
            total_asset = total_eval + cash_amt
            total_invest = total_buy + cash_amt
            total_rate = (total_asset - total_invest) / total_invest * 100 if total_invest > 0 else 0

            with ui.row().classes("w-full gap-4 flex-wrap"):
                _metric_card("총 평가금액", f"{int(total_asset):,}원")
                _metric_card("총 매입금액", f"{int(total_invest):,}원")
                _metric_card("총 평가손익", f"{int(total_asset - total_invest):+,}원", f"{total_rate:+.2f}%", total_rate >= 0)
                if cash_amt > 0:
                    _metric_card("현금 비중", f"{cash_amt/total_asset*100:.1f}%" if total_asset > 0 else "0%", f"{int(cash_amt):,}원")

            _section_title("🩺 AI 포트폴리오 진단")
            pf_rows.sort(key=lambda x: x["점수"])
            for r in pf_rows:
                with ui.card().classes("w-full p-4 mb-2 bg-[#1a1a2e] border border-gray-700 rounded-xl"):
                    with ui.row().classes("w-full justify-between items-center"):
                        with ui.column().classes("gap-0"):
                            with ui.row().classes("items-center gap-2"):
                                ui.label(r["종목명"]).classes("text-white font-bold")
                                if r.get("상태"):
                                    _rc = {"ATTACK": "red", "ARMED": "orange", "WAIT": "blue"}.get(r["상태"], "gray")
                                    ui.badge(r["상태"], color=_rc).classes("text-xs")
                            p_color = "text-red-400" if r["수익률"] > 0 else "text-blue-400"
                            _psrc = r.get("가격소스", "")
                            ui.label(f"{r['수익률']:+.2f}%  |  현재가: {int(r['현재가']):,}{_psrc}  |  평가금: {int(r['평가금']):,}원").classes(f"text-sm {p_color}")
                        with ui.column().classes("items-end gap-0"):
                            ui.label(r["AI조언"]).classes(f"text-sm font-bold").style(f"color:{r['색상']}")
                            if r["점수"] > 0:
                                _src_tag = f" ({r['소스']})" if r.get("소스") != "금일추천" else ""
                                ui.label(f"점수: {r['점수']:.0f}{_src_tag}").classes("text-xs text-gray-400")

            if pf_rows:
                pie_data = pf_rows.copy()
                if cash_amt > 0:
                    pie_data.append({"종목명": "현금", "평가금": cash_amt})
                fig = px.pie(pd.DataFrame(pie_data), values="평가금", names="종목명", title="📊 자산 구성", hole=0.4)
                ui.plotly(_plotly_dark(fig, 300)).classes("w-full")

            if KELLY_OK and pf_rows:
                kelly_section = ui.card().classes("w-full p-4 bg-[#1a1a2e] border border-yellow-700/40 rounded-xl mt-4")
                render_portfolio_kelly_summary(pf_rows, total_eval, kelly_section)

    ui.button("🤖 AI 진단 실행", on_click=analyze).classes("mt-4").props("color=primary")
