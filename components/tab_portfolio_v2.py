# -*- coding: utf-8 -*-
"""
tab_portfolio_v2.py — 💼 내 자산: AI 리밸런싱 & DART 공시 진단 (Phase 2)
═══════════════════════════════════════════════════════════════════
기존 tab_portfolio.py + DART 공시 분석 통합

[신규 기능]
 1. 보유 종목별 최근 DART 공시 자동 조회 + Gemini AI 재무 리스크 진단
 2. 종합 포트폴리오 리스크 리포트 (섹터 집중도, 변동성, 공시 리스크)
 3. AI 기반 리밸런싱 제안 (비중 조정 / 교체 종목 추천)

통합 방법: 기존 tab_portfolio.py를 이 파일로 교체
  from components.tab_portfolio_v2 import render_tab_portfolio
"""

import asyncio
import glob
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# ── DART 분석기 통합 ──
try:
    from dart_analyzer import DartAnalyzer, DART_OK as _DART_LIB_OK, GEMINI_OK as _GEMINI_LIB_OK
    DART_INTEGRATION_OK = True
except ImportError:
    DART_INTEGRATION_OK = False
    _DART_LIB_OK = False
    _GEMINI_LIB_OK = False

# ── Gemini 직접 호출 (포트폴리오 종합 진단용) ──
_GENAI_CLIENT = None
try:
    from google import genai
    from google.genai import types as genai_types
    _api_key = os.environ.get("GEMINI_API_KEY", "")
    if _api_key:
        _GENAI_CLIENT = genai.Client(api_key=_api_key)
except ImportError:
    genai = None
    genai_types = None

_logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

try:
    from version_info import PRICE_PRIME
except ImportError:
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
#  데이터 유틸 (기존 tab_portfolio.py에서 이식)
# ══════════════════════════════════════════════════════

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
#  [Phase 2 신규] DART 공시 AI 진단
# ══════════════════════════════════════════════════════

def _get_dart_cache_key() -> str:
    """오늘 날짜 기반 캐시 키."""
    return f"dart_diag_{datetime.now().strftime('%Y%m%d')}"


def _load_dart_cache() -> dict:
    """app.storage.user에서 DART 진단 캐시 로드. 오늘 날짜면 유효."""
    try:
        cached = app.storage.user.get("dart_cache", {})
        if cached.get("_date") == datetime.now().strftime("%Y%m%d"):
            return cached.get("results", {})
    except Exception:
        pass
    return {}


def _save_dart_cache(results: dict):
    """DART 진단 결과를 캐시에 저장 (하루 TTL)."""
    try:
        app.storage.user["dart_cache"] = {
            "_date": datetime.now().strftime("%Y%m%d"),
            "results": results,
        }
    except Exception as e:
        _logger.debug(f"DART 캐시 저장 실패: {e}")


def _run_dart_diagnosis(code: str, name: str) -> dict:
    """개별 종목 DART 공시 조회 + Gemini AI 분석.

    Returns:
        {
            "has_disclosure": bool,
            "disclosures": [{"report_nm": str, "score": float, "reason": str}],
            "summary_score": float,  # -10 ~ +10
            "risk_level": str,       # "🟢안전" | "🟡주의" | "🔴위험"
        }
    """
    result = {
        "has_disclosure": False,
        "disclosures": [],
        "summary_score": 0.0,
        "risk_level": "🟢안전",
    }

    if not DART_INTEGRATION_OK:
        return result

    try:
        analyzer = DartAnalyzer()
        if not analyzer.dart:
            return result

        disclosures = analyzer.get_major_disclosures(str(code).zfill(6), days=7)
        if not disclosures:
            return result

        result["has_disclosure"] = True
        scores = []

        for disc in disclosures[:5]:  # 최대 5건
            rcept_no = disc.get("rcept_no", "")
            report_nm = disc.get("report_nm", "")

            if analyzer._has_gemini:
                score, reason = analyzer.analyze_report(rcept_no, report_nm)
            else:
                score, reason = 0.0, f"[공시감지] {report_nm}"

            result["disclosures"].append({
                "report_nm": report_nm,
                "rcept_dt": disc.get("rcept_dt", ""),
                "score": score,
                "reason": reason,
            })
            scores.append(score)

        if scores:
            # 최대 임팩트 기반 (절대값이 가장 큰 점수)
            result["summary_score"] = max(scores, key=abs)

        s = result["summary_score"]
        if s <= -5:
            result["risk_level"] = "🔴위험"
        elif s <= -2:
            result["risk_level"] = "🟡주의"
        elif s >= 5:
            result["risk_level"] = "🟢호재"
        else:
            result["risk_level"] = "🟢안전"

    except Exception as e:
        _logger.error(f"DART 진단 오류 ({name}/{code}): {e}")

    return result


def _generate_portfolio_report(pf_rows: list, dart_results: dict,
                                total_eval: float, cash_amt: float) -> str:
    """Gemini를 사용한 종합 포트폴리오 AI 리포트 생성.

    Returns:
        AI 생성 한국어 리포트 텍스트
    """
    if _GENAI_CLIENT is None:
        return _generate_fallback_report(pf_rows, dart_results, total_eval, cash_amt)

    # 포트폴리오 요약 데이터 구성
    portfolio_summary = []
    for r in pf_rows:
        code = r.get("code", "")
        dart = dart_results.get(code, {})
        portfolio_summary.append({
            "종목명": r["종목명"],
            "비중": f"{r['평가금'] / total_eval * 100:.1f}%" if total_eval > 0 else "0%",
            "수익률": f"{r['수익률']:+.2f}%",
            "시스템점수": r["점수"],
            "시스템상태": r["상태"],
            "공시리스크": dart.get("risk_level", "미조회"),
            "공시요약": "; ".join([d.get("reason", "") for d in dart.get("disclosures", [])[:2]]),
        })

    import json
    pf_json = json.dumps(portfolio_summary, ensure_ascii=False, indent=2)
    cash_pct = cash_amt / (total_eval + cash_amt) * 100 if (total_eval + cash_amt) > 0 else 0

    prompt = f"""당신은 대한민국 전문 자산관리사(CFA)입니다.
아래 고객의 포트폴리오를 분석하여 종합 진단 리포트를 작성하세요.

[포트폴리오 현황]
총 평가금: {int(total_eval):,}원
현금 비중: {cash_pct:.1f}%
보유 종목:
{pf_json}

[분석 항목 — 각 항목을 명확하게 구분하여 작성]
1. 📊 포트폴리오 종합 진단 (2~3문장)
2. ⚠️ 리스크 요인 (공시 리스크, 섹터 집중도, 변동성)
3. 💡 리밸런싱 제안 (비중 조정, 교체/추가 고려 종목)
4. 🎯 향후 1주 액션 플랜

한국어로 작성하되, 구체적인 수치와 근거를 포함하세요.
총 400자 이내로 요약하세요."""

    try:
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        response = _GENAI_CLIENT.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=1024,
            ),
        )
        return response.text.strip() if response.text else _generate_fallback_report(
            pf_rows, dart_results, total_eval, cash_amt)
    except Exception as e:
        _logger.error(f"포트폴리오 AI 리포트 생성 실패: {e}")
        return _generate_fallback_report(pf_rows, dart_results, total_eval, cash_amt)


def _generate_fallback_report(pf_rows, dart_results, total_eval, cash_amt):
    """Gemini 사용 불가 시 규칙 기반 진단 리포트."""
    lines = ["📊 포트폴리오 규칙 기반 진단\n"]

    # 수익률 분석
    returns = [r["수익률"] for r in pf_rows if r["수익률"] != 0]
    if returns:
        avg_ret = sum(returns) / len(returns)
        lines.append(f"• 평균 수익률: {avg_ret:+.2f}%")
        loss_count = sum(1 for r in returns if r < 0)
        lines.append(f"• 손실 종목: {loss_count}/{len(pf_rows)}개")

    # 공시 리스크
    risk_stocks = []
    for r in pf_rows:
        dart = dart_results.get(r.get("code", ""), {})
        if dart.get("summary_score", 0) <= -3:
            risk_stocks.append(f"{r['종목명']}({dart['risk_level']})")
    if risk_stocks:
        lines.append(f"\n⚠️ 공시 리스크 종목: {', '.join(risk_stocks)}")

    # 현금 비중
    total = total_eval + cash_amt
    if total > 0:
        cash_pct = cash_amt / total * 100
        if cash_pct < 10:
            lines.append(f"\n💡 현금 비중({cash_pct:.1f}%)이 낮습니다. 10~20% 유지를 권장합니다.")

    # 집중도
    if pf_rows and total_eval > 0:
        max_weight = max(r["평가금"] / total_eval * 100 for r in pf_rows)
        if max_weight > 40:
            top_stock = max(pf_rows, key=lambda x: x["평가금"])
            lines.append(f"⚠️ {top_stock['종목명']} 비중 {max_weight:.1f}% — 과도 집중, 분산 필요")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════
#  메인 렌더
# ══════════════════════════════════════════════════════

def render_tab_portfolio(df, auth):
    """Tab 3: 내 자산 (포트폴리오 AI 진단 — Phase 2 통합)"""

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
                    <div style="color:#9CA3AF; font-size:13px; margin-top:4px;">DART 공시 분석</div>
                </div>
                <div style="text-align:center; padding:16px; border:1px solid #374151; border-radius:12px; min-width:100px;">
                    <div style="font-size:24px;">⚡</div>
                    <div style="color:#9CA3AF; font-size:13px; margin-top:4px;">재무 리스크 진단</div>
                </div>
                """)
            ui.button(
                "💎 멤버십 업그레이드 알아보기",
                on_click=lambda: ui.run_javascript(
                    "document.querySelector('[role=tab]:nth-child(4)')?.click()"
                ),
            ).classes("mt-4").props("color=primary rounded size=lg")
        return

    _section_title("💼 내 자산: AI 리밸런싱 & DART 공시 진단")
    ui.label("👇 보유 종목을 입력하세요 (종목명:평단가:수량)").classes("text-xs text-gray-400 mb-2")

    # DART 연동 상태 표시
    dart_status = "🟢 DART+AI 연동" if (DART_INTEGRATION_OK and _GENAI_CLIENT) else \
                  "🟡 DART만 연동" if DART_INTEGRATION_OK else "⚪ DART 미연결"
    ui.label(dart_status).classes("text-xs text-gray-500 mb-2")

    saved_local = app.storage.user.get("portfolio_text", "")
    saved_gist = _load_portfolio_file() if not saved_local else ""
    saved = saved_local or saved_gist or ""

    pf_input = ui.textarea("포트폴리오 입력", value=saved,
                           placeholder="종목명:평단가:수량 (줄바꿈 구분)\n예) 에코프로머티:67341:60").classes("w-full").props("rows=6")
    result_area = ui.column().classes("w-full mt-4")

    # ── DART 공시 분석 토글 ──
    dart_toggle = ui.checkbox("📜 DART 공시 분석 포함", value=True).classes("text-gray-300")

    def _auto_save():
        app.storage.user["portfolio_text"] = pf_input.value
    pf_input.on("blur", lambda _: _auto_save())

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

        # ── 비동기 현재가 조회 ──
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
                tasks_list = [
                    loop.run_in_executor(_io_pool, _fetch_current_price, t[0], t[1])
                    for t in targets
                ]
                results = await asyncio.gather(*tasks_list, return_exceptions=True)
                for res in results:
                    if isinstance(res, tuple) and len(res) == 3:
                        c, n, p = res
                        price_map[c] = p

        total_eval = total_buy = 0.0
        pf_rows = []
        for code, name, avg, qty in targets:
            curr = price_map.get(code, 0)

            # 폴백 체인 (기존과 동일)
            if curr == 0 and not df.empty and '종가' in df.columns:
                match_p = df[df['종목코드'] == str(code).zfill(6)] if '종목코드' in df.columns else pd.DataFrame()
                if match_p.empty and '종목명' in df.columns:
                    match_p = df[df['종목명'] == name]
                if not match_p.empty:
                    curr = int(nz_num(match_p.iloc[0].get('종가', 0)))

            if curr == 0:
                _ensure_hist_cache()
                hist = _hist_recommend_cache.get(str(code).zfill(6))
                if hist and hist.get("종가", 0) > 0:
                    curr = int(hist["종가"])

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

        # ═══════════════════════════════════════════
        # [Phase 2] DART 공시 진단 실행 (하루 1회 캐싱)
        # ═══════════════════════════════════════════
        dart_results = {}
        if dart_toggle.value and DART_INTEGRATION_OK:
            # 캐시 로드 — 오늘 이미 조회했으면 API 호출 스킵
            cached = _load_dart_cache()
            uncached_rows = []
            for r in pf_rows:
                code = r["code"]
                if code in cached:
                    dart_results[code] = cached[code]
                else:
                    uncached_rows.append(r)

            if cached and not uncached_rows:
                ui.notify("📜 DART 캐시 적용 (오늘 이미 분석됨)", type="info")
            elif uncached_rows:
                with result_area:
                    result_area.clear()
                    cache_msg = f" (캐시 {len(pf_rows) - len(uncached_rows)}건)" if cached else ""
                    ui.label(f"📜 DART 공시 분석 중...{cache_msg}").classes("text-gray-400")
                    dart_progress = ui.linear_progress(value=0).classes("w-full")

                for i, r in enumerate(uncached_rows):
                    code = r["code"]
                    try:
                        dart_results[code] = await run_sync(
                            _run_dart_diagnosis, code, r["종목명"]
                        )
                    except Exception as e:
                        _logger.error(f"DART 진단 실패 ({r['종목명']}): {e}")
                        dart_results[code] = {"has_disclosure": False, "disclosures": [],
                                              "summary_score": 0, "risk_level": "⚪미조회"}
                    dart_progress.set_value((i + 1) / len(uncached_rows))

                # 전체 결과 캐시 저장
                _save_dart_cache(dart_results)

        # ═══════════════════════════════════════════
        # 결과 렌더링
        # ═══════════════════════════════════════════
        result_area.clear()
        with result_area:
            total_asset = total_eval + cash_amt
            total_invest = total_buy + cash_amt
            total_rate = (total_asset - total_invest) / total_invest * 100 if total_invest > 0 else 0

            with ui.row().classes("w-full gap-4 flex-wrap"):
                _metric_card("총 평가금액", f"{int(total_asset):,}원")
                _metric_card("총 매입금액", f"{int(total_invest):,}원")
                _metric_card("총 평가손익", f"{int(total_asset - total_invest):+,}원",
                             f"{total_rate:+.2f}%", total_rate >= 0)
                if cash_amt > 0:
                    _metric_card("현금 비중",
                                 f"{cash_amt/total_asset*100:.1f}%" if total_asset > 0 else "0%",
                                 f"{int(cash_amt):,}원")

            # ── 종목별 카드 (DART 통합) ──
            _section_title("🩺 AI 포트폴리오 진단")
            pf_rows.sort(key=lambda x: x["점수"])
            for r in pf_rows:
                code = r["code"]
                dart = dart_results.get(code, {})

                with ui.card().classes("w-full p-4 mb-2 bg-[#1a1a2e] border border-gray-700 rounded-xl"):
                    with ui.row().classes("w-full justify-between items-center"):
                        with ui.column().classes("gap-0"):
                            with ui.row().classes("items-center gap-2"):
                                ui.label(r["종목명"]).classes("text-white font-bold")
                                if r.get("상태"):
                                    _rc = {"ATTACK": "red", "ARMED": "orange", "WAIT": "blue"}.get(r["상태"], "gray")
                                    ui.badge(r["상태"], color=_rc).classes("text-xs")
                                # DART 리스크 배지
                                if dart.get("has_disclosure"):
                                    risk = dart.get("risk_level", "")
                                    dart_color = "red" if "위험" in risk else "orange" if "주의" in risk else "green"
                                    ui.badge(f"📜{risk}", color=dart_color).classes("text-xs")

                            p_color = "text-red-400" if r["수익률"] > 0 else "text-blue-400"
                            _psrc = r.get("가격소스", "")
                            ui.label(
                                f"{r['수익률']:+.2f}%  |  현재가: {int(r['현재가']):,}{_psrc}  |  평가금: {int(r['평가금']):,}원"
                            ).classes(f"text-sm {p_color}")

                        with ui.column().classes("items-end gap-0"):
                            ui.label(r["AI조언"]).classes("text-sm font-bold").style(f"color:{r['색상']}")
                            if r["점수"] > 0:
                                _src_tag = f" ({r['소스']})" if r.get("소스") != "금일추천" else ""
                                ui.label(f"점수: {r['점수']:.0f}{_src_tag}").classes("text-xs text-gray-400")

                    # ── DART 공시 상세 (접이식) ──
                    if dart.get("disclosures"):
                        with ui.expansion(f"📜 공시 {len(dart['disclosures'])}건 (점수: {dart.get('summary_score', 0):+.1f})").classes("w-full mt-2"):
                            for disc in dart["disclosures"]:
                                s = disc.get("score", 0)
                                s_color = "text-green-400" if s > 0 else "text-red-400" if s < 0 else "text-gray-400"
                                with ui.row().classes("w-full py-1 border-b border-gray-800 items-center gap-2"):
                                    ui.label(f"[{disc.get('rcept_dt', '')}]").classes("text-xs text-gray-500 w-20")
                                    ui.label(disc.get("report_nm", "")).classes("text-sm text-white flex-1")
                                    ui.label(f"{s:+.1f}").classes(f"text-sm font-bold {s_color}")
                                if disc.get("reason"):
                                    ui.label(f"  → {disc['reason']}").classes("text-xs text-gray-400 ml-20 mb-1")

            # ── 자산 구성 파이 차트 ──
            if pf_rows:
                pie_data = pf_rows.copy()
                if cash_amt > 0:
                    pie_data.append({"종목명": "현금", "평가금": cash_amt})
                fig = px.pie(pd.DataFrame(pie_data), values="평가금", names="종목명",
                             title="📊 자산 구성", hole=0.4)
                ui.plotly(_plotly_dark(fig, 300)).classes("w-full")

            # ═══════════════════════════════════════
            # [Phase 2] 종합 AI 리포트
            # ═══════════════════════════════════════
            if pf_rows and (dart_results or _GENAI_CLIENT):
                _section_title("🤖 AI 종합 포트폴리오 리포트")
                with ui.card().classes("w-full p-6 bg-gradient-to-br from-[#1a1a2e] to-[#0f3460] "
                                       "border border-blue-700/40 rounded-xl"):
                    report_area = ui.column().classes("w-full")
                    with report_area:
                        ui.label("🧠 AI 분석 생성 중...").classes("text-gray-400")
                        ui.spinner("dots", size="lg", color="blue")

                    async def _gen_report():
                        report_text = await run_sync(
                            _generate_portfolio_report, pf_rows, dart_results,
                            total_eval, cash_amt
                        )
                        report_area.clear()
                        with report_area:
                            ui.markdown(report_text).classes("text-white text-sm leading-relaxed")

                    # 비동기로 리포트 생성 (UI 블로킹 방지)
                    asyncio.ensure_future(_gen_report())

            # ── Kelly 비중 분석 ──
            if KELLY_OK and pf_rows:
                kelly_section = ui.card().classes(
                    "w-full p-4 bg-[#1a1a2e] border border-yellow-700/40 rounded-xl mt-4")
                render_portfolio_kelly_summary(pf_rows, total_eval, kelly_section)

    ui.button("🤖 AI 진단 실행", on_click=analyze).classes("mt-4").props("color=primary")
