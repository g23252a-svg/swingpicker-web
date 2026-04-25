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
import hashlib
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nicegui import ui, app

from shared_utils import nz_num, safe_float

# ═══════════════════════════════════════════════════
# [v22 UI Step N] 사용자 식별 — 로그인 정보 import
# ═══════════════════════════════════════════════════
try:
    from services.auth import get_current_user
except Exception:
    def get_current_user():
        return app.storage.user.get("profile")

# ═══════════════════════════════════════════════════
# [v22 UI Step L] 공통 용어 사전 import (시장/종목 탭과 동일 패턴)
# 배포 중 import 경로 꼬여도 화면 죽지 않게 fallback 제공
# ═══════════════════════════════════════════════════
try:
    from components.ui_terms import (
        route_display,
        route_icon,
        is_truthy_flag,
    )
except Exception as _ui_terms_err:
    logging.getLogger(__name__).warning(
        f"ui_terms import 실패, fallback 사용: {_ui_terms_err}"
    )
    def route_display(x):
        _map = {"ATTACK": "🚀 적극 매수", "ARMED": "🎯 매수 준비",
                "WAIT": "⏸️ 관망", "NEUTRAL": "👁️ 중립",
                "CARRY": "📌 보유 관리", "OVERHEAT": "🔥 과열 주의",
                "EXIT_WARNING": "⚠️ 이탈 주의", "BLOCKED": "⛔ 제외"}
        return _map.get(str(x or "").strip().upper(), str(x or ""))
    def route_icon(x):
        _icons = {"ATTACK": "🚀", "ARMED": "🎯", "WAIT": "⏸️",
                  "NEUTRAL": "👁️", "CARRY": "📌"}
        return _icons.get(str(x or "").strip().upper(), "👀")
    def is_truthy_flag(v):
        if v is None: return False
        return str(v).strip().upper() in {"1", "1.0", "TRUE", "Y", "YES"}

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
def _get_user_portfolio_key(user_profile=None) -> str:
    """[Step N] 로그인 사용자별 포트폴리오 파일명 생성.
    
    이메일/id를 SHA-256 해시 (12자리) → 파일명 보안성 확보
    
    Args:
        user_profile: get_current_user() 반환값 (dict). None이면 자동 조회.
    
    Returns:
        "portfolio_<hash12>.txt" — 사용자별 고유 파일명
        로그인 정보 없으면 "portfolio.txt" (기존 동작 유지 = 호환성)
    """
    if user_profile is None:
        try:
            user_profile = get_current_user()
        except Exception:
            user_profile = None
    
    if not user_profile:
        return "portfolio.txt"  # 비로그인 fallback (기존 동작)
    
    # 우선순위: email > id > username
    user_id = (
        str(user_profile.get("email", "")).strip()
        or str(user_profile.get("id", "")).strip()
        or str(user_profile.get("username", "")).strip()
    )
    if not user_id:
        return "portfolio.txt"
    
    h = hashlib.sha256(user_id.lower().encode("utf-8")).hexdigest()[:12]
    return f"portfolio_{h}.txt"


def _load_portfolio_file(user_profile=None):
    """[Step N] 계정별 포트폴리오 로드 (Gist).
    
    동작:
      1. 사용자별 파일명으로 먼저 조회
      2. 없으면 기존 portfolio.txt fallback (마이그레이션용)
      3. fallback에서 로드되면 다음 저장 시 자동으로 사용자 파일에 저장됨
    
    Returns:
        (content, source) 튜플
        content: 포트폴리오 텍스트
        source: "user" (사용자 파일) / "legacy" (구 portfolio.txt) / ""
    """
    token = os.environ.get("LDY_GIST_TOKEN", "")
    gist_id = os.environ.get("LDY_GIST_ID", "")
    if not token or not gist_id:
        return "", ""
    try:
        import requests
        r = requests.get(f"https://api.github.com/gists/{gist_id}",
                        headers={"Authorization": f"token {token}"}, timeout=10)
        if r.ok:
            files = r.json().get("files", {})
            user_filename = _get_user_portfolio_key(user_profile)
            
            # 1순위: 사용자별 파일
            if user_filename in files and user_filename != "portfolio.txt":
                content = files[user_filename].get("content", "")
                if content:
                    return content, "user"
            
            # 2순위: 기존 portfolio.txt (마이그레이션용 fallback)
            if "portfolio.txt" in files:
                content = files["portfolio.txt"].get("content", "")
                if content:
                    return content, "legacy"
    except Exception as e:
        _logger.warning(f"포트폴리오 로드 실패: {e}")
    return "", ""


def _save_portfolio_file(text_data, user_profile=None):
    """[Step N] 계정별 포트폴리오 저장 (Gist).
    
    Returns:
        (success, filename) 튜플 — UI에 저장 상태 표시용
    """
    token = os.environ.get("LDY_GIST_TOKEN", "")
    gist_id = os.environ.get("LDY_GIST_ID", "")
    if not token or not gist_id:
        return False, ""
    
    filename = _get_user_portfolio_key(user_profile)
    try:
        import requests
        r = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {token}"},
            json={"files": {filename: {"content": text_data}}},
            timeout=10,
        )
        return r.ok, filename
    except Exception as e:
        _logger.warning(f"포트폴리오 저장 실패: {e}")
        return False, filename


def _get_user_meta_filename(user_profile=None) -> str:
    """[Step O #3] 사용자별 메타 파일명 (포트폴리오 파일과 짝)."""
    portfolio_fn = _get_user_portfolio_key(user_profile)
    if portfolio_fn == "portfolio.txt":
        return ""  # 비로그인은 메타 X
    # portfolio_<hash>.txt → portfolio_<hash>_meta.json
    return portfolio_fn.replace(".txt", "_meta.json")


def _load_portfolio_meta(user_profile=None) -> str:
    """[Step O #3] 사용자별 포트폴리오 메타 로드 → updated_at 문자열.
    
    메타 파일이 없거나 파싱 실패 시 빈 문자열 반환.
    """
    if not user_profile:
        return ""
    meta_fn = _get_user_meta_filename(user_profile)
    if not meta_fn:
        return ""
    
    token = os.environ.get("LDY_GIST_TOKEN", "")
    gist_id = os.environ.get("LDY_GIST_ID", "")
    if not token or not gist_id:
        return ""
    try:
        import requests, json
        r = requests.get(f"https://api.github.com/gists/{gist_id}",
                        headers={"Authorization": f"token {token}"}, timeout=10)
        if r.ok:
            files = r.json().get("files", {})
            if meta_fn in files:
                content = files[meta_fn].get("content", "")
                if content:
                    meta = json.loads(content)
                    return str(meta.get("updated_at", ""))
    except Exception as e:
        _logger.warning(f"포트폴리오 메타 로드 실패: {e}")
    return ""


def _save_portfolio_meta(user_profile, updated_at: str) -> bool:
    """[Step O #3] 사용자별 포트폴리오 메타 저장 → updated_at 등."""
    if not user_profile:
        return False
    meta_fn = _get_user_meta_filename(user_profile)
    if not meta_fn:
        return False
    
    token = os.environ.get("LDY_GIST_TOKEN", "")
    gist_id = os.environ.get("LDY_GIST_ID", "")
    if not token or not gist_id:
        return False
    try:
        import requests, json
        meta_content = json.dumps({
            "updated_at": updated_at,
            "version": 1,
        }, ensure_ascii=False, indent=2)
        r = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {token}"},
            json={"files": {meta_fn: {"content": meta_content}}},
            timeout=10,
        )
        return r.ok
    except Exception as e:
        _logger.warning(f"포트폴리오 메타 저장 실패: {e}")
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

# ══════════════════════════════════════════════════════
# [v22 UI Step L] Hero 카드 + 진단 요약 헬퍼
# 첫 화면 1초 답변 + 진단 결과 결론성 강화
# 시장/종목 탭과 동일한 결론 → 카드 → 상세 패턴
# ══════════════════════════════════════════════════════

def _parse_portfolio_text(text: str) -> tuple:
    """저장된 포트폴리오 텍스트를 (종목 리스트, 현금 합계)로 파싱.
    
    Returns:
        (items, cash_amt)
        items: [{"name": str, "avg": int, "qty": int}, ...]
        cash_amt: float (CASH/현금 라인의 합계)
    """
    items = []
    cash_amt = 0.0
    if not text:
        return items, cash_amt
    for line in text.split("\n"):
        if ":" not in line:
            continue
        parts = line.split(":")
        if len(parts) < 3:
            continue
        try:
            nm = parts[0].strip()
            avg = int(float(parts[1].replace(",", "").strip()))
            qty = int(float(parts[2].replace(",", "").strip()))
        except (ValueError, IndexError, TypeError):
            continue
        if nm.upper() == "CASH" or "현금" in nm:
            cash_amt += avg * qty
        else:
            items.append({"name": nm, "avg": avg, "qty": qty})
    return items, cash_amt


def _render_portfolio_hero(saved_text: str, df: pd.DataFrame):
    """[v22 UI Step L] 내자산 탭 Hero 카드 — 진입 시 1초 답변
    
    저장된 포트폴리오가 있으면:
      - 종목 수, 매입 합계, 평단 기반 추정 평가금
      - 시스템 추천에 들어와있는 종목 수 (오늘 추천 / 관찰 대상)
      - 즉시 액션 필요 종목 수 (점수 ≤ 40)
    
    저장된 포트폴리오가 없으면:
      - 온보딩 카드 (입력하면 무엇을 받을 수 있는지)
    
    안전: try/except로 카드 실패해도 진단 화면은 정상
    """
    try:
        items, cash_amt = _parse_portfolio_text(saved_text)
        n_stocks = len(items)
        
        # ─────────────────────────────────────────────
        # 빈 상태 (보유 종목 없음) → 온보딩
        # ─────────────────────────────────────────────
        if n_stocks == 0 and cash_amt == 0:
            with ui.card().classes(
                "w-full p-5 mb-4 rounded-xl border-2 border-blue-500/40"
            ).style(
                "background: linear-gradient(to right, #0a1a3d, #0d2a5b, #0a1a3d)"
            ):
                with ui.row().classes("w-full items-center gap-4"):
                    ui.icon("savings", size="48px").classes("text-blue-300")
                    with ui.column().classes("gap-1 flex-1"):
                        ui.label("💼 보유 종목을 입력하면").classes(
                            "text-base font-bold text-blue-300"
                        )
                        ui.label(
                            "✓ 매일 자동 AI 진단  ·  ✓ DART 공시 리스크 알림  ·  "
                            "✓ AI 종합 리포트  ·  ✓ 적정 비중 제안"
                        ).classes("text-xs text-blue-100")
            return
        
        # ─────────────────────────────────────────────
        # 보유 종목 있음 → 빠른 요약 (정확한 평가금은 진단 후)
        # ─────────────────────────────────────────────
        total_buy = sum(i["avg"] * i["qty"] for i in items)
        total_with_cash = total_buy + cash_amt
        
        # 시스템 추천 매칭 — 빠른 lookup (현재 df 기준)
        n_today = 0   # 오늘 추천에 있는 종목
        n_caution = 0 # 점수 ≤ 40 (교체 검토)
        n_observe = 0 # 점수 41~59 (모니터링)
        n_hold = 0    # 점수 ≥ 60 (양호)
        
        # [Step M #2 + O #2] 분류 (시장/종목 탭과 일관성)
        # 보유 유지: 점수≥60 + 안전 ROUTE (ATTACK/ARMED/CARRY/NEUTRAL)
        # 지켜보기: 점수 41~59 또는 ROUTE=WAIT
        # 교체 검토: 점수≤40 또는 BLOCKED/EXIT_WARNING/OVERHEAT
        # 
        # [Step O #2] TOP_PICK 기준 엄격화:
        #   n_today = 오늘 분석 데이터에 포함 (df에 있음)
        #   n_top_pick = 진짜 오늘의 추천 (TOP_PICK == 1)
        today_picks_in_holdings = []  # 오늘 추천(TOP_PICK)에 포함된 보유종목
        n_top_pick = 0  # [Step O #2] TOP_PICK == 1 만 카운트
        
        if df is not None and not df.empty and "종목명" in df.columns:
            df_names = set(df["종목명"].astype(str))
            _SAFE_ROUTES = {"ATTACK", "ARMED", "CARRY", "NEUTRAL"}
            _BLOCKED_ROUTES = {"BLOCKED", "EXIT_WARNING", "OVERHEAT"}
            for it in items:
                if it["name"] in df_names:
                    n_today += 1
                    match = df[df["종목명"] == it["name"]]
                    if not match.empty:
                        row = match.iloc[0]
                        score = safe_float(row.get("DISPLAY_SCORE", 0))
                        route = str(row.get("ROUTE", "")).strip().upper()
                        # [Step O #2] TOP_PICK 정확히 판정
                        is_top_pick = is_truthy_flag(row.get("TOP_PICK", 0))
                        if is_top_pick:
                            n_top_pick += 1
                            today_picks_in_holdings.append(it["name"])
                        
                        # 분류: 점수 + 상태 동시 고려
                        if route in _BLOCKED_ROUTES or (score > 0 and score <= 40):
                            n_caution += 1   # 교체 검토
                        elif route == "WAIT" or (score > 0 and score < 60):
                            n_observe += 1   # 지켜보기
                        elif score >= 60 and route in _SAFE_ROUTES:
                            n_hold += 1      # 보유 유지
                        else:
                            n_observe += 1   # 정보 부족 → 지켜보기
        
        # 헤더 카드 — 결론 한 줄
        with ui.card().classes(
            "w-full p-5 mb-4 rounded-xl border-2 border-cyan-500/50"
        ).style(
            "background: linear-gradient(to right, #0a2a3d, #0d4054, #0a2a3d)"
        ):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.column().classes("gap-1"):
                    ui.label(f"💼 보유 자산 현황").classes(
                        "text-lg font-bold text-cyan-300"
                    )
                    parts = [f"종목 {n_stocks}개"]
                    if total_buy > 0:
                        parts.append(f"매입 {int(total_buy):,}원")
                    if cash_amt > 0:
                        cash_pct = cash_amt / total_with_cash * 100 if total_with_cash > 0 else 0
                        parts.append(f"현금 {cash_pct:.0f}%")
                    ui.label("  ·  ".join(parts)).classes("text-sm text-cyan-100")
                with ui.column().classes("items-end gap-0"):
                    ui.label(f"{int(total_with_cash):,}원").classes(
                        "text-2xl font-black text-cyan-300"
                    )
                    # [Step M #1] 임시 요약 명시
                    ui.label("🔍 시세 조회 전 · 매입금 기준").classes(
                        "text-[10px] text-amber-400/80"
                    )
            # [Step M #1] 카드 하단에 추가 안내
            ui.label(
                "AI 진단 실행 후 정확한 평가금이 표시됩니다"
            ).classes("text-[10px] text-gray-500 italic mt-2")
        
        # [Step M #5 + O #2] 오늘 추천 (TOP_PICK) 포함 보유종목 미리보기
        # n_top_pick = TOP_PICK==1 만 카운트 (엄격)
        # n_today = 오늘 분석 데이터에 포함 (느슨)
        if n_top_pick > 0 and today_picks_in_holdings:
            preview = today_picks_in_holdings[:3]
            more = len(today_picks_in_holdings) - 3
            preview_text = ", ".join(preview)
            if more > 0:
                preview_text += f" 외 {more}종목"
            with ui.card().classes(
                "w-full p-3 mb-3 bg-[#0d1a14] "
                "border border-emerald-700/30 rounded-lg"
            ):
                ui.label("📌 오늘의 추천에 포함된 보유종목").classes(
                    "text-xs text-emerald-300 font-bold"
                )
                # [Step O #2] 정확한 표현: TOP_PICK 종목 / 오늘 분석 포함 / 전체
                _detail_text = f"{preview_text}  ·  오늘의 추천 {n_top_pick}개"
                if n_today > n_top_pick:
                    _detail_text += f" (오늘 분석 포함 {n_today}개 중)"
                _detail_text += f"  ·  전체 보유 {n_stocks}개"
                ui.label(_detail_text).classes(
                    "text-sm text-emerald-100 mt-1"
                )
        elif n_today > 0:
            # TOP_PICK 0이지만 분석에는 포함된 케이스
            with ui.card().classes(
                "w-full p-3 mb-3 bg-[#1a1408] "
                "border border-amber-700/30 rounded-lg"
            ):
                ui.label("ℹ️ 분석 데이터에 포함된 보유종목").classes(
                    "text-xs text-amber-300 font-bold"
                )
                ui.label(
                    f"분석 포함 {n_today}/{n_stocks}개  ·  "
                    f"오늘의 추천(TOP_PICK)에는 0개"
                ).classes("text-sm text-amber-100 mt-1")
        
        # 빠른 진단 요약 (df에 있는 종목들만)
        if n_today > 0:
            with ui.row().classes("w-full gap-3 mb-4 flex-wrap"):
                # 즉시 액션 필요 (교체 검토)
                if n_caution > 0:
                    with ui.card().classes(
                        "flex-1 min-w-[200px] p-3 rounded-lg "
                        "border-l-4 border-red-500"
                    ).style("background: #1a0a14"):
                        ui.label("🚨 즉시 액션 필요").classes(
                            "text-xs text-red-300 font-bold"
                        )
                        ui.label(f"{n_caution}종목").classes(
                            "text-2xl font-black text-red-400"
                        )
                        ui.label("점수 40 이하 — 교체 검토").classes(
                            "text-[10px] text-gray-400"
                        )
                
                # 모니터링
                if n_observe > 0:
                    with ui.card().classes(
                        "flex-1 min-w-[200px] p-3 rounded-lg "
                        "border-l-4 border-amber-500"
                    ).style("background: #1a1408"):
                        ui.label("⚠️ 모니터링").classes(
                            "text-xs text-amber-300 font-bold"
                        )
                        ui.label(f"{n_observe}종목").classes(
                            "text-2xl font-black text-amber-400"
                        )
                        ui.label("점수 41~59 — 지켜보기").classes(
                            "text-[10px] text-gray-400"
                        )
                
                # 양호
                if n_hold > 0:
                    with ui.card().classes(
                        "flex-1 min-w-[200px] p-3 rounded-lg "
                        "border-l-4 border-emerald-500"
                    ).style("background: #0a1a14"):
                        ui.label("✅ 양호").classes(
                            "text-xs text-emerald-300 font-bold"
                        )
                        ui.label(f"{n_hold}종목").classes(
                            "text-2xl font-black text-emerald-400"
                        )
                        ui.label("점수 60 이상 — 보유 유지").classes(
                            "text-[10px] text-gray-400"
                        )
                
                # 추천 외 종목
                n_outside = n_stocks - n_today
                if n_outside > 0:
                    with ui.card().classes(
                        "flex-1 min-w-[200px] p-3 rounded-lg "
                        "border-l-4 border-gray-600"
                    ).style("background: #14141a"):
                        ui.label("ℹ️ 추천 외").classes(
                            "text-xs text-gray-400 font-bold"
                        )
                        ui.label(f"{n_outside}종목").classes(
                            "text-2xl font-black text-gray-500"
                        )
                        ui.label("시스템 추천에 없음").classes(
                            "text-[10px] text-gray-500"
                        )
            
            ui.label(
                "💡 정확한 진단은 아래 [🤖 AI 진단 실행]을 눌러주세요. "
                "(시세 조회 + DART 공시 + AI 리포트)"
            ).classes("text-xs text-gray-500 italic")
        else:
            ui.label(
                "💡 보유 종목이 오늘 추천에 없습니다. "
                "[🤖 AI 진단 실행]으로 상세 분석을 받아보세요."
            ).classes("text-xs text-gray-500 italic mb-4")
    
    except Exception as _e:
        # Hero 실패해도 화면은 정상
        try:
            logging.getLogger(__name__).warning(
                f"포트폴리오 Hero 카드 렌더 실패: {_e}"
            )
        except Exception:
            pass


def _render_diagnosis_summary(pf_rows: list, cash_amt: float, dart_results: dict):
    """[v22 UI Step L] AI 진단 후 즉시 액션 요약 카드
    
    종목별 카드 직전에 표시 — 사용자가 무엇을 해야 하는지 1초 답변
    
    분류:
      🚨 즉시 액션: 점수 ≤ 40 또는 DART 위험 신호
      ⚠️ 모니터링: 점수 41~59 또는 DART 주의
      ✅ 유지: 점수 ≥ 60
      📤 금일 제외: 전일추천 (시스템에서 빠짐)
    """
    try:
        if not pf_rows:
            return
        
        # 분류
        n_action = 0  # 즉시 액션
        n_monitor = 0
        n_hold = 0
        n_out = 0  # 금일 제외
        
        action_names = []
        for r in pf_rows:
            score = r.get("점수", 0)
            source = r.get("소스", "")
            advice = r.get("AI조언", "")
            code = r.get("code", "")
            
            # DART 위험 추가 가중
            dart = dart_results.get(code, {})
            dart_risk = dart.get("risk_level", "") if dart else ""
            has_dart_warning = "위험" in dart_risk
            
            # [Step M #3] AI조언 용어 변경에 맞춰 키워드도 업데이트
            if "교체 검토" in advice or has_dart_warning:
                n_action += 1
                action_names.append(r["종목명"])
            elif "오늘 추천 제외" in advice or "금일 제외" in advice:
                n_out += 1
                action_names.append(r["종목명"])
            elif "지켜보기" in advice or "관망" in advice or "주의" in dart_risk:
                n_monitor += 1
            elif "계속 보유" in advice or "보유 유지" in advice or "강력홀딩" in advice or "보유" in advice:
                n_hold += 1
            else:
                n_monitor += 1
        
        total_eval = sum(r.get("평가금", 0) for r in pf_rows)
        total_buy = sum(r.get("매입금", 0) for r in pf_rows)
        total_pl = total_eval - total_buy
        total_rate = (total_pl / total_buy * 100) if total_buy > 0 else 0
        
        # 결론 카드
        if n_action > 0 or n_out > 0:
            verdict_emoji = "🚨"
            verdict_text = "즉시 검토 필요"
            verdict_subtitle = f"손절/교체 검토 {n_action}건"
            if n_out > 0:
                verdict_subtitle += f"  ·  금일 추천 제외 {n_out}건"
            border_color = "border-red-500/50"
            text_main = "text-red-300"
            text_sub = "text-red-100"
            grad = "linear-gradient(to right, #3d0a14, #541324, #3d0a14)"
        elif n_monitor > 0:
            verdict_emoji = "⚠️"
            verdict_text = "일부 모니터링 필요"
            verdict_subtitle = f"지켜볼 종목 {n_monitor}건"
            if n_hold > 0:
                verdict_subtitle += f"  ·  양호 {n_hold}건"
            border_color = "border-amber-500/50"
            text_main = "text-amber-300"
            text_sub = "text-amber-100"
            grad = "linear-gradient(to right, #3d2a0a, #544013, #3d2a0a)"
        else:
            verdict_emoji = "✅"
            verdict_text = "포트폴리오 양호"
            verdict_subtitle = f"전 종목 점수 60 이상  ·  유지 권장"
            border_color = "border-emerald-500/50"
            text_main = "text-emerald-300"
            text_sub = "text-emerald-100"
            grad = "linear-gradient(to right, #0a3d2a, #0d5440, #0a3d2a)"
        
        with ui.card().classes(
            f"w-full p-5 mb-3 rounded-xl border-2 {border_color}"
        ).style(f"background: {grad}"):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.column().classes("gap-1"):
                    ui.label(f"{verdict_emoji} {verdict_text}").classes(
                        f"text-lg font-bold {text_main}"
                    )
                    ui.label(verdict_subtitle).classes(f"text-sm {text_sub}")
                with ui.column().classes("items-end gap-0"):
                    pl_color = "text-emerald-300" if total_pl >= 0 else "text-red-300"
                    ui.label(f"{total_pl:+,}원").classes(
                        f"text-xl font-black {pl_color}"
                    )
                    ui.label(f"{total_rate:+.2f}%").classes(
                        f"text-sm {pl_color}"
                    )
        
        # 액션 필요 종목 미리보기 (3개까지)
        if action_names:
            with ui.card().classes(
                "w-full p-3 mb-3 bg-[#0d0d1a] "
                "border border-gray-700 rounded-lg"
            ):
                ui.label("📋 검토 대상 종목").classes(
                    "text-xs text-gray-400 font-bold mb-1"
                )
                preview = action_names[:3]
                more = len(action_names) - 3
                preview_text = "  ·  ".join(preview)
                if more > 0:
                    preview_text += f"  ·  외 {more}종목"
                ui.label(preview_text).classes("text-sm text-white")
                ui.label("⬇️ 아래 카드에서 종목별 상세 진단 확인").classes(
                    "text-[10px] text-gray-500 italic mt-1"
                )
    
    except Exception as _e:
        try:
            logging.getLogger(__name__).warning(
                f"진단 요약 카드 렌더 실패: {_e}"
            )
        except Exception:
            pass


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

    # DART 연동 상태 표시
    # [Step M #4] DART/Gemini 연동 상태 + 설명
    if DART_INTEGRATION_OK and _GENAI_CLIENT:
        dart_status = "🟢 DART+AI 연동"
        dart_explain = "공시 리스크와 AI 리포트 모두 반영"
        dart_color = "text-emerald-400"
    elif DART_INTEGRATION_OK:
        dart_status = "🟡 DART만 연동"
        dart_explain = "AI 리포트는 규칙 기반으로 대체"
        dart_color = "text-amber-400"
    else:
        dart_status = "⚪ DART 미연결"
        dart_explain = "공시 리스크 제외, 가격/점수 기준 진단"
        dart_color = "text-gray-400"
    
    with ui.row().classes("items-center gap-2 mb-2"):
        ui.label(dart_status).classes(f"text-xs {dart_color} font-bold")
        ui.label(f"— {dart_explain}").classes("text-xs text-gray-500")

    # ═══════════════════════════════════════════════════
    # [Step N] 계정별 포트폴리오 자동 저장
    # 우선순위: 사용자 Gist 파일 → 기존 portfolio.txt(legacy) → app.storage.user
    # ═══════════════════════════════════════════════════
    user_profile = None
    try:
        user_profile = get_current_user()
    except Exception:
        pass
    
    # ═══════════════════════════════════════════════════
    # [Step O #1] 로드 우선순위 — 로그인 상태에서는 Gist가 진실의 원천
    # 
    # 로그인 상태:
    #   1. 사용자별 Gist 파일 (portfolio_<hash>.txt) — 진실의 원천
    #   2. 로컬 app.storage.user — Gist 미발견 시 fallback
    #   3. legacy portfolio.txt — 마이그레이션용
    # 
    # 비로그인 상태:
    #   1. 로컬 app.storage.user
    # 
    # 효과:
    #   - 회사 PC에서 수정 후 집 PC 접속 시 최신 Gist 자동 로드
    #   - 브라우저 캐시 삭제해도 정확히 복원
    # ═══════════════════════════════════════════════════
    saved_local = app.storage.user.get("portfolio_text", "")
    saved_gist, gist_source = ("", "")
    saved_meta_at = ""  # Gist 메타에서 가져온 마지막 저장 시각
    
    if user_profile:
        # 로그인 상태: Gist 먼저 읽기
        saved_gist, gist_source = _load_portfolio_file(user_profile)
        saved_meta_at = _load_portfolio_meta(user_profile)
        
        if saved_gist:
            saved = saved_gist
            # 로컬 캐시 동기화 (다음 진입 시 빠른 표시 + 일관성)
            app.storage.user["portfolio_text"] = saved_gist
            if saved_meta_at:
                app.storage.user["portfolio_saved_at"] = saved_meta_at
        elif saved_local:
            saved = saved_local
        else:
            saved = ""
    else:
        # 비로그인: 로컬만
        saved = saved_local or ""
    
    # 저장 상태 안내 (계정별 자동 저장 명시)
    # [Step O #3] 마지막 저장 시각은 Gist 메타 우선 → 로컬 fallback
    last_saved_at = saved_meta_at or app.storage.user.get("portfolio_saved_at", "")
    if user_profile:
        if last_saved_at:
            _save_msg = f"✅ 이 포트폴리오는 내 계정에 자동 저장됩니다.  ·  마지막 저장: {last_saved_at}"
        elif gist_source == "legacy":
            _save_msg = "📦 이전 포트폴리오를 불러왔습니다. 다음 저장 시 내 계정으로 이전됩니다."
        elif gist_source == "user":
            _save_msg = "✅ 이 포트폴리오는 내 계정에 자동 저장됩니다."
        else:
            _save_msg = "💾 보유 종목을 추가하면 내 계정에 자동 저장됩니다."
        ui.label(_save_msg).classes("text-[11px] text-emerald-400/80 mb-2 italic")
    else:
        ui.label(
            "⚠️ 비로그인 상태 — 브라우저에만 임시 저장됩니다. 로그인 시 계정에 자동 저장됩니다."
        ).classes("text-[11px] text-amber-400/80 mb-2 italic")
    
    # [Step N + O] 백그라운드 자동 저장 헬퍼 (종목 추가/삭제 시 사용)
    async def _bg_save_portfolio(text: str, profile):
        try:
            ok, _fn = await run_sync(_save_portfolio_file, text, profile)
            if ok:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                app.storage.user["portfolio_saved_at"] = now_str
                # [Step O #3] 메타 파일도 함께 저장 (기기 간 동기화)
                if profile:
                    await run_sync(_save_portfolio_meta, profile, now_str)
        except Exception as _e:
            _logger.warning(f"백그라운드 저장 실패: {_e}")

    # ═══════════════════════════════════════════════════
    # [v22 UI Step L] Hero 카드 — 첫 화면 1초 답변
    # 시장/종목 탭과 동일 패턴: 결론 → 카드 → 상세
    # ═══════════════════════════════════════════════════
    _render_portfolio_hero(saved, df)

    # [v21.3] 종목 검색 기반 입력 UI
    code_map = _get_code_map(df)
    _ensure_krx_map()
    all_names = set(code_map.keys()) | set(_KRX_NAME_MAP.keys())
    stock_names = sorted(all_names) if all_names else []

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
            existing_names = [l.split(":")[0] for l in current.split("\n") if ":" in l]
            if name in existing_names:
                lines = current.split("\n")
                updated = [new_line if l.startswith(f"{name}:") else l for l in lines]
                pf_input.value = "\n".join(updated)
                ui.notify(f"✏️ {name} 업데이트 완료", type="positive")
            else:
                pf_input.value = f"{current}\n{new_line}" if current else new_line
                ui.notify(f"✅ {name} 추가 완료", type="positive")

            app.storage.user["portfolio_text"] = pf_input.value
            # [Step N] 백그라운드 자동 저장 (Gist) — UI 블로킹 없이
            asyncio.ensure_future(_bg_save_portfolio(pf_input.value, user_profile))
            stock_select.value = None
            avg_price_input.value = None
            qty_input.value = None
            _refresh_holdings()

        ui.button("➕ 추가", on_click=_add_stock).props("color=primary dense").classes("h-10")

    # 현재 보유 목록 미니 카드
    holding_area = ui.column().classes("w-full mt-2 mb-2")

    def _refresh_holdings():
        holding_area.clear()
        text = pf_input.value.strip()
        if not text:
            return
        items = []
        for line in text.split("\n"):
            if ":" not in line: continue
            parts = line.split(":")
            if len(parts) < 3: continue
            try:
                items.append({"name": parts[0].strip(), "avg": int(parts[1]), "qty": int(parts[2])})
            except (ValueError, IndexError):
                pass
        if not items:
            return
        with holding_area:
            # [Step P] 헤더 — 종목 수 + 전체 삭제 버튼
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label(f"📌 보유 종목 {len(items)}개").classes(
                    "text-sm font-bold text-cyan-300"
                )
                
                def _clear_all():
                    """전체 삭제 — 확인 다이얼로그"""
                    with ui.dialog() as dlg, ui.card().classes("p-4"):
                        ui.label("⚠️ 보유 종목 전체 삭제").classes(
                            "text-lg font-bold text-red-400"
                        )
                        ui.label(f"{len(items)}개 종목을 모두 삭제합니다. 계속할까요?").classes(
                            "text-sm text-gray-300 my-2"
                        )
                        ui.label("⚠️ 이 작업은 되돌릴 수 없습니다.").classes(
                            "text-xs text-amber-400 mb-3"
                        )
                        with ui.row().classes("gap-2 justify-end"):
                            ui.button("취소", on_click=dlg.close).props("flat")
                            
                            def _confirm_clear():
                                pf_input.value = ""
                                app.storage.user["portfolio_text"] = ""
                                asyncio.ensure_future(_bg_save_portfolio("", user_profile))
                                ui.notify("🗑️ 전체 종목 삭제 완료", type="warning")
                                dlg.close()
                                _refresh_holdings()
                            
                            ui.button("전체 삭제", on_click=_confirm_clear).props(
                                "color=red"
                            )
                    dlg.open()
                
                if items:
                    ui.button("🗑️ 전체 삭제", on_click=_clear_all).props(
                        "flat dense size=sm color=red"
                    ).classes("text-xs")
            
            # 종목 카드 그리드
            with ui.row().classes("w-full gap-2 flex-wrap"):
                for item in items:
                    val = item['avg'] * item['qty']
                    with ui.card().classes(
                        "p-3 bg-[#0d0d1a] border border-gray-700 rounded-lg "
                        "min-w-[260px] hover:border-cyan-500/50"
                    ):
                        # 종목명 + 합계
                        with ui.row().classes("w-full items-center justify-between mb-1"):
                            ui.label(f"📌 {item['name']}").classes(
                                "text-white text-sm font-bold"
                            )
                            ui.label(f"{val:,}원").classes(
                                "text-sm text-cyan-400 font-bold"
                            )
                        
                        # 평단가 × 수량
                        ui.label(f"{item['avg']:,}원 × {item['qty']}주").classes(
                            "text-xs text-gray-400 mb-2"
                        )
                        
                        # [Step P] 편집 / 삭제 버튼 (충분한 터치 영역)
                        with ui.row().classes("w-full gap-2"):
                            def _make_edit(it=item):
                                """평단가/수량 수정 다이얼로그"""
                                def _open_edit():
                                    with ui.dialog() as dlg, ui.card().classes("p-4 min-w-[300px]"):
                                        ui.label(f"✏️ {it['name']} 수정").classes(
                                            "text-base font-bold text-white mb-3"
                                        )
                                        new_avg = ui.number(
                                            "평단가 (원)", value=it['avg'],
                                            min=0, step=100, format="%.0f"
                                        ).classes("w-full mb-2")
                                        new_qty = ui.number(
                                            "수량 (주)", value=it['qty'],
                                            min=1, step=1, format="%.0f"
                                        ).classes("w-full mb-3")
                                        
                                        with ui.row().classes("gap-2 justify-end"):
                                            ui.button("취소", on_click=dlg.close).props("flat")
                                            
                                            def _save_edit():
                                                avg_v = int(new_avg.value or 0)
                                                qty_v = int(new_qty.value or 0)
                                                if avg_v <= 0 or qty_v <= 0:
                                                    ui.notify(
                                                        "⚠️ 평단가와 수량을 입력하세요",
                                                        type="warning"
                                                    )
                                                    return
                                                # 해당 라인 교체
                                                lines = pf_input.value.strip().split("\n")
                                                new_line = f"{it['name']}:{avg_v}:{qty_v}"
                                                updated = [
                                                    new_line if l.startswith(f"{it['name']}:") else l
                                                    for l in lines
                                                ]
                                                pf_input.value = "\n".join(updated)
                                                app.storage.user["portfolio_text"] = pf_input.value
                                                asyncio.ensure_future(
                                                    _bg_save_portfolio(pf_input.value, user_profile)
                                                )
                                                ui.notify(
                                                    f"✏️ {it['name']} 수정 완료 "
                                                    f"({avg_v:,}원 × {qty_v}주)",
                                                    type="positive"
                                                )
                                                dlg.close()
                                                _refresh_holdings()
                                            
                                            ui.button("저장", on_click=_save_edit).props(
                                                "color=primary"
                                            )
                                    dlg.open()
                                return _open_edit
                            
                            def _make_remove(it=item):
                                """삭제 — 확인 다이얼로그"""
                                def _open_remove():
                                    with ui.dialog() as dlg, ui.card().classes("p-4"):
                                        ui.label(f"🗑️ {it['name']} 삭제").classes(
                                            "text-base font-bold text-red-400 mb-2"
                                        )
                                        ui.label(
                                            f"{it['avg']:,}원 × {it['qty']}주 = "
                                            f"{it['avg']*it['qty']:,}원"
                                        ).classes("text-sm text-gray-300 mb-3")
                                        ui.label("이 종목을 삭제할까요?").classes(
                                            "text-sm text-gray-400 mb-3"
                                        )
                                        
                                        with ui.row().classes("gap-2 justify-end"):
                                            ui.button("취소", on_click=dlg.close).props("flat")
                                            
                                            def _confirm_remove():
                                                lines = [
                                                    l for l in pf_input.value.strip().split("\n")
                                                    if not l.startswith(f"{it['name']}:")
                                                ]
                                                pf_input.value = "\n".join(lines)
                                                app.storage.user["portfolio_text"] = pf_input.value
                                                asyncio.ensure_future(
                                                    _bg_save_portfolio(pf_input.value, user_profile)
                                                )
                                                ui.notify(
                                                    f"🗑️ {it['name']} 삭제됨",
                                                    type="info"
                                                )
                                                dlg.close()
                                                _refresh_holdings()
                                            
                                            ui.button("삭제", on_click=_confirm_remove).props(
                                                "color=red"
                                            )
                                    dlg.open()
                                return _open_remove
                            
                            # 편집 버튼 (회색)
                            ui.button(
                                "✏️ 수정", on_click=_make_edit(item)
                            ).props("flat dense size=sm color=blue").classes(
                                "text-xs flex-1"
                            )
                            # 삭제 버튼 (빨강)
                            ui.button(
                                "🗑️ 삭제", on_click=_make_remove(item)
                            ).props("flat dense size=sm color=red").classes(
                                "text-xs flex-1"
                            )

    # 기존 textarea (접힘)
    with ui.expansion("📋 텍스트 직접 편집 (고급)", value=False).classes("w-full text-xs text-gray-500"):
        pf_input = ui.textarea("포트폴리오 데이터", value=saved,
                               placeholder="종목명:평단가:수량\n예) 에코프로머티:67341:60").classes("w-full").props("rows=4")

    result_area = ui.column().classes("w-full mt-4")

    # ── DART 공시 분석 토글 ──
    dart_toggle = ui.checkbox("📜 DART 공시 분석 포함", value=True).classes("text-gray-300")

    def _auto_save():
        app.storage.user["portfolio_text"] = pf_input.value
        # [Step N] textarea blur 시에도 백그라운드 자동 저장
        asyncio.ensure_future(_bg_save_portfolio(pf_input.value, user_profile))
        _refresh_holdings()
    pf_input.on("blur", lambda _: _auto_save())
    _refresh_holdings()

    async def analyze():
        result_area.clear()
        text = pf_input.value.strip()
        if not text: return

        app.storage.user["portfolio_text"] = text
        # [Step N + O] 계정별 자동 저장 + 메타 동시 저장
        save_ok, _saved_filename = await run_sync(_save_portfolio_file, text, user_profile)
        if save_ok:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            app.storage.user["portfolio_saved_at"] = now_str
            # [Step O #3] 메타 파일 함께 저장 (기기 간 동기화)
            if user_profile:
                await run_sync(_save_portfolio_meta, user_profile, now_str)
                ui.notify(f"💾 내 계정에 저장됨 ({now_str})", type="positive")
            else:
                ui.notify("💾 브라우저에 임시 저장됨 (로그인 시 계정 저장)", type="info")
        else:
            ui.notify(
                "⚠️ 계정 저장 실패 — 현재 브라우저에만 임시 저장되었습니다.",
                type="warning"
            )

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
                if score >= 80: advice, acolor = "✅ 계속 보유", "#10B981"
                elif score >= 60: advice, acolor = "✅ 보유 유지", "#3B82F6"
                elif score <= 40 and score > 0: advice, acolor = "🚨 교체 검토", "#EF4444"
                else: advice, acolor = "👀 지켜보기", "#F59E0B"
            elif source.startswith("전일추천"):
                if score >= 70: advice, acolor = f"📤 오늘 추천 제외 (전일 {score:.0f}점) — 홀딩 검토", "#F59E0B"
                elif score >= 50: advice, acolor = f"📤 오늘 추천 제외 (전일 {score:.0f}점) — 모니터링", "#F59E0B"
                else: advice, acolor = f"📤 오늘 추천 제외 (전일 {score:.0f}점) — 손절 검토", "#EF4444"
            else:
                if curr == 0: advice, acolor = "❓ 시세조회 실패", "#EF4444"
                else: advice, acolor = "ℹ️ 추천 데이터 없음", "#9CA3AF"

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

            # ═══════════════════════════════════════════════════
            # [v22 UI Step L] 진단 요약 카드 — 즉시 액션 결론
            # 종목별 상세 카드 직전에 표시 → 사용자 1초 답변
            # ═══════════════════════════════════════════════════
            _render_diagnosis_summary(pf_rows, cash_amt, dart_results)

            # ── 종목별 카드 (DART 통합) ──
            _section_title("🩺 종목별 AI 진단 상세")
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
                                    # [Step L] ROUTE 영문 → 한국어 (route_display 사용)
                                    _route_show = route_display(r["상태"])
                                    _rc = {
                                        "ATTACK": "red", "ARMED": "orange",
                                        "WAIT": "blue", "NEUTRAL": "gray",
                                        "CARRY": "purple", "OVERHEAT": "red",
                                    }.get(str(r["상태"]).upper(), "gray")
                                    ui.badge(_route_show, color=_rc).classes("text-xs")
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
