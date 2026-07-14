# -*- coding: utf-8 -*-
"""
components/action_cards.py — v24.5 '오늘의 추천' 액션 카드 렌더러
═══════════════════════════════════════════════════════════════════
모바일 우선 재설계. 핵심 화면을 순정 위젯 스택 대신 액션 카드로:
  ① 판단 먼저 — 매수구간 / 눌림대기 / 추격주의 (verdict)
  ② 가격 사다리 — 손절·진입·현재·1차 익절을 한 줄에 시각화 (signature)
  ③ PMS 추천 레인 분리 (v24.4 SHADOW, 검증중 태그)

설계 원칙
  · 기존 v18 'Premium Dark Finance' 토큰 계승, CSS는 sp- 프리픽스로 스코프(충돌 0).
  · 데이터는 recommend df 실제 컬럼만 사용(종가/추천매수가/손절가/추천매도가1/RSI14/
    DISPLAY_SCORE/ROUTE/거래대금(억원)/TP1_PROB) + 기존 get_buy_now_display 재사용.
  · 결측 강건: 컬럼 없거나 0이면 해당 요소 생략, 절대 예외 던지지 않음.
  · drop-in: dashboard.py tab1/tab2에서 render_action_cards(df) 한 줄로 교체 가능.

사용
  from components.action_cards import render_action_cards
  render_action_cards(view_df, is_admin=is_admin)        # 카드 렌더
  # (선택) 상단 필터는 네이티브 위젯으로:
  #   mode = st.segmented_control(...) → render_action_cards(df, filter_mode=mode)
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from services.recommendation_quality import production_buy_mask

try:
    import streamlit as st
except Exception:  # 테스트/오프라인 렌더용
    st = None

# 기존 표시 모델 재사용 (없어도 동작하도록 폴백)
try:
    from components.buy_now_badge import get_buy_now_display
except Exception:
    try:
        from buy_now_badge import get_buy_now_display
    except Exception:
        get_buy_now_display = None


# ── 안전 변환 ────────────────────────────────────────────────────
def _f(v, d=0.0) -> float:
    try:
        if v is None:
            return d
        s = str(v).replace(",", "").replace("%", "").strip()
        if s in ("", "nan", "None", "-"):
            return d
        return float(s)
    except Exception:
        return d


def _i(v, d=0) -> int:
    return int(round(_f(v, d)))


def _col(row, *names, default=0.0):
    """여러 후보 컬럼명 중 첫 유효값."""
    for n in names:
        if n in row and pd.notna(row[n]):
            val = _f(row[n], None)
            if val is not None:
                return val
    return default


# 눌림 대기 ↔ 추격 주의 경계 (진입가 대비 상단 이격 %). 제품 튜닝 포인트.
PULLBACK_WAIT_MAX_PCT = 12.0
OVERHEAT_RSI = 75.0


def _upside_pct(entry: float, target: float) -> float:
    if entry <= 0 or target <= 0:
        return 0.0
    return (target / entry - 1.0) * 100.0


# ── verdict: 지금 뭘 해야 하나 ───────────────────────────────────
def _verdict(row) -> dict:
    """현재가 vs 진입가를 1차 기준으로 행동 판단.

    공식 BUY_NOW_GRADE 'AVOID'는 '지금 시장가 매수 금지'(=진입가 위)를 뜻할 뿐
    영구 회피가 아니므로, 본 카드는 가격 갭으로 '눌림 대기'와 '추격 주의'를 구분한다.
      · 진입가 도달/이탈(아래)        → 매수 구간
      · 진입가 위 소폭(≤경계%)·비과열 → 눌림 대기 (지정가 걸고 대기)
      · 진입가 위 과대 or 강과열       → 추격 주의
    """
    cur = _col(row, "종가", "현재가")
    buy = _col(row, "추천매수가")
    route = str(row.get("ROUTE", "")).upper()
    rsi = _col(row, "RSI14")
    overheat_hard = ("OVERHEAT" in route) and (rsi >= OVERHEAT_RSI)

    if buy <= 0:
        return {"key": "wait", "label": "관찰", "mark": "◐",
                "sub": "진입가 미설정 — 관찰 단계."}

    gap = (cur - buy) / buy * 100.0

    if cur <= buy * 1.01:
        return {"key": "buy", "label": "매수 구간", "mark": "●",
                "sub": "현재가가 진입가에 도달 — <b>분할 진입 가능</b>. 손절 기준 관리."}
    if overheat_hard or gap > PULLBACK_WAIT_MAX_PCT:
        tail = " · 과열 구간" if overheat_hard else ""
        return {"key": "avoid", "label": "추격 주의", "mark": "✕",
                "sub": f"진입가 대비 <b>+{gap:.1f}%</b> 이격{tail} — 추격 자제, 눌림 확인."}
    return {"key": "wait", "label": "눌림 대기", "mark": "◐",
            "sub": f"진입가까지 <b>약 {gap:.1f}% 눌림</b> 대기 — {_i(buy):,} 지정가 권장."}


# ── 가격 사다리 좌표 (손절·진입·현재·목표 → 2~98%) ──────────────
def _ladder_html(row) -> str:
    stop = _col(row, "손절가")
    buy = _col(row, "추천매수가")
    cur = _col(row, "종가", "현재가")
    tp = _col(row, "추천매도가1")
    tp_pct = _upside_pct(buy, tp)
    pts = [("stop", "손절", stop), ("buy", "진입", buy),
           ("now", "현재", cur), ("tp", "1차 익절", tp)]
    vals = [v for _, _, v in pts if v > 0]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0

    def pos(v):
        return max(3.0, min(97.0, (v - lo) / span * 100.0))

    marks = []
    for key, lab, v in pts:
        if v <= 0:
            continue
        cls = "now" if key == "now" else key
        label_text = f"{lab} {_i(v):,}"
        if key == "tp" and tp_pct > 0:
            label_text = f"{lab} +{tp_pct:.1f}%"
        marks.append(
            f"<div class='sp-mk sp-{cls}' style='left:{pos(v):.1f}%'>"
            f"<div class='sp-pin'></div>"
            f"<div class='sp-lab'>{label_text}</div></div>"
        )
    return (f"<div class='sp-ladder'><div class='sp-track'>"
            f"<div class='sp-bar'></div>{''.join(marks)}</div></div>")


def _targets_html(row) -> str:
    """Separate the model's first take-profit from its extension target.

    추천매도가1 is intentionally a first exit level, not a ceiling for the
    stock.  Showing 추천매도가2 alongside its upside prevents the first level
    from being mistaken for the model's final valuation.
    """
    entry = _col(row, "추천매수가")
    tp1 = _col(row, "추천매도가1")
    tp2 = _col(row, "추천매도가2")
    cells = []
    for label, value, css in [
        ("1차 익절", tp1, "sp-target-1"),
        ("연장 목표", tp2, "sp-target-2"),
    ]:
        if value <= 0:
            continue
        if label == "연장 목표" and tp1 > 0 and value <= tp1:
            continue
        upside = _upside_pct(entry, value)
        pct = f"+{upside:.1f}%" if upside >= 0 else f"{upside:.1f}%"
        cells.append(
            f"<div class='sp-target {css}'><span>{label}</span>"
            f"<b>{_i(value):,}</b><em>{pct}</em></div>"
        )
    return f"<div class='sp-targets'>{''.join(cells)}</div>" if cells else ""


# ── 점수 게이지 (DISPLAY_SCORE 원형) ────────────────────────────
def _gauge_html(score: float, color: str) -> str:
    s = max(0.0, min(100.0, score))
    circ = 2 * math.pi * 17  # ≈106.8
    off = circ * (1 - s / 100.0)
    return (
        f"<div class='sp-gauge'><svg width='42' height='42'>"
        f"<circle cx='21' cy='21' r='17' fill='none' stroke='rgba(255,255,255,.08)' stroke-width='4'/>"
        f"<circle cx='21' cy='21' r='17' fill='none' stroke='{color}' stroke-width='4' "
        f"stroke-linecap='round' stroke-dasharray='{circ:.1f}' stroke-dashoffset='{off:.1f}' "
        f"transform='rotate(-90 21 21)'/></svg>"
        f"<div class='sp-gv' style='color:{color}'>{s:.0f}</div></div>"
    )


_VERDICT_COLOR = {"buy": "#10B981", "wait": "#F59E0B",
                  "avoid": "#EF4444", "pms": "#8B5CF6"}  # SVG stroke 속성엔 hex 필요(var 미해석)
_ROUTE_CLASS = {"OVERHEAT": "sp-r-over", "ATTACK": "sp-r-atk", "ARMED": "sp-r-arm"}


# ── 카드 1장 HTML ────────────────────────────────────────────────
def _card_html(row, lane: str = "official") -> str:
    name = str(row.get("종목명", "—"))
    code = str(row.get("종목코드", "")).zfill(6) if str(row.get("종목코드", "")).strip() else ""
    disp = _col(row, "DISPLAY_SCORE")
    rsi = _col(row, "RSI14")
    turn = _col(row, "거래대금(억원)")
    tp_prob = _col(row, "TP1_PROB")
    vint = _col(row, "거래강도")
    route = str(row.get("ROUTE", "")).upper().strip()
    pms = _col(row, "PMS_SCORE")

    if lane == "pms":
        vk = "pms"
        verdict_label, verdict_mark = "PMS Top", "⚡"
        color = _VERDICT_COLOR["pms"]
        elite = _col(row, "ELITE_SCORE")
        sub = (f"모멘텀확인 6피처 <b>상위 {pms:.0f}점</b> — "
               f"<b style='color:#EF4444'>⚠️ 백테스트에서 체결편향 발견(로그 +17.7% → 실제 +0.7%), "
               f"재검증 전까지 매수 참고 금지.</b> 추적 기록용으로만 표시.")
        gauge_val, gauge_color = (pms if pms > 0 else disp), color
    elif lane == "watch":
        vk = "wait"
        verdict_label, verdict_mark = "관찰 후보", "◐"
        color = _VERDICT_COLOR["wait"]
        gauge_val, gauge_color = disp, color
        reason = str(row.get("QUALITY_GUARD_REASON", "") or "최종 품질 기준 미달")
        sub = ("<b>매수 추천 아님.</b> 공식 품질게이트를 통과하지 못했습니다. "
               f"<span style='color:var(--sp-t3)'>{reason}</span>")
    else:
        v = _verdict(row)
        vk = v["key"]
        verdict_label, verdict_mark, sub = v["label"], v["mark"], v["sub"]
        color = _VERDICT_COLOR.get(vk, "#F59E0B")
        gauge_val, gauge_color = disp, color
        # [v24.7] 매수구간 카드에 D+1 체크포인트 컷 표시 (컬럼 있을 때만, 하위호환)
        if vk == "buy":
            d1p = _col(row, "D1_CHECKPOINT_PRICE")
            if d1p > 0:
                sub += (f" <span style='color:#EF4444'>🛡 D+1 컷 {_i(d1p):,}원</span>"
                        f" — 익일 종가 이탈 시 조기청산.")
        # [v25.1] 청산 규율 — 고위험 루트 경고 + 권고 손절/익절선 (컬럼 있을 때만)
        er = str(row.get("EXIT_ROUTE_RISK", "") or "")
        if er == "HIGH_AVOID":
            sub += (" <span style='color:#EF4444;font-weight:700'>⛔ 고위험 루트</span>"
                    " — 정직검증상 진입 보류 권고.")
        est = _col(row, "EXIT_STOP_TIGHT"); etp = _col(row, "EXIT_TP_QUICK")
        if vk == "buy" and est > 0 and etp > 0:
            sub += (f" <span style='color:var(--sp-t3)'>규율: 손절 {_i(est):,}(-7%)"
                    f" · 50% 부분익절 {_i(etp):,}(+10%)</span>")

    edge = f"sp-edge-{vk}"
    verdict_cls = f"sp-v-{vk}"
    card_cls = "sp-card sp-pms-card" if lane == "pms" else "sp-card"

    # 하단 스탯 (값 있는 것만)
    stat_cells = []
    if lane == "pms" and vint > 0:
        stat_cells.append(f"<div class='sp-stat'><div class='sp-sv'>{vint:.2f}</div><div class='sp-sk'>거래강도</div></div>")
    if rsi > 0:
        stat_cells.append(f"<div class='sp-stat'><div class='sp-sv'>{rsi:.1f}</div><div class='sp-sk'>RSI</div></div>")
    if turn > 0:
        ttxt = f"{turn/10000:.1f}조" if turn >= 10000 else f"{turn:,.0f}억"
        stat_cells.append(f"<div class='sp-stat'><div class='sp-sv'>{ttxt}</div><div class='sp-sk'>거래대금</div></div>")
    if tp_prob > 0 and lane != "pms":
        stat_cells.append(f"<div class='sp-stat'><div class='sp-sv'>{tp_prob:.0f}/100</div><div class='sp-sk'>목표 근거점수</div></div>")
    if route:
        rcls = _ROUTE_CLASS.get(route, "sp-r-wait")
        stat_cells.append(f"<div class='sp-stat'><div class='sp-route {rcls}'>{route}</div><div class='sp-sk' style='margin-top:5px'>루트</div></div>")

    # [v24.9] 정직 기저확률 — 공식 레인 카드 하단 한 줄 (컬럼 있을 때만)
    srp = _col(row, "SRP_BASE_PROB_2D")
    srp_html = ""
    if lane == "official" and srp > 0:
        srp_html = (f"<div style='font-size:10px;color:var(--sp-t3);margin-top:8px'>"
                    f"🎲 시장기저 2일 상승률 <b style='color:var(--sp-t2)'>{srp:.0f}%</b>"
                    f" (실측 · 종목별 유의차 미검증)</div>")

    gauge = _gauge_html(gauge_val, gauge_color)
    stats = f"<div class='sp-stats'>{gauge}{''.join(stat_cells)}</div>" if stat_cells else ""
    code_html = f"<span class='sp-code'>{code}</span>" if code else ""

    return f"""<div class="{card_cls}">
  <div class="sp-ledge {edge}"></div>
  <div class="sp-chead">
    <div class="sp-name">{name} {code_html}</div>
    <div class="sp-verdict {verdict_cls}">{verdict_mark} {verdict_label}</div>
  </div>
  <div class="sp-sub">{sub}</div>
  {_ladder_html(row)}
  {_targets_html(row)}
  {stats}
  {srp_html}
</div>"""


# ── 스코프 CSS (1회 주입) ────────────────────────────────────────
_CSS = """<style>
.sp-feed{--sp-buy:#10B981;--sp-wait:#F59E0B;--sp-avoid:#EF4444;--sp-pms:#8B5CF6;
  --sp-info:#3B82F6;--sp-t1:#F1F5F9;--sp-t2:#9AA7BD;--sp-t3:#5C6B83;
  --sp-card:#141925;--sp-line:rgba(255,255,255,.07);
  --sp-mono:'JetBrains Mono',monospace;display:flex;flex-direction:column;gap:12px;margin-top:6px}
.sp-lane{display:flex;align-items:center;gap:8px;font-size:11px;font-weight:700;letter-spacing:.04em;
  color:var(--sp-t3);text-transform:uppercase;margin:10px 2px 0}
.sp-lane.pms{color:var(--sp-pms)}
.sp-lane .sp-tag{font-size:9px;font-weight:700;padding:2px 6px;border-radius:5px;background:rgba(139,92,246,.15);
  color:var(--sp-pms);border:1px solid rgba(139,92,246,.3);text-transform:none}
.sp-lane .sp-ln{flex:1;height:1px;background:var(--sp-line)}
.sp-card{position:relative;background:var(--sp-card);border:1px solid var(--sp-line);border-radius:16px;
  padding:15px 15px 13px;overflow:hidden}
.sp-pms-card{border-color:rgba(139,92,246,.28);background:linear-gradient(180deg,rgba(139,92,246,.05),var(--sp-card))}
.sp-ledge{position:absolute;left:0;top:0;bottom:0;width:3px}
.sp-edge-buy{background:var(--sp-buy)}.sp-edge-wait{background:var(--sp-wait)}
.sp-edge-avoid{background:var(--sp-avoid)}.sp-edge-pms{background:var(--sp-pms)}
.sp-chead{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}
.sp-name{font-size:17px;font-weight:700;letter-spacing:-.01em;color:var(--sp-t1);display:flex;align-items:center;gap:7px}
.sp-code{font-size:11px;color:var(--sp-t3);font-family:var(--sp-mono);font-weight:500}
.sp-verdict{font-size:12px;font-weight:700;padding:5px 10px;border-radius:9px;white-space:nowrap}
.sp-v-buy{background:rgba(16,185,129,.14);color:var(--sp-buy)}
.sp-v-wait{background:rgba(245,158,11,.14);color:var(--sp-wait)}
.sp-v-avoid{background:rgba(239,68,68,.13);color:var(--sp-avoid)}
.sp-v-pms{background:rgba(139,92,246,.15);color:var(--sp-pms)}
.sp-sub{font-size:12.5px;color:var(--sp-t2);margin-top:5px;line-height:1.45}
.sp-sub b{color:var(--sp-t1);font-weight:600}
.sp-ladder{margin:14px 2px 4px}
.sp-track{position:relative;height:34px}
.sp-bar{position:absolute;top:15px;left:0;right:0;height:4px;border-radius:3px;
  background:linear-gradient(90deg,var(--sp-avoid),var(--sp-wait) 42%,var(--sp-buy));opacity:.32}
.sp-mk{position:absolute;top:6px;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center}
.sp-mk .sp-pin{width:2px;height:22px;border-radius:2px;background:var(--sp-t3)}
.sp-mk .sp-lab{font-size:9px;color:var(--sp-t3);font-weight:600;white-space:nowrap;font-family:var(--sp-mono);
  position:absolute;top:24px}
.sp-now{z-index:3}
.sp-now .sp-pin{width:3px;height:26px;background:var(--sp-t1);box-shadow:0 0 0 3px var(--sp-card),0 0 12px rgba(241,245,249,.4)}
.sp-now .sp-lab{color:var(--sp-t1);font-weight:700;top:-15px}
.sp-stop .sp-pin{background:var(--sp-avoid)}.sp-stop .sp-lab{color:var(--sp-avoid)}
.sp-buy .sp-pin{background:var(--sp-buy)}.sp-buy .sp-lab{color:var(--sp-buy)}
.sp-tp .sp-pin{background:var(--sp-info)}.sp-tp .sp-lab{color:var(--sp-info)}
.sp-targets{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-top:17px}
.sp-target{display:grid;grid-template-columns:1fr auto;align-items:center;gap:2px 8px;padding:8px 10px;
  border:1px solid var(--sp-line);border-radius:10px;background:rgba(15,23,42,.45)}
.sp-target span{font-size:9.5px;color:var(--sp-t3);font-weight:700}.sp-target b{font-size:12px;color:var(--sp-t1);
  font-family:var(--sp-mono)}.sp-target em{grid-column:2;font-size:10px;font-style:normal;font-weight:700;color:var(--sp-info)}
.sp-target-2 em{color:#A78BFA}
.sp-cash{padding:14px;border-radius:12px;background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2)}
.sp-cash b{display:block;color:#FCD34D;font-size:14px}.sp-cash span{display:block;color:var(--sp-t2);font-size:11px;margin-top:3px}
.sp-stats{display:flex;gap:6px;margin-top:16px;padding-top:11px;border-top:1px solid var(--sp-line);align-items:center}
.sp-stat{flex:1;text-align:center}
.sp-sv{font-size:13px;font-weight:700;font-family:var(--sp-mono);color:var(--sp-t1)}
.sp-sk{font-size:9.5px;color:var(--sp-t3);font-weight:600;letter-spacing:.03em;margin-top:2px;text-transform:uppercase}
.sp-gauge{position:relative;width:42px;height:42px;flex:0 0 auto}
.sp-gv{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;font-family:var(--sp-mono)}
.sp-route{font-size:10px;font-weight:700;padding:2px 7px;border-radius:6px;letter-spacing:.03em}
.sp-r-over{background:rgba(239,68,68,.13);color:var(--sp-avoid)}
.sp-r-atk{background:rgba(239,68,68,.13);color:var(--sp-avoid)}
.sp-r-arm{background:rgba(245,158,11,.14);color:var(--sp-wait)}
.sp-r-wait{background:rgba(148,163,184,.12);color:var(--sp-t2)}
</style>"""


def _inject_css():
    if st is None:
        return
    if not st.session_state.get("_sp_action_css"):
        st.markdown(_CSS, unsafe_allow_html=True)
        st.session_state["_sp_action_css"] = True


# ── 메인 진입점 ──────────────────────────────────────────────────
def render_action_cards(df: pd.DataFrame,
                        *,
                        is_admin: bool = False,
                        max_cards: Optional[int] = 30,
                        show_pms: bool = False,
                        filter_mode: str = "공식 매수") -> None:
    """추천 df를 액션 카드로 렌더. dashboard.py tab1/tab2 drop-in.

    filter_mode: '공식 매수' | '관찰 후보' | '전체' | 'PMS 레인'
    기본 화면은 공식 매수만 노출한다. 구버전 가격 필터도 계속 허용한다.
    """
    if st is None:
        raise RuntimeError("streamlit 미설치 — build_cards_html()로 오프라인 렌더 사용")
    if df is None or len(df) == 0:
        st.info("표시할 추천 종목이 없습니다.")
        return
    _inject_css()
    st.markdown(build_cards_html(df, max_cards=max_cards, show_pms=show_pms,
                                 filter_mode=filter_mode), unsafe_allow_html=True)


def build_cards_html(df: pd.DataFrame,
                     *,
                     max_cards: Optional[int] = 30,
                     show_pms: bool = False,
                     filter_mode: str = "공식 매수",
                     include_css: bool = False) -> str:
    """카드 피드 HTML 문자열 (오프라인 렌더/테스트 가능).

    include_css=True면 스코프 CSS(<style>)를 앞에 포함 — NiceGUI ui.html()용.
    (Streamlit은 _inject_css로 세션당 1회 주입하므로 False 유지)
    """
    work = df.copy()
    # 정렬: 최종 품질게이트 점수를 우선하고, 없으면 기존 표시점수를 사용한다.
    sort_col = "QUALITY_GUARD_SCORE" if "QUALITY_GUARD_SCORE" in work.columns else "DISPLAY_SCORE"
    if sort_col in work.columns:
        work = work.sort_values(sort_col, ascending=False)

    # 공식 vs PMS 레인 분리
    has_pms = "PMS_PROFIT_PICK" in work.columns
    if has_pms:
        pms_mask = pd.to_numeric(
            work["PMS_PROFIT_PICK"], errors="coerce"
        ).fillna(0).astype(int).eq(1)
        off_df = work[~pms_mask]
        pms_df = work[pms_mask] if show_pms else work.iloc[0:0]
    else:
        pms_df, off_df = work.iloc[0:0], work

    production = production_buy_mask(off_df)
    has_production = bool(production.any())
    official_df = off_df[production]
    action = off_df.get("ACTION_DECISION", pd.Series("CASH", index=off_df.index))
    action = action.fillna("CASH").astype(str).str.upper()
    watch_df = off_df[(~production) & action.eq("WATCH")]

    # 필터
    def _vk(row):
        return _verdict(row)["key"]

    if filter_mode == "공식 매수":
        watch_df = watch_df.iloc[0:0]
        pms_df = pms_df.iloc[0:0]
    elif filter_mode == "관찰 후보":
        official_df = official_df.iloc[0:0]
        pms_df = pms_df.iloc[0:0]
    elif filter_mode == "매수 구간":  # 구버전 API 호환
        official_df = official_df[official_df.apply(lambda r: _vk(r) == "buy", axis=1)]
        watch_df = watch_df[watch_df.apply(lambda r: _vk(r) == "buy", axis=1)]
        pms_df = pms_df.iloc[0:0]
    elif filter_mode == "눌림 대기":  # 구버전 API 호환
        official_df = official_df[official_df.apply(lambda r: _vk(r) == "wait", axis=1)]
        watch_df = watch_df[watch_df.apply(lambda r: _vk(r) == "wait", axis=1)]
        pms_df = pms_df.iloc[0:0]
    elif filter_mode == "PMS 레인":
        official_df = official_df.iloc[0:0]
        watch_df = watch_df.iloc[0:0]

    if max_cards:
        official_df = official_df.head(max_cards)
        watch_df = watch_df.head(min(3, max_cards))
        pms_df = pms_df.head(max(3, max_cards // 5))

    parts = ['<div class="sp-feed">']
    if not has_production:
        parts.append('<div class="sp-cash"><b>오늘 공식 매수 0개 · 현금 보유</b>'
                     '<span>품질게이트를 통과한 종목이 없습니다. 관찰 후보는 매수 추천이 아닙니다.</span></div>')
    if len(official_df) > 0:
        parts.append('<div class="sp-lane">✓ 공식 매수 <span class="sp-ln"></span> 품질게이트 통과</div>')
        for _, row in official_df.iterrows():
            parts.append(_card_html(row, lane="official"))
    if len(watch_df) > 0:
        parts.append('<div class="sp-lane">◐ 관찰 후보 <span class="sp-tag">매수 추천 아님</span><span class="sp-ln"></span></div>')
        for _, row in watch_df.iterrows():
            parts.append(_card_html(row, lane="watch"))
    if len(pms_df) > 0:
        parts.append('<div class="sp-lane pms">⚡ PMS 추천 레인 '
                     '<span class="sp-tag">⚠️ 편향 재검증 중 · 매수참고 금지</span><span class="sp-ln"></span></div>')
        for _, row in pms_df.iterrows():
            parts.append(_card_html(row, lane="pms"))
    if len(official_df) == 0 and len(watch_df) == 0 and len(pms_df) == 0 and has_production:
        parts.append('<div style="color:#5C6B83;font-size:13px;padding:20px;text-align:center">'
                     '해당 조건의 종목이 없습니다.</div>')
    parts.append("</div>")
    body = "".join(parts)
    return (_CSS + body) if include_css else body


# ── NiceGUI 렌더러 (라이브 앱 = main.py → tab_stocks.py) ─────────
def render_action_cards_nicegui(df: pd.DataFrame,
                                *,
                                max_cards: Optional[int] = 8,
                                show_pms: bool = False,
                                filter_mode: str = "공식 매수",
                                title: str = "오늘의 매매 액션"):
    """NiceGUI용 액션 카드 레인. tab_stocks.py(render_tab_stocks) drop-in.

    사용:
        from components.action_cards import render_action_cards_nicegui
        render_action_cards_nicegui(df, max_cards=8)   # 🏆 실전 후보 근처

    · 순수 build_cards_html을 재사용하고 ui.html()로 출력(include_css=True).
    · 데이터 없거나 nicegui 미가용이면 조용히 건너뜀(기존 화면 영향 0).
    """
    try:
        from nicegui import ui
    except Exception:
        return  # NiceGUI 환경 아님 — 무시

    if df is None or len(df) == 0:
        return

    # CSS는 <head>에 주입해야 적용됨 (NiceGUI ui.html 내부 <style>는 무시됨).
    # 기존 DARK_CSS와 동일 방식(ui.add_head_html). 스코프(sp-)라 중복 주입도 무해.
    try:
        ui.add_head_html(_CSS)
    except Exception:
        pass

    # 필터 칩(네이티브) — 선택값에 따라 build_cards_html 재호출
    state = {"mode": filter_mode}
    try:
        with ui.column().classes("w-full gap-2"):
            ui.label(title).classes("text-sm font-bold text-slate-300 mt-2")
            html_holder = ui.html("").classes("w-full")

            def _refresh():
                html_holder.set_content(
                    build_cards_html(df, max_cards=max_cards, show_pms=show_pms,
                                     filter_mode=state["mode"], include_css=False)
                )

            with ui.row().classes("gap-1 flex-wrap"):
                labels = ["공식 매수", "관찰 후보"]
                if show_pms:
                    labels.append("PMS 레인")
                for label in labels:
                    def _mk(lb):
                        def _click():
                            state["mode"] = lb
                            _refresh()
                        return _click
                    btn = ui.button(label, on_click=_mk(label))
                    btn.props("flat dense no-caps size=sm").classes("text-xs")
            _refresh()
    except Exception:
        # NiceGUI 컨텍스트 밖에서 호출되는 등 예외 시 무시 (기존 화면 보존)
        return
