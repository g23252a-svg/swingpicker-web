# -*- coding: utf-8 -*-
"""
ui_utils.py — 공통 UI 유틸리티
═══════════════════════════════════════════════════
DARK_CSS, metric_card, section_title, price_bar_html, _plotly_dark
"""
from nicegui import ui

DARK_CSS = """
<style>
    body { background: #0d0d1a !important; }
    .nicegui-content { max-width: 1400px; margin: 0 auto; }
    .q-tab-panel { padding: 8px 0 !important; }
    .q-card { background: #1a1a2e !important; }
    .q-table { background: #1a1a2e !important; color: white !important; }
    .q-table th { color: #94A3B8 !important; }
    .kanban-col { min-width: 280px; flex: 1; }
    .kanban-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 12px; margin-bottom: 8px;
        transition: transform 0.2s; cursor: pointer;
    }
    .kanban-card:hover { transform: translateY(-2px); background: rgba(255,255,255,0.08); }
</style>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<!-- PWA -->
<link rel="manifest" href="/static/manifest.json">
<meta name="theme-color" content="#1a1a2e">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="LDY Trader">
<link rel="apple-touch-icon" href="/static/icon-192.png">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5, user-scalable=yes">
<script>
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/sw.js').then(r => console.log('SW registered'));
}
</script>
"""


def metric_card(title, value, delta="", positive=True):
    with ui.card().classes("p-4 min-w-[140px] bg-[#1a1a2e] border border-gray-700 rounded-xl"):
        ui.label(title).classes("text-xs text-gray-400 uppercase tracking-wide")
        ui.label(str(value)).classes("text-xl font-bold text-white mt-1")
        if delta:
            color = "text-green-400" if positive else "text-red-400"
            ui.label(str(delta)).classes(f"text-sm {color} mt-0.5")


def section_title(text):
    ui.label(text).classes("text-lg font-bold text-white mt-6 mb-2 border-b border-gray-700 pb-2")


def price_bar_html(stop, entry, close, t1, t2=0):
    points = [("손절", stop, "#EF4444"), ("매수", entry, "#3B82F6"), ("현재", close, "#FFFFFF")]
    if t1 > 0: points.append(("T1", t1, "#10B981"))
    if t2 > 0 and t2 != t1: points.append(("T2", t2, "#EAB308"))
    points.sort(key=lambda x: x[1])
    p_min, p_max = points[0][1] * 0.98, points[-1][1] * 1.02
    rng = p_max - p_min
    if rng <= 0: return ""

    html = '<div style="position:relative;height:55px;background:linear-gradient(90deg,rgba(239,68,68,0.15) 0%,rgba(16,185,129,0.15) 100%);border-radius:10px;margin:8px 0 20px 0;">'
    for label, price, color in points:
        pct = max(3, min((price - p_min) / rng * 100, 97))
        is_cur = label == "현재"
        sz = "14px" if is_cur else "10px"
        bdr = "2px solid #FFF" if is_cur else "none"
        fw = "bold" if is_cur else "normal"
        html += (
            f'<div style="position:absolute;left:{pct}%;top:50%;transform:translate(-50%,-50%);z-index:{"10" if is_cur else "5"};text-align:center;">'
            f'<div style="width:{sz};height:{sz};background:{color};border-radius:50%;border:{bdr};margin:0 auto;"></div>'
            f'<div style="font-size:10px;color:{color};white-space:nowrap;margin-top:3px;font-weight:{fw};">{label}<br>{int(price):,}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def plotly_dark(fig, height=300):
    if fig:
        fig.update_layout(
            height=height, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
            margin=dict(t=30, b=10, l=10, r=10),
        )
    return fig
