# 🚀 SwingPicker Phase 2 통합 가이드

## 파일 구성

```
신규 파일 (4개):
├── components/tab_portfolio_v2.py  ← 📜 DART 공시 AI 진단 통합 포트폴리오 탭
├── viz_components.py               ← 📊 고급 시각화 (히트맵/캘린더/트리맵)
├── trailing_stop.py                ← 🛡️ 트레일링 스탑 로직
├── telegram_sender_v2.py           ← 📱 확장형 텔레그램 알림
```

---

## 1. 포트폴리오 AI 진단 (tab_portfolio_v2.py)

### 변경 사항
- 기존 `tab_portfolio.py`의 모든 기능 유지
- DART 공시 자동 조회 + Gemini AI 재무 리스크 진단 추가
- 종합 포트폴리오 AI 리포트 (Gemini) 자동 생성
- 접이식 공시 상세 뷰 + 리스크 배지 표시

### 적용 방법

**main.py** 임포트 변경:
```python
# 기존
from components.tab_portfolio import render_tab_portfolio
# 변경
from components.tab_portfolio_v2 import render_tab_portfolio
```

나머지는 동일 — `render_tab_portfolio(df, auth)` 시그니처 호환.

### 필요 환경변수 (Railway)
```
GEMINI_API_KEY=xxx    # 이미 설정됨
DART_API_KEY=xxx      # 이미 설정됨
```

---

## 2. 시각화 대시보드 (viz_components.py)

### 제공 차트 6종
| 함수 | 용도 | 적용 탭 |
|------|------|---------|
| `plot_equity_curve()` | 누적 수익곡선 + MDD | Tab 7, 10 |
| `plot_monthly_heatmap()` | 월별/요일별 수익 히트맵 | Tab 7, 9 |
| `plot_calendar_pnl()` | GitHub 잔디 스타일 일별 PnL | Tab 9 |
| `plot_score_winrate()` | 점수 구간별 승률 바차트 | Tab 7, 10 |
| `plot_daily_dual_axis()` | 일별 거래수 + 수익률 이중축 | Tab 10 |
| `plot_sector_treemap()` | 섹터별 수익 기여 트리맵 | Tab 7 |

### 적용 예시: tab_perf.py에 히트맵 추가

```python
from viz_components import plot_monthly_heatmap, plot_score_winrate, render_viz_section

def render_tab_perf():
    # 기존 코드...
    
    # 월별 히트맵 추가
    if not history.empty and 'profit_pct' in history.columns:
        fig_heatmap = plot_monthly_heatmap(history, date_col="Date", profit_col="profit_pct")
        ui.plotly(fig_heatmap).classes("w-full")
```

### 적용 예시: tab_backtest.py에 승률 차트 추가

```python
from viz_components import plot_score_winrate, plot_daily_dual_axis

# _run_backtest() 결과에서 trades DataFrame 활용
fig_winrate = plot_score_winrate(trades_df, score_col="score", profit_col="net_ret")
ui.plotly(fig_winrate).classes("w-full")
```

### 적용 예시: trade_journal_tab.py에 캘린더 추가

```python
from viz_components import plot_calendar_pnl, plot_monthly_heatmap

# 매매 일지 데이터에서
fig_cal = plot_calendar_pnl(journal_df, date_col="created_at", profit_col="profit_pct")
ui.plotly(fig_cal).classes("w-full")

fig_weekday = plot_monthly_heatmap(journal_df, mode="weekday")
ui.plotly(fig_weekday).classes("w-full")
```

---

## 3. 트레일링 스탑 (trailing_stop.py)

### 3가지 모드
| 모드 | 설명 | 프리셋 함수 |
|------|------|-------------|
| `fixed` | 고점 대비 고정 비율 하락 시 청산 | `config_conservative()`, `config_aggressive()` |
| `atr` | ATR 배수 기반 동적 트레일 | `config_atr_based()` |
| `step` | 수익 구간별 계단식 보호 | `config_step()` |

### tab_backtest.py 통합

```python
from trailing_stop import TrailingStopConfig, apply_trailing_to_backtest, compare_strategies

def render_tab_backtest(df, auth):
    # 기존 파라미터 패널에 추가:
    with ui.column():
        ui.label("🛡️ 트레일링 스탑").classes("text-sm font-bold text-purple-400 mb-2")
        trail_toggle = ui.checkbox("트레일링 스탑 활성화", value=False)
        trail_mode = ui.select(["fixed", "atr", "step"], value="fixed", label="모드")
        sl_activation = ui.slider(min=2, max=10, value=3, step=1)
        ui.label("").bind_text_from(sl_activation, "value",
                                     backward=lambda v: f"활성화 수익률: +{v}%")
        sl_trail = ui.slider(min=1, max=8, value=3, step=0.5)
        ui.label("").bind_text_from(sl_trail, "value",
                                     backward=lambda v: f"트레일 거리: {v}%")
```

### _run_backtest() 내부 수정

```python
# 기존:
if raw_ret <= -stop_pct:
    applied_ret = -stop_pct; status = "STOP"
elif raw_ret >= target_pct:
    applied_ret = target_pct; status = "WIN"
else:
    applied_ret = raw_ret; status = "HOLD_EXIT"

# 변경 (트레일링 활성화 시):
if use_trailing:
    from trailing_stop import apply_trailing_to_backtest, TrailingStopConfig
    cfg = TrailingStopConfig(
        mode=trail_mode, activation_pct=activation_pct,
        trail_pct=trail_pct,
    )
    applied_ret, status = apply_trailing_to_backtest(
        raw_ret, stop_pct=stop_pct, target_pct=target_pct, cfg=cfg
    )
else:
    # 기존 로직 유지
```

### 비교 분석 (기존 vs 트레일링)

```python
from trailing_stop import compare_strategies, TrailingStopConfig

result = compare_strategies(
    returns=[r["raw_ret"] for r in trades],
    stop_pct=5.0, target_pct=10.0,
    trailing_cfg=TrailingStopConfig(mode="step"),
)
# result["fixed"]     → 기존 전략 통계
# result["trailing"]  → 트레일링 전략 통계
# result["improvement"] → 개선폭
```

---

## 4. 텔레그램 자동화 (telegram_sender_v2.py)

### 기존 호환 (drop-in 교체)
```python
# 기존과 100% 호환
from telegram_sender_v2 import send_telegram_auto
```

### 확장 함수로 교체 (collector.py)

```python
# collector.py Step 9 수정
# 기존:
send_telegram_auto(df_out, trade_ymd, market_summary=summary_text, limit_count=rec_limit_cnt)

# 변경:
from telegram_sender_v2 import send_telegram_enhanced

send_telegram_enhanced(
    df_out, trade_ymd,
    market_summary=summary_text,
    market_temp=mkt_temp,
    breadth=breadth,
    macro_msg=macro_msg,
    leading_sectors=list(top_sectors) if 'top_sectors' in dir() else None,
    blocked_stocks=blocked_list,  # Hard Block 차단 목록
    limit_count=rec_limit_cnt,
    send_briefing=True,   # Top 3 브리핑 별도 알림
    send_blocks=True,     # Hard Block 결과 알림
)
```

### collector.py에서 blocked_list 수집
```python
# collector.py의 Hard Block 처리 부분에서:
blocked_list = []
for idx, row in df_candidates.iterrows():
    defense = check_entry_defense(row)
    if defense["action"] == "hold":
        blocked_list.append({
            "name": row.get("종목명", ""),
            "code": str(row.get("종목코드", "")).zfill(6),
            "block_reason": defense["reason"],
            "score": float(row.get("DISPLAY_SCORE", 0)),
        })
```

---

## 배포 체크리스트

### GitHub (백엔드)
- [ ] `components/tab_portfolio_v2.py` 추가
- [ ] `viz_components.py` 프로젝트 루트에 추가
- [ ] `trailing_stop.py` 프로젝트 루트에 추가
- [ ] `telegram_sender_v2.py` 프로젝트 루트에 추가
- [ ] `main.py` 임포트 변경 (tab_portfolio → tab_portfolio_v2)
- [ ] `collector.py` 텔레그램 발송 부분 수정

### Railway (프론트엔드)
- [ ] 환경변수 확인 (GEMINI_API_KEY, DART_API_KEY)
- [ ] `git push` → 자동 빌드 확인
- [ ] Tab 3 (내 자산) DART 연동 테스트
- [ ] 텔레그램 알림 수신 확인

### requirements_nicegui.txt 변경 사항
추가 패키지 없음 — 모든 의존성은 이미 기존 requirements에 포함:
- `plotly` ✅
- `opendartreader` ✅
- `google-genai` ✅
- `pandas`, `numpy` ✅
