# SwingPicker Recommendation Engine v23 — 4-Tier 설계서

> **Status**: Phase 0 — 설계 (코드 0줄)
> **Owner**: LDY Trader
> **Last updated**: 2026-05-10
> **Target**: 6주 분할 도입, Phase 5 전까지 사용자 노출 0

---

## 1. 문제 정의

기존 v22.x 추천 시스템의 핵심 한계:

1. **단일 점수 정렬** — DISPLAY_SCORE / ELITE_SCORE 기반 Top-N 추천. 점수 높지만 현재가가 추천매수가보다 +3% 위인 종목도 그대로 추천 → 추격 매수 유도.
2. **즉시 추세 vs 매집 후 폭발 구분 없음** — "내일 갈 종목"과 "1-2주 후 폭발할 종목"이 하나의 점수로 섞여 정렬.
3. **위험 신호 무시** — 윗꼬리 / 거래량 클라이맥스 / 휩쏘 위험은 점수에 반영 안 됨.
4. **Top 3 강제** — 그날 진짜 좋은 후보가 0개여도 3개를 채워서 보여줌.

## 2. 핵심 인사이트

진짜 좋은 스윙 후보는 두 가지로 나뉜다:

| 카테고리 | 본질 | 진입 액션 | 보유 기간 | 평가 horizon |
|---|---|---|---|---|
| **NOW_BUY** | trend onset (이미 추세 시작) | 100% 진입 | 3-5일 | **3-5일** |
| **ACCUMULATION_READY** | compression before expansion (압축 후 폭발 직전) | 분할 매집 (30/40/30) | 7-14일 | **7-14일** |

이 둘을 **다른 horizon으로 검증해야** calibration이 fair.

## 3. 4-Tier 분류 + 결정 트리

```
┌─────────────────────────────────────────────────────┐
│ 1단계 — 수식 무결성 (B 카테고리, 자명)              │
│   TP1 > CURRENT_PRICE                               │
│   CURRENT_PRICE > STOP                              │
│   RR_NOW_TP1 > 0                                    │
│   실패 → ⬜ BLOCKED (UI 비표시, log only)          │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 2단계 — 위험 필터                                   │
│   CHASE_RISK_SCORE > cutoff_chase    → 🔴 NO_CHASE │
│   WHIPSAW_RISK_SCORE > cutoff_whip  → 🔴 NO_CHASE │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 3단계 — 가격 위치                                   │
│   ENTRY_GAP_NOW_PCT > 1.5%  → 🟠 PULLBACK_WAIT     │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│ 4단계 — 변동성 / 추세 레짐 (mutually exclusive)     │
│   TREND_HEAT 상위 30% & RR_NOW_TP1 ≥ 1.5            │
│       → 🟢 NOW_BUY                                  │
│   SQUEEZE_INTENSITY 상위 30% & RR_NOW_TP1 ≥ 1.3    │
│       → 🟡 ACCUMULATION_READY                       │
│   둘 다 아님 → ⬜ ACTIVE_PASS (일반 active, UI 표시 X)│
└─────────────────────────────────────────────────────┘
```

**TOP_PICK은 직교 (orthogonal) 품질 배지**, 액션 등급 아님.

## 4. 임계값 분류 (A/B/C/D)

### A. 운영 검증된 값 재사용 (재결정 불필요)

```
RR_NOW_TP1 >= 1.0          ← v22.3 hard gate
turnover >= 50억원          ← liquidity guard
entry_gap <= 5%             ← scoring_engine
ROUTE in {ATTACK, ARMED}    ← active 정의
EBS PASS                    ← existing
```

### B. 수식 무결성 (자명)

```
TP1 > CURRENT_PRICE > STOP
RR_NOW_TP1 > 0
```

### C. 분포 기반 동적 임계값 (percentile, 1주치 데이터)

```
CHASE_RISK signals       → percentile 평균 (각 신호 백분위 → 단순 평균)
WHIPSAW_RISK signals     → percentile 평균
TREND_HEAT 상위 30%      → 동적 threshold (당일 분포)
SQUEEZE_INTENSITY 상위 30% → 동적 threshold (당일 분포)
```

**고정값 아님 — 매일 분포 다시 계산. 시장 레짐 자동 적응.**

### D. 백테스트 결정 필요 (5개)

```
1. cutoff_chase           — CHASE_RISK_SCORE NO_CHASE 임계값
2. cutoff_whip            — WHIPSAW_RISK_SCORE NO_CHASE 임계값
3. ENTRY_GAP_NOW_PCT > X  — PULLBACK 분기점 (현 1.5% 후보)
4. RR_NOW_TP1 NOW_BUY 추가 마진 — 1.0 / 1.3 / 1.5 / 1.8 sweep
5. RR_NOW_TP1 ACCUMULATION 마진 — 1.0 / 1.2 / 1.3 / 1.5 sweep
```

## 5. Risk Score 정의 (v1: 동일가중 시작)

### Percentile 모집단 정의 (v1)

```
v1: 당일 active 후보군 기준 percentile
    (ROUTE in {ATTACK, ARMED} & EBS PASS 통과 후보 내에서 분포)
v2: 시장 국면별 보정 (regime-aware percentile)
v3: 섹터별 보정 percentile 추가
```

> **v1으로 시작 — 모집단을 "당일 active"로 고정.**
> 전체 KRX 기준이면 너무 broad해서 위험 신호 감도 떨어짐.
> 섹터별이면 표본 부족으로 noise 큼. v1은 이 trade-off의 sweet spot.

### CHASE_RISK_SCORE (0-100)

```
신호 (각각 당일 active 모집단 내 percentile):
  ENTRY_GAP_NOW_PCT       (높을수록 위험)
  ret_1d_pct              (높을수록 위험)
  upper_shadow_ratio      (높을수록 위험)
  1 - close_location      (높을수록 위험, 종가 밀림)
  volume_ratio            (높을수록 위험, 클라이맥스)

v1 (초기): 5개 percentile의 단순 평균
v2 (백테스트 후): 설명력 기반 가중치 조정
```

### WHIPSAW_RISK_SCORE (0-100)

```
신호:
  ATR_PCT                          (높을수록 위험)
  Range_Pos                        (높을수록 위험, 0.85+ 위험)
  1 - STOP_DISTANCE / ATR          (낮을수록 위험, 손절 너무 가까움)

v1: 3개 percentile의 단순 평균
v2: 백테스트 후 가중치 조정
```

> **이전 안의 0.30/0.20/0.20/0.15/0.15 가중치 폐기** — 백테스트 전 임의 가중치 금지.
> 동일가중 percentile 평균이 가장 안전한 출발점.

## 6. SQUEEZE_INTENSITY / TREND_HEAT 정의 (명확화)

### SQUEEZE_INTENSITY (압축 강도, 높을수록 강한 압축)

```
BB_BW_PCTL_250D = 현재 BB_BW가 과거 250일 중 percentile (낮을수록 좁음)
SQUEEZE_INTENSITY = 1 - BB_BW_PCTL_250D
```

이미 있는 컬럼 활용 가능:
- `TTM_SQUEEZE`, `TTM_SQUEEZE_CNT`, `BB_SQUEEZE_BW` (직접 신호)
- `BB_BW` (raw) — percentile 계산 필요

### TREND_HEAT (추세 발현 강도)

```
구성 (각각 percentile 변환):
  MACD_Slope_PCT
  ret_5d_pct
  Above_MA20 (binary, 0 또는 1)
  HMA_Trend (직접 신호)

TREND_HEAT = 4 신호 percentile 평균
```

## 7. 검증 (monotonicity_report v23 확장)

### 카테고리별 calibration 분리

```python
report["tier_calibration"] = {
    "now_buy": {
        "horizon_days": 5,           # ★ horizon 분리
        "n": ...,
        "declared_wr": ...,           # 선언 승률 평균
        "realized_wr_5d": ...,        # 실현 승률 (5일)
        "realized_wr_3d": ...,        # 보너스: 3일도 추적
        "tp1_hit_rate": ...,
        "max_favorable_excursion_5d": ...,
        "initial_stop_rate": ...,
    },
    "accumulation_ready": {
        "horizon_days": 14,           # ★ 다른 horizon
        "n": ...,
        "declared_wr": ...,
        "realized_wr_10d": ...,
        "realized_wr_15d": ...,
        "breakout_rate_10d": ...,
        "breakout_rate_15d": ...,
        "squeeze_release_return": ...,
        "box_breakdown_rate": ...,
    },
    "pullback_wait": { "horizon_days": 7, ... },
    "no_chase": { "horizon_days": 5, ... },
}
```

### HARD gate 추가 (v23)

```python
# H1. 분류 무결성 — NOW_BUY가 ACCUMULATION보다 빨리 가야 정상
ci_hard.append({
    "gate": "now_buy_faster_than_accumulation",
    "check": "now_buy.realized_wr_5d >= accumulation.realized_wr_5d",
    "rationale": "NOW_BUY는 즉시 추세, ACCUMULATION은 압축 → 5d 단기 평가에서 NOW_BUY가 우위여야",
})

# H2. NO_CHASE 필터링 효과 — 제외 후 EV 향상
ci_hard.append({
    "gate": "no_chase_filter_improves_ev",
    "check": "ACTIVE_PASS_no_chase_excluded.ev_5d > ACTIVE_PASS_with_no_chase.ev_5d",
    "rationale": "NO_CHASE 제외로 후보 expected value가 의미 있게 올라가야 필터 정당",
})

# H3. PULLBACK_WAIT calibration
ci_hard.append({
    "gate": "pullback_wait_recovery_rate",
    "check": "pullback_wait 종목이 추천매수가까지 내려와서 재진입 가능해진 비율 >= 30%",
})
```

> **이전 안의 "NO_CHASE realized < BLOCKED 평균" 폐기.**
> BLOCKED는 수식 무결성 실패라 비교군 아님. 진짜 비교는 **NO_CHASE 제외 전 후보 vs 제외 후 후보의 EV 차이**.

## 8. UI 표시 (Phase 5 도입)

### 사용자 화면

```
오늘의 추천 (2026-05-XX)

🟢 즉시진입 후보 (N건)
   1. 알루코 — RR 2.1, 갭 +0.5%, ELITE 79.7  ⭐TOP
      "이미 추세 시작, 손익비 양호. 즉시 100% 진입 가능."

🟡 매집 후보 (M건)
   1. 종목B — BB squeeze 4일째, OBV↑, RR 1.7  ⭐TOP
      "압축 후 변동성 확장 직전. 1차 30% 매집 권장."

🟠 눌림 대기 (K건, 펼치기)
   1. 종목C — 구조는 좋으나 현재가가 매수가보다 +3.2%
      "추격 금지. 추천매수가 근처 재진입 시 다시 평가."

(NOW_BUY 0건일 때)
   "오늘은 즉시진입 후보가 없습니다. 매집 후보 위주로 검토 권장."
```

### 관리자 전용

```
🔴 추격 금지 (NO_CHASE) - X건
   chase_risk_score / whipsaw_risk_score / 트리거된 신호

⬜ BLOCKED - Y건
   수식 무결성 실패 사유

📊 분류 calibration (어제 추천 → 오늘 결과)
   NOW_BUY 5d realized_wr: ...
   ACCUMULATION 10d realized_wr: ...
```

### TOP_PICK 표시 원칙

```
NOW_BUY      = 액션 (지금 사라)
ACCUMULATION = 액션 (지금 매집 시작하라)
TOP_PICK     = 품질 배지 (이 카테고리 안에서 최상위)

표기: "🟡 매집 후보 · ⭐TOP 등급"
액션 메시지: "즉시 추격이 아니라 1차 분할매수/눌림 지지 확인형"
```

## 9. 도입 로드맵 (6주, Phase 5까지 사용자 노출 0)

| Phase | 기간 | 작업 | 위험 | 사용자 노출 |
|---|---|---|---|---|
| **0** | 0 | 본 문서 작성, 컬럼 매핑 분석 | 0 | 없음 |
| **1** | 1주 | scoring_engine.py에 신규 컬럼 추가<br>(ENTRY_GAP_NOW_PCT, RR_NOW_TP1 명확화, upper_shadow_ratio, close_location, ATR_PCT, OBV_Slope, BB_BW_PCTL_250D) | 0 | 없음 |
| **2** | 1주 | recommend_latest.csv에 신규 컬럼 + tier 컬럼 추가<br>(ACTION_TIER, CHASE_RISK_SCORE, WHIPSAW_RISK_SCORE, TREND_HEAT, SQUEEZE_INTENSITY)<br>**추천 로직 변경 0, log only** | 0 | 없음 |
| **3** | 1주 | monotonicity_report v23 확장<br>(tier_calibration with horizon split) | 0 | 없음 |
| **4** | 2주 | 백테스트 — D 카테고리 임계값 5개 결정<br>(per_trade_log.csv + historical recommend_*.csv) | 0 | 없음 |
| **5** | 1주 | UI 변경 — 4-tier 표시<br>**사용자 노출 시작** | 중 | **시작** |
| **6** | 지속 | 실제 매매 결과 vs 분류 비교, fine-tune | 저 | 진행 |

### Phase 5 진입 조건 (gate)

Phase 5는 **다음 5가지 모두 통과**해야 진행:

```
[절대 기준]
✓ NOW_BUY 5d realized_wr >= 50% (구독자에게 보일 만한 수준)
✓ ACCUMULATION 14d realized_wr >= 55% (보유기간 길어 더 높아야)
✓ NO_CHASE 제외 효과 EV +1%p 이상 (필터가 진짜 도움)

[상대 기준 — baseline 대비, 시장 국면 영향 차단]
✓ NOW_BUY 5d EV >= active baseline 5d EV + 1%p
✓ ACCUMULATION 14d EV >= active baseline 14d EV + 1%p
  (냉각장에서 절대 55% 못 넘어도 baseline +1%p면 통과)

[무결성]
✓ tier 무결성 HARD gate 통과 (H1, H2, H3 모두 PASS)
```

**baseline 정의**: 같은 날 ACTIVE_PASS (NO_CHASE 제외, tier 미분류 일반 active) 종목들의 평균 EV. 시장 전체가 약세면 baseline도 낮아지므로, 분류가 진짜 가치 있는지 fair 측정.

**위 조건 미달 시 분류 자체 폐기 또는 재설계.** 그게 100점짜리 시스템의 자세 — 자기 가설을 데이터로 reject 할 준비.

## 10. Risk Acknowledgments

| 리스크 | 영향 | 완화 |
|---|---|---|
| 6주 작업, 8 PR 분할 | 개발 시간 | 작은 PR로 분할, Phase 1-4는 사용자 노출 0 |
| 백테스트 결과 미달 | 분류 폐기 | Phase 0 GDD에 명시, 회피 안 함 |
| Phase 5 사용자 혼란 | 브랜드 영향 | 변경 announcement, 4-tier 의미 설명, FAQ |
| 신규 컬럼 중 데이터 없음 | Phase 1 작업 증대 | Phase 0 마지막에 컬럼 매핑 분석 (다음 단계) |

## 11. Phase 0 다음 액션

1. **컬럼 매핑 분석** — 다음 명령으로 신규 vs 기존 식별:
   ```bash
   head -1 data/recommend_latest.csv | tr ',' '\n' > /tmp/existing_cols.txt
   # 제안 컬럼: ENTRY_GAP_NOW_PCT, RR_NOW_TP1, upper_shadow_ratio,
   #           close_location, ATR_PCT, OBV_Slope, BB_BW
   # 기존 매칭 / 신규 / derive 필요 분류
   ```

2. **per_trade_log.csv 구조 확인** — 백테스트 가능한 데이터인지:
   ```bash
   head -2 data/per_trade_log.csv
   wc -l data/per_trade_log.csv
   # entry_date, exit_date, entry_price, exit_price, max_high_5d, max_high_15d 등 있는지
   ```

3. **scoring_engine.py 현 게이트 검토** — 어디에 4-tier 결정 트리를 넣을지:
   ```bash
   grep -n "TOP_PICK\|ROUTE_ACTIVE\|hard_gate" scoring_engine.py | head -30
   ```

위 3개 분석 후 → Phase 1 PR 시작.

## 12. CURRENT_PRICE 정책 (v0.3 신규)

이 설계의 핵심은 **현재가 기준** 평가입니다. CURRENT_PRICE의 정의가 모호하면 NOW_BUY/PULLBACK 분류가 매일 바뀝니다.

### 출처별 정의

```
CURRENT_PRICE_SOURCE = {
    "realtime",       # 장중 (09:00 ~ 15:30 KST) — KIS API 실시간가
    "close",          # 장마감 후 ~ 다음날 장 시작 전 — 당일 종가
    "delayed",        # 장중인데 실시간 캐시 stale (>5분)
    "premarket_gap",  # 다음날 장 시작 후 5분 — 시초 갭 발생, 다시 realtime으로 갱신
}
```

### 운영 규칙

```python
# pipeline_finalize.py에서 결정
if 09:00 <= now <= 15:30:
    if realtime_cache_age < 5min:
        CURRENT_PRICE = realtime
        SOURCE = "realtime"
    else:
        CURRENT_PRICE = realtime  # 사용은 하되 표시 변경
        SOURCE = "delayed"
elif 15:30 < now < 09:00 next_day:
    CURRENT_PRICE = close
    SOURCE = "close"
elif 09:00 next_day <= now < 09:05 next_day:
    CURRENT_PRICE = realtime  # 시초가
    SOURCE = "premarket_gap"
```

### UI 표시 의무

```
CURRENT_PRICE_SOURCE != "realtime" 일 때:
  추천 카드 상단에 배지: "전일 종가 기준" 또는 "5분 지연 시세"
  
사용자에게 "지금 사라"는 액션을 줄 때:
  반드시 SOURCE 표시 (잘못된 가격으로 진입 결정 방지)
```

### 분류 영향

```
NOW_BUY 진입은 SOURCE == "realtime" 일 때만 표시
SOURCE == "close" 일 때는 NOW_BUY 후보를 "내일 장 시작 후 재평가"로 강등
```

---

## 13. 컬럼 매핑 템플릿 (Phase 0 마지막 작업)

Phase 1 PR 작성 전 반드시 이 표를 채워서 별도 파일(`docs/RECOMMEND_v23_COLUMN_MAPPING.md`)로 저장.

### GDD 제안 컬럼 → 현재 시스템 매핑

| GDD 제안 컬럼 | 추정 기존 컬럼 | 존재? | 신규 계산 필요 | fallback | 담당 모듈 |
|---|---|---|---|---|---|
| `ENTRY_GAP_NOW_PCT` | `ENTRY_GAP_PCT`? | ? | 재계산 (현재가 vs 추천매수가) | `\|close/추천매수가 - 1\|` | scoring_engine.py |
| `RR_NOW_TP1` | `RR_NOW_TP1` | ✓ | 정의 검증 (이미 v22.3.1 hard gate에 사용) | — | scoring_engine.py |
| `upper_shadow_ratio` | `Upper_Shadow_Ratio`? | ? | OHLC에서 derive | `(high - max(open,close)) / (high - low)` | indicator calc |
| `close_location` | `Range_Pos`? | ✓ (이름 다름?) | 정의 일치 확인 | `(close - low) / (high - low)` | indicator calc |
| `ATR_PCT` | `ATR`? | ? | derive: `ATR / close * 100` | — | indicator calc |
| `OBV_Slope` | `OBV_Div` (다른 정의) | ✗ | 5일 OBV 회귀 기울기 | — | indicator calc |
| `BB_BW_PCTL_250D` | `BB_BW`, `BB_SQUEEZE_BW` | ✓ raw 있음 | 250d percentile 계산 | — | indicator calc |
| `MACD_Slope_PCT` | `MACD_Slope_PCT` | ✓ | 정의 검증 | — | indicator calc |
| `ret_5d_pct` | `ret_5d_%` | ✓ | — | — | scoring_engine.py |
| `Above_MA20` | `Above_MA20` | ✓ | — | — | scoring_engine.py |
| `HMA_Trend` | `HMA_Trend` | ✓ | — | — | scoring_engine.py |
| `volume_ratio` | `V_POWER`? `Vol_Quality`? | ? | 정의 결정 (5일 평균 대비?) | `volume / volume_5d_mean` | indicator calc |
| `STOP_DISTANCE` | `손절가` 있음 | derive | `(close - 손절가) / close * 100` | — | scoring_engine.py |
| `TREND_HEAT` | 신규 합성 | ✗ | 4 신호 percentile 평균 | — | scoring_engine.py |
| `SQUEEZE_INTENSITY` | 신규 | ✗ | `1 - BB_BW_PCTL_250D` | — | scoring_engine.py |
| `CHASE_RISK_SCORE` | 신규 | ✗ | 5 신호 percentile 평균 (v1 동일가중) | — | scoring_engine.py |
| `WHIPSAW_RISK_SCORE` | 신규 | ✗ | 3 신호 percentile 평균 (v1 동일가중) | — | scoring_engine.py |

### Phase 0 마지막 분석 명령

```bash
cd ~/swingpicker-web

# 1. 기존 컬럼 전수 조사
head -1 data/recommend_latest.csv | tr ',' '\n' | nl | grep -iE "atr|obv|range|shadow|location|gap|rr_|macd|ma20|hma|squeeze|bb_|vwap|vol|trigger|chase|whipsaw"

# 2. scoring_engine.py에서 이미 계산되는 신호
grep -nE "Upper_Shadow|Range_Pos|ATR\b|OBV|BB_BW|MACD_Slope|HMA_Trend|TTM_SQUEEZE" scoring_engine.py | head -30

# 3. indicator 계산 모듈 위치 확인
find . -name "*.py" | xargs grep -l "def.*atr\|def.*upper_shadow\|def.*obv" 2>/dev/null | head
```

### 매핑 결과로 분기되는 결정

- **존재 컬럼만으로 충분** → Phase 1 단순 (signal compose만 추가, 1주 → 3일)
- **3-4개 신규 derive 필요** → Phase 1 정상 (1주)
- **6개 이상 신규 + indicator 모듈 작업** → Phase 1을 1A/1B 분할 (1주 + 1주)

---

## 14. 액션 사유 기록 컬럼 (v0.3 신규)

각 종목 row에 분류 결과뿐 아니라 **사유**를 함께 저장. Phase 5 이전 검증 + 운영 디버깅에 필수.

### 신규 컬럼 (recommend_latest.csv 추가)

```python
ACTION_TIER          # str: "NOW_BUY" / "ACCUMULATION_READY" / "PULLBACK_WAIT" /
                     #      "NO_CHASE" / "BLOCKED" / "ACTIVE_PASS"
ACTION_TIER_REASON   # str: 어느 분기에서 결정됐는지 한 줄
BLOCK_REASON         # str | "" : BLOCKED일 때만 채움
CHASE_RISK_SCORE     # float 0-100
CHASE_RISK_REASONS   # str: 점수 기여 신호 (예: "high_gap, large_upper_shadow")
WHIPSAW_RISK_SCORE   # float 0-100
WHIPSAW_RISK_REASONS # str: 기여 신호
TREND_HEAT           # float 0-100
SQUEEZE_INTENSITY    # float 0-100
ENTRY_GAP_NOW_PCT    # float
CURRENT_PRICE_SOURCE # str: "realtime" / "close" / "delayed" / "premarket_gap"
```

### 예시 row

```csv
종목명, ACTION_TIER, ACTION_TIER_REASON, CHASE_RISK_SCORE, CHASE_RISK_REASONS, ...
알루코, NOW_BUY, "trend_heat=82 (top 30%) AND rr=2.1 >= 1.5", 23, "low risk", ...
종목B, ACCUMULATION_READY, "squeeze=78 (top 30%) AND rr=1.7 >= 1.3", 31, "moderate", ...
종목C, PULLBACK_WAIT, "entry_gap_now=3.2% > 1.5%", —, —, ...
종목D, NO_CHASE, "chase_score=87 > cutoff=75", 87, "high_gap, ret_1d=8%, vol_climax", ...
종목E, BLOCKED, "TP1<=CURRENT_PRICE (수식 무결성)", —, —, ...
```

### 운영 효과

- **관리자 검증**: `ACTION_TIER_REASON` 한 줄로 "왜 이 분류가 됐는지" 즉시 파악
- **사용자 cross-check**: 추천 카드 우측에 "왜 즉시진입?" 토글로 사유 표시
- **백테스트 분석**: tier 분류와 reason 별 historical 성과 분리 분석 가능
- **회귀 디버깅**: 어제 NOW_BUY → 오늘 PULLBACK 변경 시 사유 비교

### REASON 문자열 규칙

```
짧고 정형화 — 분석 가능하게
  GOOD: "high_gap=3.2%, ret_1d=8.5%, vol_climax"
  BAD:  "현재가가 너무 올랐고 어제도 많이 올라서 위험해 보임"

키 정규화 (소문자 snake_case):
  high_gap / large_upper_shadow / weak_close / vol_climax
  high_atr / extreme_range_pos / tight_stop
```

---

## 변경 이력

- **2026-05-10 v0.1** — 초안 (Claude 제안)
- **2026-05-10 v0.2** — 5개 보정 적용 (사용자 리뷰)
  - Risk Score 가중치 폐기 → 동일가중 percentile 평균
  - NO_CHASE 검증 비교군 변경 (BLOCKED → ACTIVE_PASS)
  - horizon 분리 (NOW_BUY 5d, ACCUMULATION 14d)
  - SQUEEZE_INTENSITY 정의 명확화 (1 - BB_BW_PCTL)
  - TOP_PICK = 품질 배지, 액션 아님 명시
- **2026-05-10 v0.3** — 5개 보완 적용 (사용자 2차 리뷰)
  - Section 5: percentile 모집단 v1=당일 active 명시
  - Section 9: Phase 5 gate에 baseline 상대성과 조건 2개 추가 (시장 국면 영향 차단)
  - **Section 12 신규**: CURRENT_PRICE_SOURCE 정책 (realtime/close/delayed/premarket_gap)
  - **Section 13 신규**: 컬럼 매핑 템플릿 (Phase 1 진입 전 필수)
  - **Section 14 신규**: 액션 사유 기록 컬럼 (ACTION_TIER_REASON, CHASE_RISK_REASONS 등)
