# v32 — ROUTE 게이트 대개혁: 알파 전면 진입 게이트

## 배경 — 사용자의 의심이 데이터로 확인됨

"armed, attack, wait 자체에 대한 로직이 나는 의심스러워서."

ROUTE(ATTACK/ARMED/WAIT)는 TTM 스퀴즈·MACD 기울기·레인지 위치·거래품질 등으로
"돌파 임박/돌입" 상태를 판정하는 모멘텀 택소노미다. TOP_PICK은 `ROUTE ∈ {ARMED, ATTACK}`,
공식매수(PRODUCTION_BUY)는 `ROUTE == ATTACK`을 필수로 요구해 왔다.

## 실증 — ROUTE는 역신호다

3~7월(≈100 거래일) 워크포워드 OOS 측정. 각 종목의 t+1 시가 진입 → t+1+5 종가
선행수익을 실측하고, 같은 날 유니버스 평균 대비 알파로 평가(시장 방향 효과 제거).

| 신호 | 표본 | 시장 대비 알파 | 승률 | t-stat (p) |
|---|---|---|---|---|
| **ROUTE ATTACK** (최강 매수) | 248 | **-2.91%p** | 33% | **-3.56 (p=0.0004)** |
| ROUTE ARMED | 2,168 | +0.00%p | 46% | +0.02 (p=0.99, 노이즈) |
| ELITE_SCORE (랭킹 점수) | 27,593 | IC **-0.063** | — | **-5.30** |

- ATTACK은 5개월 **전부** 음수(-2.6/-1.4/-4.8/-3.9/-3.8%p), 상승·하락 국면 모두 음수
  (상승국면 -3.01%p·승률 28%로 오히려 더 나쁨) — "약세장 모멘텀 크래시" 핑계 불가.
- ELITE_SCORE IC는 상승국면에서 가장 강하게 뒤집힘(-0.0585, t=-5.19). "점수 높을수록
  수익 낮음"이 통계적으로 확정 — 착시가 아니라 실재.

## 대안 검증 — 알파 모델은 진짜 작동한다

동일 잣대(워크포워드 OOS, 미래정보 미사용, 프로덕션 클린 피처 95개):

- OOS IC **+0.19** (t=6.8), AUC 0.60, Q5-Q1 스프레드 **+5.05%p**
- 십분위 실측 승률 **24% → 41% 단조** 증가

반사실 top-K(44 OOS일, 유니버스 평균 -4.26%인 험한 장):

| 선택 방식 | 상위 3 실현수익 | 승률 | vs 유니버스 |
|---|---|---|---|
| **알파(ML) top3** | **+0.81%** | 41% | **+5.07%p** |
| ELITE top3 | -1.63% | 30% | +2.63%p |
| **ROUTE ATTACK/ARMED** | **-5.91%** | — | **-1.65%p** |
| 유니버스(무작위) | -4.26% | — | 0 |

알파만 유일하게 플러스. 현재 게이트인 ROUTE ATTACK/ARMED는 무작위보다도 나쁘다.

## 개혁 내용

### 1. ROUTE 거부권 제거 (4곳)

- `scoring_engine.compute_elite_score`: `_hard_gate`에서 `_route_active` 제거.
  리스크 가드만 남긴 `ENTRY_RISK_GATE_OK` 컬럼 신설(자리·손익비≥1.0·유동성≥50억·
  진입갭≤5%·POC≤20).
- `pipeline_finalize`: route-cap 후 TOP_PICK strip은 알파 게이트가 나중에 재정의하므로
  무해화(순서상 알파가 최종 SSOT).
- `services/recommendation_quality`: `route.isin(ACTIVE_ROUTES)` 거부권 제거.
  POLICY_VERSION `high_prob_v27` → `alpha_gate_v32`.

### 2. 알파 전면 진입 게이트 (`alpha_engine.apply_alpha_entry_gate`)

검증 통과(ALPHA_VALIDATED==1)일 때:

```
TOP_PICK = ENTRY_RISK_GATE_OK
           AND ALPHA_SCORE ≥ 레짐문턱
           AND ~손실방어차단(NEW_ENTRY_BLOCKED/JULY/RECOVERY)
BUY_NOW_ELIGIBLE = TOP_PICK AND BUY_NOW_PASS
```

- ROUTE는 건드리지 않는다(타이밍 배지로 존치).
- 미검증이면 TOP_PICK 유지 + 레거시 폴백 게이트(단, 폴백에도 ROUTE 거부권 없음).

### 3. 레짐 적응형 선별 강도 (사용자 선택: 적응형·반감 사이즈)

| 레짐 | 알파 문턱 | 선별 강도 | 사이즈 배수 |
|---|---|---|---|
| UP (상승) | ALPHA_SCORE ≥ 70 | 상위 30% | 1.0 |
| NEUTRAL (중립) | ≥ 80 | 상위 20% | 0.5 |
| DOWN (하락) | ≥ 90 | 상위 10% | 0.3 (기존 0.0) |

- DOWN 사이즈 0.0→0.3: 내부약세(breadth<35)에서도 알파 최상위 픽은 +0.84%/승률51%로 양호.
  recommendation_quality 알파 경로에서 breadth/regime 하드블록 제거, 문턱+사이즈로 대응.
- **risk_off(KOSPI<하락MA20 -3%)는 하드블록 유지** — 실측상 risk_off 알파 픽 승률 17%·
  절대손실 -2.9%. NEW_ENTRY_BLOCKED가 별도로 강제.

## 효과 (7/15 실데이터 end-to-end)

- 7/15(risk_off): TOP_PICK 0 — **정상**(risk_off 방어, 데이터 지지). 단 ROUTE·불가능한
  승률 게이트가 아니라 검증된 방어 사유로만 0.
- risk_off 해제 가정 시: 알파 픽 38개 발생(전부 ROUTE=WAIT 대형주), 공식매수 = **이수화학
  (알파 91.8, ROUTE=WAIT, RR 1.99)** — ROUTE 시절 ATTACK→WAIT 강등으로 영구 차단되던
  종목이 알파로 선별됨.

## 한계·후속

- risk_off는 KOSPI MA20 기준의 크루드 지표라 회복장 진입에 1.5~3주 지연 가능. 별도 개선 대상.
- DOWN 레짐 알파 픽의 표본이 작음(관찰 지속 필요).
- 백테스트 하네스(backtest_validation)는 '🏆 최강' 라벨 + 알파 바닥 필터를 계속 사용 —
  알파 primary 정렬로의 정렬은 후속.
