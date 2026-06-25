# v3.9.29 Profit Recovery Suite

목적: 최근 손실 구간에서 반복된 **추격매수 손절 조합**을 더 강하게 차단하고, 공식 신규매수가 0개인 날에도 관찰 우선순위와 비중 축소 기준을 보여주는 큰 패치입니다.

## 핵심 변경

### 1) 손실 조합 하드 차단
과거 `recommend_YYYYMMDD.csv`와 `backtest_top3_trades_20260624.csv`를 조인해 본 결과, 아래 조합은 손절 비율이 매우 높았습니다.

- `ret_5d_% >= 5` + `VWAP_GAP >= 20`
- `ret_5d_% >= 5` + `POC_GAP >= 60`
- `ret_5d_% >= 10` + `VWAP_GAP >= 15`
- `ret_5d_% >= 10` + `POC_GAP >= 40`
- `ret_5d_% <= -10` + `MARKET_BREADTH < 45`

v3.9.29는 이 조합을 `PROFIT_RECOVERY_BLOCK_FLAG=1`로 표시하고, 신규진입 후보라면 `TOP_PICK/BUY_NOW_ELIGIBLE/BUY_NOW_PASS/IS_NOW_ENTRY`를 0으로 낮춥니다.

### 2) 회복 후보 점수화
새 컬럼:

- `PROFIT_RECOVERY_SCORE`: 0~100 회복장 대응 점수
- `PROFIT_RECOVERY_TIER`: `A / B / C / BLOCK`
- `PROFIT_RECOVERY_SETUP`: `PULLBACK_BREADTH / QUALITY_BREADTH / REALISTIC_RR / WATCH / FOMO_COLLISION / WEAK_KNIFE / GUARD_BLOCK`
- `PROFIT_RECOVERY_BLOCK_FLAG`: 신규진입 차단 여부
- `PROFIT_RECOVERY_SIZE_MULT`: 추천금액/수량에 곱하는 안전 multiplier, 최대 0.70
- `PROFIT_RECOVERY_ACTION`: `최우선 관찰 / 소액 후보 / 관찰 후보 / 신규진입 차단`
- `PROFIT_RECOVERY_REASON`: 사람이 읽는 사유

### 3) 공식 매수 승격 금지
이 패치는 좋은 후보를 공식 매수로 승격하지 않습니다. 오직 손실 위험 후보를 차단하고, 관찰 후보의 우선순위와 비중만 조정합니다.

### 4) 정렬 보조축 추가
`finalize_sort()`가 `JULY_PROFIT_DEFENSE_SCORE` 다음에 `PROFIT_RECOVERY_SCORE`를 반영합니다.

## 운영 원칙

- 공식 신규매수는 여전히 `TOP_PICK + BUY_NOW_ELIGIBLE`만 인정합니다.
- `PROFIT_RECOVERY_TIER=A`는 공식 매수가 아니라 “가장 먼저 볼 관찰 후보”입니다.
- `PROFIT_RECOVERY_BLOCK_FLAG=1`은 신규진입 금지입니다.
- `PROFIT_RECOVERY_SIZE_MULT`가 0.70을 넘지 않도록 하여 7월 회복 시도에서도 풀베팅을 막습니다.

## 리포트 생성

```bash
python scripts/build_profit_recovery_report.py
```

생성 파일:

- `reports/profit_recovery_current_top50.csv`
- `reports/profit_recovery_summary.json`

## 테스트

```bash
python -m pytest -q tests/test_profit_recovery_suite_v3929.py tests/test_july_profit_defense.py test_policy_consistency.py test_route_contract_v22.py
```

검증 결과: 51 passed.
