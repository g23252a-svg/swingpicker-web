# SwingPicker v3.9.28 July Profit Defense Patch

## 목적
최근 손실이 큰 구간에서 `점수 상위`보다 `손실 차단`을 우선하도록 Stage 6 최종 저장 직전에 방어 레이어를 추가했습니다.

## 핵심 진단
- 최신 Top1 백테스트: n=67, 승률 26.9%, 평균 -1.58%, 합계 -105.61%.
- 최신 Top3 백테스트: n=163, 승률 25.8%, 평균 -2.17%, 합계 -353.28%.
- Top3 월별: 5월 평균 -5.07%, 6월 평균 -5.91%.

## 추가된 룰
- `JULY_PROFIT_DEFENSE_SCORE`: 0~100 방어 점수. 정렬 보조축으로 사용.
- `JULY_PROFIT_PROFILE_PASS`: 최근 손실국면에서 상대적으로 양호했던 조합 통과 여부.
- `JULY_PROFIT_BLOCK_FLAG`: 신규진입 차단 여부.
- `JULY_PROFIT_DEFENSE_LEVEL`: PASS / CAUTION / BLOCK.
- `JULY_PROFIT_DEFENSE_REASON`: 사람이 읽는 차단/주의 이유.

## 생산 차단 조건
- ret_5d_% > 20
- VWAP_GAP > 35
- POC_GAP > 80
- MARKET_BREADTH < 35 이면서 ret_5d_% > 5
- ABNORMAL_HISTORY_GUARD_FLAG 또는 SPIKE_REVERSAL_GUARD_FLAG
- ENTRY_EDGE_LEVEL == BLOCK
- ENTRY_RISK_LEVEL == ORANGE 이면서 ret_5d_% > 10

## 7월 방어 profile
- MARKET_BREADTH ≥ 45
- ret_5d_% ≤ 5
- Vol_Quality ≥ 1.2
- STRUCT_SCORE ≥ 80 또는 FINAL_SCORE ≥ 75
- VWAP_GAP ≤ 20
- POC_GAP ≤ 60
- abnormal/spike/entry edge block 없음

## 현재 recommend_latest 적용 결과
- PASS profile: 0건
- BLOCK: 56건
- 신규진입 차단 표시: 39건
- 공식 신규매수: 0 → 0건

## 테스트
- `PYTHONPATH=. pytest -q tests/test_july_profit_defense.py test_policy_consistency.py test_route_contract_v22.py`
- 결과: 47 passed
