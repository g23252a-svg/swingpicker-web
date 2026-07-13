# SwingPicker v26.1 — Profit Recovery

한국 주식 스윙 후보를 수집·평가하고 NiceGUI로 보여주는 퀀트 애플리케이션입니다.

## v26.1 핵심 원칙

- 공식 신규매수는 `PRODUCTION_BUY=1` 한 가지 값만 사용합니다.
- 좋은 후보가 없으면 `CASH`가 정상 결정입니다. 종목 수를 억지로 채우지 않습니다.
- `TOP_PICK`, `LDY_RANK`, 높은 AI 점수, 관찰 레인은 단독으로 매수 추천이 아닙니다.
- 머신러닝은 시간순 OOS 검증(AUC·Brier·Top-decile lift)을 모두 통과해야만 점수에 반영됩니다.
- 검증된 기술·유동성 셋업은 기존 `TOP_PICK` 플래그와 무관하게 후보가 될 수 있습니다.
- 시장 상승종목 비율이 30% 미만이면 공식 매수는 차단하고 실제 셋업을 관찰 후보로 보여줍니다.
- 공식 손절은 -6%, 목표는 +10%, 종목당 권장 최대 비중은 NORMAL 3%/CAUTION 2%입니다.
- 하루 신규진입은 최대 1종목입니다.
- 종목탭은 `공식 매수 / 관찰 후보(매수 아님) / 전체 분석`을 분리해 보여주며, 매수 0개인 날에도 관찰 후보와 차단 사유를 첫 화면에 표시합니다.

## 손실 진단 결과

2026-07-10까지 저장된 고유 Top-3 체결 표본 기준 평균 수익률은 `-2.70%`, Top-1은 `-2.82%`였습니다. 기존 ML/AI 점수와 실현수익의 Spearman 상관은 `0.005`로 사실상 예측력이 없었고, AGGRESSIVE 표본은 체결 7건 모두 손실(평균 `-13.60%`)이었습니다.

v26.1은 추가로 지정가 3일 체결, 10거래일 경로, 왕복비용 0.22%, 동일 봉 손절 우선을 적용한 시간순 검증에서 Train/Test 평균이 모두 양수였던 규칙만 제한 배포합니다. 2026-04-10~06-29 구간에서 42신호/35체결, Train 체결 평균 +2.16%, 최근 Test +0.50%였습니다. 다만 승률과 부트스트랩 신뢰구간을 고려하면 미래 수익을 보장할 수 없으므로 비중을 2~3%로 제한합니다.

## 실행

```bash
python -m pip install -r requirements_nicegui.txt
export STORAGE_SECRET="replace-with-a-long-random-secret"
python main.py
```

Windows PowerShell에서는 `export` 대신 `$env:STORAGE_SECRET="..."`를 사용합니다.

기존 종목탭을 긴급 복구해야 할 때만 `USE_LEGACY_STOCK_TAB=1`을 설정할 수 있습니다. 기본값은 새 의사결정형 종목탭입니다.

## 검증

```bash
python -m pytest -q
python scripts/analyze_realized_edge.py --data-dir data
python scripts/validate_profit_recovery_v261.py
```

새 모델을 학습하면 `data/trading_meta_v19.json`의 `validation.reliable`을 확인하세요. `false`이거나 검증 메타가 없으면 모델 원시값은 `ML_RAW_SCORE`에만 남고 추천 가중치는 0입니다.

## Railway

`Dockerfile`과 `railway.toml`을 사용합니다. Railway 서비스에는 최소한 `STORAGE_SECRET`을 설정해야 합니다. No-Buy Breaker는 기본적으로 꺼져 있으며, 손실방어 정책상 `ENABLE_NO_BUY_BREAKER`를 설정하지 않는 것을 권장합니다.

자세한 변경 내용은 [docs/PROFIT_RECOVERY_V261.md](docs/PROFIT_RECOVERY_V261.md)를 참고하세요.
