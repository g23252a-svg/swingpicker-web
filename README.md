# SwingPicker v26 — Loss Defense

한국 주식 스윙 후보를 수집·평가하고 NiceGUI로 보여주는 퀀트 애플리케이션입니다.

## v26 핵심 원칙

- 공식 신규매수는 `PRODUCTION_BUY=1` 한 가지 값만 사용합니다.
- 좋은 후보가 없으면 `CASH`가 정상 결정입니다. 종목 수를 억지로 채우지 않습니다.
- `TOP_PICK`, `LDY_RANK`, 높은 AI 점수, 관찰 레인은 단독으로 매수 추천이 아닙니다.
- 머신러닝은 시간순 OOS 검증(AUC·Brier·Top-decile lift)을 모두 통과해야만 점수에 반영됩니다.
- 종목당 권장 최대 비중은 5%, 하루 신규진입은 최대 1종목입니다.

## 손실 진단 결과

2026-07-10까지 저장된 고유 Top-3 체결 표본 기준 평균 수익률은 `-2.70%`, Top-1은 `-2.82%`였습니다. 기존 ML/AI 점수와 실현수익의 Spearman 상관은 `0.005`로 사실상 예측력이 없었고, AGGRESSIVE 표본은 체결 7건 모두 손실(평균 `-13.60%`)이었습니다.

따라서 v26은 검증되지 않은 ML 가중치를 0으로 만들고, 공격형 후보와 진입 품질 미달 후보를 최종 공식 추천에서 제외합니다. 이는 수익을 보장하는 변경이 아니라 확인된 손실 경로를 우선 차단하는 변경입니다.

## 데이터 신뢰성 패치 (2026-09-06)

새로고침의 원격 갱신, 실제 기준일 표시, CSV 검증·원자적 캐시 교체, 비정상 가격·손익비 차단, 과거 시점 캐시 검증을 개선했습니다. [변경 내용과 검증 범위](docs/SNAPSHOT_RELIABILITY_PATCH.md)를 확인하세요.

## 실행

```bash
python -m pip install -r requirements_nicegui.txt
export STORAGE_SECRET="replace-with-a-long-random-secret"
python main.py
```

Windows PowerShell에서는 `export` 대신 `$env:STORAGE_SECRET="..."`를 사용합니다.

## 검증

```bash
python -m pytest -q
python scripts/analyze_realized_edge.py --data-dir data
```

새 모델을 학습하면 `data/trading_meta_v19.json`의 `validation.reliable`을 확인하세요. `false`이거나 검증 메타가 없으면 모델 원시값은 `ML_RAW_SCORE`에만 남고 추천 가중치는 0입니다.

## Railway

`Dockerfile`과 `railway.toml`을 사용합니다. Railway 서비스에는 최소한 `STORAGE_SECRET`을 설정해야 합니다. No-Buy Breaker는 기본적으로 꺼져 있으며, 손실방어 정책상 `ENABLE_NO_BUY_BREAKER`를 설정하지 않는 것을 권장합니다.

자세한 변경 내용은 [docs/LOSS_DEFENSE_V26.md](docs/LOSS_DEFENSE_V26.md)를 참고하세요.
