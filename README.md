# SwingPicker

한국 주식 스윙 후보를 수집·평가하고 NiceGUI 웹 화면으로 제공하는 프로젝트입니다. 운영 버전은 `build_manifest.json`을 단일 출처로 사용합니다.

## 현재 버전

- UI: **v25.1.1**
- 추천 엔진: **v3.19.0**
- 검증 엔진: **v3.9.5**
- 백테스트 엔진: **v3.9.2**
- 데이터 스키마: **v5**

`build_manifest.json`과 `version_info.py`의 최신 CHANGELOG 버전이 다르면 CI와 앱 시작이 실패합니다.

## 주요 실행 경로

```text
collector.py
  → pipeline_finalize.py
  → data/recommend_latest.csv
  → main.py / NiceGUI
```

일일 자동 수집은 추천 산출 후 백테스트, 공식매수 검증, no-buy breaker 검증, historical alpha 검증을 모두 재실행합니다. 검증 하나라도 실패하거나 산출물이 이번 실행에서 갱신되지 않으면 신규 데이터 커밋을 중단합니다.

## 로컬 실행

```bash
python -m pip install -r requirements_nicegui.txt
python main.py
```

수집기 실행에는 KRX, DART, Telegram 등 운영 환경변수가 추가로 필요합니다.

## 안전 검사

```bash
python scripts/check_build_manifest.py
python scripts/check_deps.py
python scripts/check_silent_exceptions.py
python check_contract_gate.py
python -m pytest tests/test_safety_reset_v2511.py -q
```

Silent exception baseline을 재생성해 신규 예외를 숨기지 말고, 예외 처리에는 로그, 명시적 fallback 또는 재발생을 넣어야 합니다.

## 데이터 건강도

`DATA_HEALTH_SCORE`는 수집 성공, 결측, fallback 사용 여부를 나타냅니다. 예상 승률이나 수익 신뢰도가 아닙니다. 기존 소비자를 위해 `CONFIDENCE_SCORE`는 같은 값의 호환 alias로 유지됩니다.

## 저장소 운영 원칙

- 자동화는 `data/` 산출물만 커밋합니다.
- 자동화에서 `git add -A`와 충돌 시 `ours` 강제 merge를 사용하지 않습니다.
- 코드 변경은 별도 브랜치와 Pull Request를 통해 반영합니다.
- SHADOW 연구 컬럼은 공식 추천 컬럼을 직접 변경하지 않습니다.

## 면책

본 프로젝트의 출력은 투자 권유가 아니며, 최종 투자 판단과 책임은 사용자에게 있습니다.
