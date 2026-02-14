# 🏆 LDY Pro Trader (v11.0 AI-Elite)
> **AI-Driven Professional Quantitative Investment Terminal**
> *Precision Scoring, LLM Sentiment, Capital Management.*

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg)
![Model](https://img.shields.io/badge/AI_Engine-v15.6_Master-red.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 📈 개요 (Overview)
**LDY Pro Trader**는 단순한 분석기를 넘어, **AI 머신러닝 예측**과 **LLM 뉴스 분석**, 그리고 **정통 퀀트 로직**을 결합한 하이엔드 투자 결정 시스템입니다. 
시장의 잡음을 제거하고, 데이터가 입증한 승리 확률(P-hit)이 높은 '정예군' 종목만을 선별하여 지휘관에게 보고합니다.

## 🚀 핵심 전술 지표 (Key Tactical Intelligence)

### 1. 🧠 v15.6 Master-Grade AI Engine
* **Hybrid Scoring:** 차트 구조(STRUCT), 진입 타이밍(TIMING), AI 예측(ML)의 3원화 통합 점수 산출
* **LLM Intelligence:** Google Gemini Pro를 연동하여 실시간 뉴스 및 DART 공시 호재/악재 판독
* **Macro Risk Filter:** 나스닥 지수 및 USD/KRW 환율 변동에 따른 동적 비중 조절 로직 가동

### 2. 📊 Elite Commander Dashboard
* **Kanban Strategy View:** ATTACK(진격), ARMED(임박), WAIT(매복) 상태별 칸반 보드 시각화
* **7-Factor Radar Chart:** 모멘텀, 수급, 안전마진 등 7가지 핵심 지표의 오각형 능력치 분석
* **Score Waterfall:** 최종 점수가 산출된 근거를 투명하게 분해하여 시각화

### 3. 💰 Capital Management (Betting)
* **Kelly Criterion:** 승률과 손익비를 계산하여 종목별 최적 배팅 비중(Bet-Size) 자동 산출
* **Smart Stop-Loss:** ATR(변동성) 및 주요 지지선(Swing Low) 기반의 지능형 손절가 제시

## 🛠️ 시스템 구성 (Structure)

* **`collector.py`**: 데이터 수집, ML 모델 학습, LLM 분석 및 퀀트 스코어링 엔진 (The Brain)
* **`dashboard.py`**: 실시간 시장 지도 및 종목별 상세 정찰 리포트 UI (The View)
* **`ml_engine.py`**: 시계열 데이터를 학습하여 반등 확률을 예측하는 핵심 AI 엔진 (The Core)
* **`dart_analyzer.py`**: 기업의 공시 보고서를 정밀 분석하여 리스크 감지 (The Scout)



## 🛠️ 실행 방법 (Usage)

```bash
# 1. 환경변수 설정 (.streamlit/secrets.toml)
GEMINI_API_KEY = "your_key"
DART_API_KEY = "your_key"

# 2. 데이터 분석 및 AI 모델 훈련
python collector.py

# 3. 전술 지휘소(대시보드) 기동
streamlit run dashboard.py




⚠️ 면책 조항 (Disclaimer)
본 소프트웨어는 투자를 보조하는 데이터 분석 도구입니다. 모든 투자 결정의 최종 책임은 지휘관 본인에게 있으며, 과거의 수익률이 미래의 결과를 보장하지 않습니다.
