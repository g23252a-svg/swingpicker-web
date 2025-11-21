# 🏆 LDY Pro Trader (v4.6)
> **AI Powered Quantitative Swing Trading System** > *Serverless, Automated, Data-Driven.*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.34+-FF4B4B.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 📈 개요 (Overview)
**LDY Pro Trader**는 직장인을 위한 자동화된 스윙 트레이딩 분석 시스템입니다.  
단순한 가격 등락이 아닌, **기관 수급(MFI), 장기 추세(MA120), 에너지 응축(Squeeze)** 등 전문 퀀트 로직을 통해 승률 높은 종목을 선별합니다.

## 🚀 핵심 기능 (Key Features)

### 1. 🧠 Intelligent Collector (Engine)
* **MFI (Money Flow Index):** 단순 거래량이 아닌 '자금의 질'을 분석하여 매집 종목 포착
* **MA120 Trend Filter:** 6개월 장기 추세 위에 있는 안전한 종목만 선별
* **Volatility Squeeze:** 볼린저 밴드 수축을 감지하여 폭발 직전 타이밍 포착
* **Github Actions:** 매일 장 마감 후 서버 없이 자동으로 데이터 수집 및 갱신

### 2. 📊 Interactive Dashboard (Viewer)
* **Market Radar:** KOSPI/KOSDAQ 지수 국면(Bull/Bear) 실시간 신호등 표시
* **Instant Chart:** HTS 없이 웹에서 즉시 60일 캔들 차트 및 진입가/손절가 확인
* **Strategy Tags:** 돌파(BRK), 눌림(PULL) 등 전략별 자동 태깅

## 🛠️ 실행 방법 (Usage)

### 🌐 웹에서 바로 보기 (Live Demo)
복잡한 설치 과정 없이, 아래 버튼을 눌러 실시간 분석 결과를 확인하세요.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ldy-quant.streamlit.app)

### 💻 로컬 실행 (Local Installation)
직접 코드를 수정하거나 로컬 환경에서 실행하려면 아래 순서를 따르세요.

```bash
# 1. 레포지토리 클론
git clone https://github.com/g23252a-svg/swingpicker-web.git
cd swingpicker-web

# 2. 필수 패키지 설치
pip install -r requirements.txt

# 3. 수집기 실행 (데이터 생성)
# (약 5~10분 소요, 완료 후 data 폴더에 csv 생성됨)
python collector.py

# 4. 대시보드 실행
streamlit run dashboard.py
