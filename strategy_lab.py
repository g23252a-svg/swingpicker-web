# strategy_lab.py (v1.0) - LDY 퀀트 전략 시뮬레이터
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# 페이지 설정
st.set_page_config(page_title="🧪 LDY 전략 실험실", layout="wide", page_icon="🧪")

# -------------------------- 데이터 로딩 함수 --------------------------
@st.cache_data
def load_all_data():
    """
    1. data/recommend_*.csv 파일을 모두 읽어서 하나의 DataFrame으로 합칩니다.
    2. data/ohlcv_cache_*.pkl 중 가장 최신 파일을 읽어서 가격 데이터로 사용합니다.
    """
    # 1. 추천 이력 로드
    rec_files = sorted(glob.glob(os.path.join("data", "recommend_*.csv")))
    dfs = []
    for f in rec_files:
        try:
            # 파일명에서 날짜 추출 (recommend_20240101.csv)
            base = os.path.basename(f)
            date_str = base.split("_")[1].split(".")[0]
            if len(date_str) >= 8 and date_str.isdigit():
                df = pd.read_csv(f, dtype={'종목코드': str})
                df['추천일'] = pd.to_datetime(date_str[:8], format="%Y%m%d")
                
                # 필수 컬럼 확인 (없으면 생성)
                for col in ['RANK_SCORE', 'TOTAL_SCORE', 'RSI14', 'MFI14', '추천매수가', '종가']:
                    if col not in df.columns:
                        df[col] = 0
                
                dfs.append(df)
        except Exception as e:
            continue
            
    if not dfs:
        return pd.DataFrame(), {}
    
    full_recs = pd.concat(dfs, ignore_index=True)
    full_recs['종목코드'] = full_recs['종목코드'].str.zfill(6)
    
    # 2. 최신 가격 데이터 로드 (OHLCV 캐시)
    cache_files = sorted(glob.glob(os.path.join("data", "ohlcv_cache_*.pkl")), reverse=True)
    price_map = {}
    
    if cache_files:
        latest_cache = cache_files[0]
        # print(f"Loading price data from {latest_cache}...")
        try:
            with open(latest_cache, 'rb') as f:
                price_map = pickle.load(f)
        except Exception:
            st.error("가격 데이터 로드 실패")
            
    return full_recs, price_map

# -------------------------- 시뮬레이션 함수 --------------------------
def run_simulation(df, price_map, hold_days, target_pct, stop_pct):
    """
    필터링된 추천 종목들에 대해 N일 보유 시뮬레이션을 수행합니다.
    """
    results = []
    
    # 진행 상황바
    progress_bar = st.progress(0)
    total = len(df)
    
    for i, row in df.iterrows():
        if i % 100 == 0:
            progress_bar.progress(min(i / total, 1.0))
            
        code = row['종목코드']
        entry_date = row['추천일']
        entry_price = float(row['추천매수가']) if row['추천매수가'] > 0 else float(row['종가'])
        
        if entry_price <= 0: continue
        if code not in price_map: continue
        
        # 해당 종목의 전체 차트 데이터
        ohlcv = price_map[code]
        
        # 추천일 이후의 데이터만 필터링
        # (추천일 다음날 시가 진입 가정, 혹은 추천일 종가 진입 가정. 여기선 추천일 종가 진입으로 단순화)
        future_data = ohlcv[ohlcv.index > entry_date].head(hold_days)
        
        if future_data.empty:
            continue
            
        # 시뮬레이션 로직
        exit_price = future_data.iloc[-1]['종가'] # 기본: 만기 청산
        exit_date = future_data.index[-1]
        status = "HOLD"
        
        # 고가/저가 확인하여 익절/손절 체크
        max_h = future_data['고가'].max()
        min_l = future_data['저가'].min()
        
        target_price = entry_price * (1 + target_pct/100)
        stop_price = entry_price * (1 - stop_pct/100)
        
        # 1. 손절 체크 (보수적 관점: 저가가 손절가 터치하면 즉시 청산)
        if min_l <= stop_price:
            exit_price = stop_price
            status = "STOP"
        # 2. 익절 체크 (손절 안 당하고 고가가 목표가 터치)
        elif max_h >= target_price:
            exit_price = target_price
            status = "WIN"
            
        # 수익률 계산 (수수료 0.25% 반영)
        ret = (exit_price - entry_price) / entry_price * 100 - 0.25
        
        results.append({
            '추천일': entry_date,
            '종목명': row['종목명'],
            '점수': row.get('TOTAL_SCORE', row.get('RANK_SCORE', 0)),
            '진입가': entry_price,
            '청산가': int(exit_price),
            '수익률': ret,
            '상태': status
        })
        
    progress_bar.empty()
    return pd.DataFrame(results)

# -------------------------- UI 구성 --------------------------
st.title("🧪 LDY 퀀트 전략 실험실 (Backtest Lab)")
st.markdown("과거 추천 종목에 대해 **파라미터(점수, RSI 등)**를 조정하여 가상 매매 수익률을 검증해보세요.")

# 1. 데이터 로드
all_recs, price_map = load_all_data()

if all_recs.empty:
    st.warning("⚠️ 데이터가 없습니다. `collector.py`를 먼저 실행하여 추천 데이터를 생성하세요.")
    st.stop()

st.sidebar.header("🛠️ 전략 설정")

# 2. 필터링 조건 입력
min_score = st.sidebar.slider("최소 점수 (TOTAL_SCORE)", 0, 100, 80, step=5)
min_rsi = st.sidebar.slider("최소 RSI", 0, 100, 30)
max_rsi = st.sidebar.slider("최대 RSI", 0, 100, 70)
min_mfi = st.sidebar.slider("최소 MFI (수급)", 0, 100, 50)

st.sidebar.markdown("---")
st.sidebar.header("💰 매매 규칙")
hold_days = st.sidebar.number_input("최대 보유 기간 (일)", 1, 60, 10)
target_pct = st.sidebar.number_input("목표 수익률 (%)", 1.0, 50.0, 10.0)
stop_pct = st.sidebar.number_input("손절 제한 (%)", 1.0, 30.0, 5.0)

# 3. 데이터 필터링
filtered_df = all_recs[
    (all_recs['TOTAL_SCORE'] >= min_score) &
    (all_recs['RSI14'] >= min_rsi) &
    (all_recs['RSI14'] <= max_rsi) &
    (all_recs['MFI14'] >= min_mfi)
].copy()

st.metric("검증 대상 종목 수", f"{len(filtered_df)}개", f"전체 {len(all_recs)}개 중")

# 4. 시뮬레이션 실행 버튼
if st.button("🚀 시뮬레이션 시작", type="primary"):
    if filtered_df.empty:
        st.error("조건에 맞는 종목이 없습니다.")
    else:
        with st.spinner("과거 데이터로 매매 시뮬레이션 중..."):
            res_df = run_simulation(filtered_df, price_map, hold_days, target_pct, stop_pct)
            
        if not res_df.empty:
            # 결과 집계
            win_rate = len(res_df[res_df['수익률'] > 0]) / len(res_df) * 100
            avg_ret = res_df['수익률'].mean()
            total_ret = res_df['수익률'].sum() # 단리 합산 (참고용)
            
            # --- 결과 대시보드 ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("승률 (Win Rate)", f"{win_rate:.1f}%")
            c2.metric("평균 수익률", f"{avg_ret:.2f}%")
            c3.metric("총 거래 횟수", f"{len(res_df)}회")
            c4.metric("누적 수익 (단리)", f"{total_ret:.1f}%")
            
            # 누적 수익 곡선 (Equity Curve)
            res_df = res_df.sort_values('추천일')
            res_df['누적수익률'] = res_df['수익률'].cumsum()
            
            fig = px.line(res_df, x='추천일', y='누적수익률', title='📈 누적 수익률 곡선 (Equity Curve)')
            st.plotly_chart(fig, use_container_width=True)
            
            # 상세 거래 내역
            st.subheader("📝 상세 거래 내역")
            st.dataframe(res_df.style.format({'수익률': '{:.2f}%', '진입가': '{:,.0f}', '청산가': '{:,.0f}'}))
            
        else:
            st.warning("시뮬레이션 결과가 없습니다. (가격 데이터 부족 등)")
