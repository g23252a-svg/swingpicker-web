



# pages/900_debug_market_data.py
# -*- coding: utf-8 -*-

import streamlit as st
from auth_user import render_auth_box

st.set_page_config(page_title="Debug Market Data", layout="wide")

with st.sidebar:
    user = render_auth_box()

if not user or user.get("role") != "admin":
    st.error("🚫 이 페이지는 관리자 전용입니다.")
    st.stop()

st.title("📡 KOSPI / KOSDAQ 데이터 디버그 페이지")

# 1) 라이브러리 import 상태 체크
try:
    import FinanceDataReader as fdr
    FDR_OK = True
    FDR_ERR = None
except Exception as e:
    FDR_OK = False
    FDR_ERR = e

try:
    from pykrx import stock
    PYKRX_OK = True
    PYKRX_ERR = None
except Exception as e:
    PYKRX_OK = False
    PYKRX_ERR = e

st.title("📡 KOSPI / KOSDAQ 데이터 디버그 페이지")

st.write("이 페이지는 KS11 / KQ11 지수 데이터를 FDR, pykrx로 직접 불러와서\n"
         "어디서 막히는지 확인하기 위한 테스트 페이지입니다.")

st.divider()

# 2) 라이브러리 로딩 결과 표시
st.subheader("1. 라이브러리 로딩 상태")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### FinanceDataReader (fdr)")
    st.write("FDR_OK =", FDR_OK)
    if FDR_ERR:
        st.error(f"Import Error: {FDR_ERR}")

with col2:
    st.markdown("#### pykrx")
    st.write("PYKRX_OK =", PYKRX_OK)
    if PYKRX_ERR:
        st.error(f"Import Error: {PYKRX_ERR}")

st.divider()

# 3) FDR로 KS11 / KQ11 불러보기
st.subheader("2. FDR로 KS11 / KQ11 테스트")

if not FDR_OK:
    st.warning("FinanceDataReader import 자체가 실패했습니다. (위 에러 참고)")
else:
    try:
        df_ks11 = fdr.DataReader("KS11")  # 코스피 지수
        st.markdown("**KS11 (KOSPI Index) tail(5)**")
        st.dataframe(df_ks11.tail(), use_container_width=True)

        st.write("행 개수:", len(df_ks11))
        st.write("컬럼:", list(df_ks11.columns))
    except Exception as e:
        st.error(f"KS11 DataReader 실패: {e}")

    st.markdown("---")

    try:
        df_kq11 = fdr.DataReader("KQ11")  # 코스닥 지수
        st.markdown("**KQ11 (KOSDAQ Index) tail(5)**")
        st.dataframe(df_kq11.tail(), use_container_width=True)

        st.write("행 개수:", len(df_kq11))
        st.write("컬럼:", list(df_kq11.columns))
    except Exception as e:
        st.error(f"KQ11 DataReader 실패: {e}")

st.divider()

# 4) pykrx로 1001(코스피) / 2001(코스닥) 불러보기
st.subheader("3. pykrx로 1001 / 2001 지수 테스트")

if not PYKRX_OK:
    st.warning("pykrx import 자체가 실패했습니다. (위 에러 참고)")
else:
    today = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    st.caption(f"조회 기간: {start} ~ {today}")

    # 코스피 지수 1001
    try:
        df_1001 = stock.get_index_ohlcv_by_date(start, today, "1001")
        st.markdown("**1001 (KOSPI Index) tail(5)**")
        st.dataframe(df_1001.tail(), use_container_width=True)

        st.write("행 개수:", len(df_1001))
        st.write("컬럼:", list(df_1001.columns))
    except Exception as e:
        st.error(f"pykrx 1001 조회 실패: {e}")

    st.markdown("---")

    # 코스닥 지수 2001
    try:
        df_2001 = stock.get_index_ohlcv_by_date(start, today, "2001")
        st.markdown("**2001 (KOSDAQ Index) tail(5)**")
        st.dataframe(df_2001.tail(), use_container_width=True)

        st.write("행 개수:", len(df_2001))
        st.write("컬럼:", list(df_2001.columns))
    except Exception as e:
        st.error(f"pykrx 2001 조회 실패: {e}")
