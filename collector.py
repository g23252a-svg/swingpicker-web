def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    """
    KOSPI/KOSDAQ 전체 티커의 당일 OHLCV를 받아 '거래대금' 기준 상위 N개 추출.
    반환: ['종목코드','거래대금(원)']
    """
    # 시장별 OHLCV (당일 스냅샷)
    kospi = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market="KOSPI")
    kosdaq = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market="KOSDAQ")

    # 인덱스(티커) -> 컬럼 '종목코드'
    def _prep(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=["종목코드", "거래대금"])
        df = df.copy()
        # pykrx 버전별로 인덱스명이 다를 수 있으므로 인덱스를 항상 티커로 가정
        df = df.reset_index()
        # 일반적으로 컬럼명이 '티커'이므로 정규화
        df = df.rename(columns={"티커":"종목코드"}, errors="ignore")
        if "종목코드" not in df.columns:
            # 그래도 없으면 첫 컬럼을 티커로 간주
            df.insert(0, "종목코드", df.iloc[:,0])
        # 거래대금 컬럼 안전 확보(버전에 따라 존재 보장)
        cand = [c for c in df.columns if "거래대금" in str(c)]
        if not cand:
            # 거래대금이 없으면 비워서 반환
            df["거래대금"] = np.nan
        else:
            df["거래대금"] = df[cand[0]].astype("float64")
        return df[["종목코드","거래대금"]]

    k1 = _prep(kospi)
    k2 = _prep(kosdaq)
    all_df = pd.concat([k1, k2], ignore_index=True)

    # 정렬 및 상위 N
    all_df = (
        all_df.dropna(subset=["거래대금"])
              .sort_values("거래대금", ascending=False)
              .head(top_n)
              .reset_index(drop=True)
    )

    # 출력 형식 통일: '거래대금(원)'
    all_df = all_df.rename(columns={"거래대금":"거래대금(원)"})
    # 종목코드는 6자리 zero-fill 보정
    all_df["종목코드"] = all_df["종목코드"].astype(str).str.zfill(6)
    return all_df
