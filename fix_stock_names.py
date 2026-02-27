"""
종목명 복구 스크립트
==================
recommend CSV에서 종목명이 종목코드로 오염된 경우,
Naver Finance API로 실제 종목명을 조회하여 자동 복구합니다.

사용법:
  python fix_stock_names.py                          # data/recommend_latest.csv 자동 패치
  python fix_stock_names.py data/recommend_20260227.csv  # 특정 파일 지정
"""

import pandas as pd
import requests
import time
import sys
import os

def get_stock_name_naver(code: str) -> str:
    """Naver Finance API로 종목명 조회"""
    code = str(code).strip().zfill(6)
    try:
        url = f"https://m.stock.naver.com/api/stock/{code}/basic"
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if r.ok:
            data = r.json()
            name = data.get("stockName", "")
            if name and name != code:
                return name
    except Exception as e:
        print(f"  ⚠️ {code} 조회 실패: {e}")
    return ""


def fix_csv(path: str):
    print(f"📂 파일 로드: {path}")
    df = pd.read_csv(path, dtype={"종목코드": str, "종목명": str})

    if "종목코드" not in df.columns or "종목명" not in df.columns:
        print("❌ 종목코드/종목명 컬럼이 없습니다.")
        return

    # 종목명이 숫자로만 이루어진 행 찾기
    mask = df["종목명"].astype(str).str.match(r"^\d+$")
    bad_count = mask.sum()

    if bad_count == 0:
        print("✅ 종목명 오염 없음 — 패치 불필요")
        return

    print(f"🔍 종목명 오염 감지: {bad_count}/{len(df)}건")
    print(f"🔄 Naver Finance API로 종목명 조회 중...\n")

    fixed = 0
    failed = []
    codes = df.loc[mask, "종목코드"].astype(str).str.zfill(6).unique()

    for i, code in enumerate(codes):
        name = get_stock_name_naver(code)
        if name:
            df.loc[(mask) & (df["종목코드"].astype(str).str.zfill(6) == code), "종목명"] = name
            fixed += 1
            print(f"  ✅ [{i+1}/{len(codes)}] {code} → {name}")
        else:
            failed.append(code)
            print(f"  ❌ [{i+1}/{len(codes)}] {code} → 조회 실패")
        time.sleep(0.1)  # Rate limiting

    print(f"\n{'='*50}")
    print(f"📊 결과: {fixed}/{len(codes)}건 복구 완료")

    if failed:
        print(f"⚠️ 실패: {', '.join(failed)}")

    # 저장
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"💾 저장 완료: {path}")

    # krx_names_latest.csv도 함께 생성
    names_df = df[["종목코드", "종목명"]].drop_duplicates("종목코드")
    names_df = names_df[names_df["종목명"].astype(str) != names_df["종목코드"].astype(str)]
    names_path = os.path.join(os.path.dirname(path), "krx_names_latest.csv")
    names_df.to_csv(names_path, index=False, encoding="utf-8-sig")
    print(f"📋 종목명 매핑 저장: {names_path} ({len(names_df)}건)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # 기본 경로
        for p in ["data/recommend_latest.csv", "recommend_latest.csv"]:
            if os.path.exists(p):
                target = p
                break
        else:
            print("❌ recommend_latest.csv를 찾을 수 없습니다. 경로를 인자로 지정하세요.")
            sys.exit(1)

    fix_csv(target)
