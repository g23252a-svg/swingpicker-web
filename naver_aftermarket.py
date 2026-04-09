# -*- coding: utf-8 -*-
"""naver_aftermarket.py - Naver API after-market price updater [v20.6.4]

[v20.6.4] sidecar 방식 추가 — recommend CSV 원본 오염 방지
 - fetch_after_market_prices_sidecar(): 별도 CSV에 시간외 가격 저장
 - update_csv_with_aftermarket(): 레거시 호환 (str 크래시 수정)
"""
import logging, time, pandas as pd, requests

logger = logging.getLogger(__name__)

def fetch_after_market_price(code):
    url = f"https://m.stock.naver.com/api/stock/{code}/basic"
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200: return {}
        d = r.json()
        close = int(d.get("closePrice", "0").replace(",", ""))
        result = {"close": close, "after": 0, "final": close}
        over = d.get("overMarketPriceInfo")
        if over and over.get("overPrice"):
            after = int(over["overPrice"].replace(",", ""))
            if after > 0:
                result["after"] = after
                result["final"] = after
        return result
    except Exception:
        return {}


def fetch_after_market_prices_sidecar(csv_path, sidecar_path, snap_path=None):
    """recommend CSV 원본 불변. 시간외 가격은 sidecar CSV에 저장."""
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    except Exception as e:
        logger.warning(f"CSV read fail: {e}")
        return 0

    # [v20.3.2] 이름 기반 컬럼 탐색 — 위치 의존 제거
    code_col = next((c for c in df.columns if '종목코드' in c or c == 'code'), df.columns[1])
    close_col = next((c for c in df.columns if c == '종가' or 'close' in c.lower()), df.columns[5])
    codes = df[code_col].astype(str).str.zfill(6).tolist()
    total = len(codes)
    logger.info(f"After-market update start ({total} stocks)")

    sidecar_rows = []
    updated = 0
    snap_updates = {}

    for i, code in enumerate(codes):
        result = fetch_after_market_price(code)
        if not result: continue
        try:
            old = float(df.loc[df[code_col].str.zfill(6)==code, close_col].iloc[0])
        except (IndexError, ValueError): continue
        new = result["final"]
        if new > 0 and new != old:
            sidecar_rows.append({
                "종목코드": code, "분석종가": int(old),
                "시간외종가": new, "변동률": round((new/old - 1) * 100, 2),
            })
            snap_updates[code] = new
            updated += 1
        if (i+1) % 20 == 0: time.sleep(0.5)

    if sidecar_rows:
        sidecar_df = pd.DataFrame(sidecar_rows)
        sidecar_df.to_csv(sidecar_path, index=False, encoding="utf-8-sig")
        logger.info(f"After-market sidecar: {updated} stocks -> {sidecar_path}")
        if snap_path and snap_updates:
            try:
                snap = pd.read_csv(snap_path, dtype=str, encoding="utf-8-sig")
                snap_code = next((c for c in snap.columns if '종목코드' in c or c=='code'), snap.columns[0])
                snap_close = next((c for c in snap.columns if '종가' in c), None)
                if snap_close:
                    cnt = 0
                    for c, p in snap_updates.items():
                        m = snap[snap_code].astype(str).str.zfill(6)==c
                        if m.any(): snap.loc[m, snap_close] = str(p); cnt += 1
                    snap.to_csv(snap_path, index=False, encoding="utf-8-sig")
                    logger.info(f"After-market snapshot: {cnt} -> {snap_path}")
            except Exception as e: logger.warning(f"Snapshot update fail: {e}")
    else:
        logger.info("No after-market changes")
    return updated


def update_csv_with_aftermarket(csv_path, snap_path=None):
    """레거시 호환 — str 크래시 수정."""
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    except Exception as e:
        logger.warning(f"CSV read fail: {e}"); return 0
    # [v20.3.2] 이름 기반 컬럼 탐색
    code_col = next((c for c in df.columns if '종목코드' in c or c == 'code'), df.columns[1])
    close_col = next((c for c in df.columns if c == '종가' or 'close' in c.lower()), df.columns[5])
    codes = df[code_col].astype(str).str.zfill(6).tolist()
    total = len(codes); updated = 0; price_map = {}
    logger.info(f"After-market update start ({total} stocks)")
    for i, code in enumerate(codes):
        result = fetch_after_market_price(code)
        if not result: continue
        try:
            old = float(df.loc[df[code_col].str.zfill(6)==code, close_col].iloc[0])
        except (IndexError, ValueError): continue
        new = result["final"]
        if new > 0 and new != old:
            df.loc[df[code_col].str.zfill(6)==code, close_col] = str(new)
            price_map[code] = new; updated += 1
        if (i+1) % 20 == 0: time.sleep(0.5)
    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        if snap_path and price_map:
            try:
                snap = pd.read_csv(snap_path, dtype=str, encoding="utf-8-sig")
                snap_code = next((c for c in snap.columns if '종목코드' in c or c=='code'), snap.columns[0])
                snap_close = next((c for c in snap.columns if '종가' in c), None)
                if snap_close:
                    cnt = 0
                    for c, p in price_map.items():
                        m = snap[snap_code].astype(str).str.zfill(6)==c
                        if m.any(): snap.loc[m, snap_close] = str(p); cnt += 1
                    snap.to_csv(snap_path, index=False, encoding="utf-8-sig")
            except Exception as e: logger.warning(f"Snapshot update fail: {e}")
    else:
        logger.info("No after-market changes")
    return updated
