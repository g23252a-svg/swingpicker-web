# -*- coding: utf-8 -*-
"""naver_aftermarket.py - Naver API after-market price updater"""
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

def update_csv_with_aftermarket(csv_path, snap_path=None):
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    except Exception as e:
        logger.warning(f"CSV read fail: {e}")
        return 0
    codes = df.iloc[:, 1].astype(str).str.zfill(6).tolist()
    code_col = df.columns[1]
    close_col = df.columns[5]
    total = len(codes)
    updated = 0
    price_map = {}
    logger.info(f"After-market update start ({total} stocks)")
    for i, code in enumerate(codes):
        result = fetch_after_market_price(code)
        if not result: continue
        old = float(df.loc[df[code_col].str.zfill(6)==code, close_col].iloc[0])
        new = result["final"]
        if new > 0 and new != old:
            df.loc[df[code_col].str.zfill(6)==code, close_col] = str(new)
            price_map[code] = new
            updated += 1
            diff = (new/old-1)*100
            logger.info(f"  After: {code} {int(old)}->{new} ({diff:+.1f}%)")
        if (i+1) % 20 == 0: time.sleep(0.5)
    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"After-market done: {updated} updated -> {csv_path}")
        if snap_path and price_map:
            try:
                snap = pd.read_csv(snap_path, dtype=str, encoding="utf-8-sig")
                # [v20.6.5] 종목코드 컬럼 명시 탐색 (columns[1] 하드코딩 제거)
                snap_code = None
                for c in snap.columns:
                    if '종목코드' in c or c == 'code':
                        snap_code = c
                        break
                if not snap_code:
                    snap_code = snap.columns[0]
                snap_close = None
                for c in snap.columns:
                    if c.encode('utf-8',errors='ignore') in [b'\xec\xa2\x85\xea\xb0\x80']:
                        snap_close = c
                        break
                if not snap_close:
                    for c in snap.columns:
                        if '종가' in c:
                            snap_close = c
                            break
                if snap_close:
                    cnt = 0
                    for c, p in price_map.items():
                        m = snap[snap_code].astype(str).str.zfill(6)==c
                        if m.any(): snap.loc[m, snap_close] = str(p); cnt += 1
                    snap.to_csv(snap_path, index=False, encoding="utf-8-sig")
                    logger.info(f"After-market snapshot: {cnt} -> {snap_path}")
            except Exception as e:
                logger.warning(f"Snapshot update fail: {e}")
    else:
        logger.info("No after-market changes")
    return updated
