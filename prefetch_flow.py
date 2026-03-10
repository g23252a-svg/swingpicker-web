import os, sys, json, time, logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = os.environ.get("LDY_OUT_DIR", "data")
os.makedirs(OUT_DIR, exist_ok=True)
CACHE_LATEST = os.path.join(OUT_DIR, "flow_cache_latest.json")
CACHE_STALE_HOURS = 20
KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"

def _last_weekday(d):
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")

def _kis_get_token(app_key, app_secret):
    import requests
    try:
        r = requests.post(f"{KIS_BASE_URL}/oauth2/tokenP",
            json={"grant_type":"client_credentials","appkey":app_key,"appsecret":app_secret},
            timeout=10)
        token = r.json().get("access_token")
        if token:
            log.info("KIS 토큰 발급 성공")
            return token
        log.warning(f"KIS 토큰 응답 이상: {r.text[:200]}")
    except Exception as e:
        log.warning(f"KIS 토큰 실패: {e}")
    return None

def _kis_fetch_investor(token, app_key, app_secret, investor_code):
    import requests
    headers = {
        "Authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
        "tr_id": "FHPTJ04400000",
        "content-type": "application/json; charset=utf-8",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "V",
        "FID_COND_SCR_DIV_CODE": "16449",
        "FID_INPUT_ISCD": "0000",
        "FID_DIV_CLS_CODE": "1",
        "FID_RANK_SORT_CLS_CODE": "0",
        "FID_ETC_CLS_CODE": investor_code,
    }
    try:
        r = requests.get(f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/foreign-institution-total",
            headers=headers, params=params, timeout=15)
        log.info(f"KIS HTTP {r.status_code} inv={investor_code} body={r.text[:400]}")
        data = r.json()
        if data.get("rt_cd") != "0":
            log.warning(f"KIS rt_cd={data.get('rt_cd')} msg={data.get('msg1','')}")
            return None
        result = {}
        for row in data.get("output", []):
            code = str(row.get("mksc_shrn_iscd","")).zfill(6)
            result[code] = row
        return result
    except Exception as e:
        log.warning(f"KIS fetch 실패 investor={investor_code}: {e}")
        return None

def fetch_flow(ymd):
    app_key = os.environ.get("KIS_APP_KEY","")
    app_secret = os.environ.get("KIS_APP_SECRET","")
    if not app_key or not app_secret:
        log.warning("KIS_APP_KEY/KIS_APP_SECRET 미설정")
        return {},{},{},"FETCH_FAIL"
    token = _kis_get_token(app_key, app_secret)
    if not token:
        return {},{},{},"FETCH_FAIL"
    log.info(f"KIS 수급 선수집 시작 (기준일: {ymd})")
    raw = _kis_fetch_investor(token, app_key, app_secret, "0") or {}
    frg, inst, ant = {}, {}, {}
    for code, row in raw.items():
        if isinstance(row, dict):
            try: frg[code] = int(str(row.get("frgn_ntby_tr_pbmn","0")).replace(",",""))
            except: pass
            try: inst[code] = int(str(row.get("orgn_ntby_tr_pbmn","0")).replace(",",""))
            except: pass
        else:
            frg[code] = row
    log.info(f"외인: {len(frg)}건  기관: {len(inst)}건  개인: {len(ant)}건")
    major_ok = len(frg)>0 or len(inst)>0
    if major_ok: status="OK"
    elif len(ant)>0: status="PARTIAL"
    else: status="EMPTY"
    return frg, inst, ant, status

def save_cache(frg, inst, ant, ymd, status):
    payload = {"ymd":ymd,"status":status,"fetched_at":datetime.now().isoformat(),
        "source":"KIS","frg":frg,"inst":inst,"ant":ant,
        "counts":{"frg":len(frg),"inst":len(inst),"ant":len(ant)}}
    for path in [os.path.join(OUT_DIR,f"flow_{ymd}.json"), CACHE_LATEST]:
        with open(path,"w",encoding="utf-8") as f:
            json.dump(payload,f,ensure_ascii=False)
    log.info(f"수급 캐시 저장: {CACHE_LATEST} (status={status})")
    return CACHE_LATEST

def load_cache(ymd):
    if not os.path.exists(CACHE_LATEST): return None
    try:
        with open(CACHE_LATEST,encoding="utf-8") as f: c=json.load(f)
    except: return None
    stale=True
    if c.get("fetched_at"):
        try:
            age=(datetime.now()-datetime.fromisoformat(c["fetched_at"])).total_seconds()/3600
            stale=age>CACHE_STALE_HOURS
        except: pass
    cached_ymd = c.get("ymd","")
    if cached_ymd != ymd:
        from datetime import datetime, timedelta
        def _last_wd(d):
            while d.weekday()>=5: d-=timedelta(days=1)
            return d.strftime("%Y%m%d")
        try:
            req_last = _last_wd(datetime.strptime(ymd,"%Y%m%d"))
            stale = cached_ymd != req_last
        except:
            stale = True
    c["stale"]=stale
    return c

if __name__=="__main__":
    ymd = sys.argv[1] if len(sys.argv)>1 else _last_weekday(datetime.now())
    frg,inst,ant,status = fetch_flow(ymd)
    save_cache(frg,inst,ant,ymd,status)
    print(f"\n결과: {status}")
    print(f"외인 {len(frg)}건 / 기관 {len(inst)}건 / 개인 {len(ant)}건")
    sys.exit(1 if status=="FETCH_FAIL" else 0)
