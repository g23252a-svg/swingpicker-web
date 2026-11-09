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

def batch_target(now=None):
    """[v86] scripts/batch_target.sh와 같은 창 규칙으로 수급 파일 날짜(거래일)를 정한다.

    예전엔 _last_weekday(now)(시계 날짜)를 써서, 11/02 prefetch가 11/03 01:27에 돌며
    flow_20261103.json을 썼고 다음 날 11/03 prefetch가 그것을 덮어써 flow_20261102.json이
    영영 생기지 않았다(11/02 배치는 다른 날짜의 latest 캐시로 조용히 폴백).
      16:00~24:00 평일  → ("evening", 오늘)
      00:00~08:00       → ("overnight", 직전 평일)
      15:30~16:00 평일  → ("postclose", 오늘)  ← cron 35 6(15:35 KST)이 제때 돌 때. 장 마감 뒤라
                           그날 수급이다. batch_target.sh는 이 구간을 market으로 보지만
                           거기선 auto_collect 가드용이라 수급 선수집과 목적이 다르다.
      그 밖(장중·주말 저녁) → (window, None)  — 수집하지 않는다.
    반환: (window, ymd 또는 None)
    """
    now = now or datetime.now()  # 워크플로 TZ=Asia/Seoul
    h, dow = now.hour, now.weekday()
    if h >= 16:
        if dow >= 5:
            return "weekend", None
        return "evening", now.strftime("%Y%m%d")
    if h < 8:
        d = now - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return "overnight", d.strftime("%Y%m%d")
    if dow < 5 and (h, now.minute) >= (15, 30):
        return "postclose", now.strftime("%Y%m%d")
    return ("weekend" if dow >= 5 else "market"), None


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

def fetch_flow(ymd, token=None):
    app_key = os.environ.get("KIS_APP_KEY","")
    app_secret = os.environ.get("KIS_APP_SECRET","")
    if not app_key or not app_secret:
        log.warning("KIS_APP_KEY/KIS_APP_SECRET 미설정")
        return {},{},{},"FETCH_FAIL"
    # [v79.2] 토큰은 분당 1회 발급 제한 — 호출자가 넘기면 재발급하지 않는다.
    token = token or _kis_get_token(app_key, app_secret)
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

def _collect_full_universe(token):
    """[v79.2] 전종목 투자자별 순매수(KIS 종목별 API, 30일치) — 실패해도 본 흐름 무영향."""
    try:
        import requests
        from services import investor_flow_kis as _K
        app_key = os.environ.get("KIS_APP_KEY",""); app_secret = os.environ.get("KIS_APP_SECRET","")
        codes = _K.universe_codes(OUT_DIR)
        if not codes:
            log.warning("[v79.2] 유니버스 코드 없음 — OHLCV 캐시 부재"); return
        with requests.Session() as sess:
            s = _K.collect_universe(sess, token, app_key, app_secret, codes, OUT_DIR)
        log.info("[v79.2] " + _K.line(s))
    except Exception as e:
        log.warning(f"[v79.2] 전종목 수급 수집 스킵: {e}")


if __name__=="__main__":
    # [v86] 인자 > BATCH_TARGET 환경변수 > 창 규칙(batch_target). 시계 날짜는 쓰지 않는다.
    ymd = (sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else "") \
        or os.environ.get("BATCH_TARGET", "").strip()
    if not ymd:
        _win, ymd = batch_target()
        if not ymd:
            log.info(f"[v86] {_win} 창 — 대상 세션 없음, 수급 선수집 생략")
            sys.exit(0)
        log.info(f"[v86] 창 {_win} → 수급 기준일 {ymd}")
    _tok = None
    if os.environ.get("KIS_APP_KEY") and os.environ.get("KIS_APP_SECRET"):
        _tok = _kis_get_token(os.environ["KIS_APP_KEY"], os.environ["KIS_APP_SECRET"])
    frg,inst,ant,status = fetch_flow(ymd, token=_tok)
    save_cache(frg,inst,ant,ymd,status)
    print(f"\n결과: {status}")
    print(f"외인 {len(frg)}건 / 기관 {len(inst)}건 / 개인 {len(ant)}건")
    if _tok:
        _collect_full_universe(_tok)
    sys.exit(1 if status=="FETCH_FAIL" else 0)
