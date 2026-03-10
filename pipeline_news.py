# -*- coding: utf-8 -*-
"""pipeline_news.py — Stage 4: 뉴스/DART/LLM 감성분석 [v20.2]"""
import os, json, time, asyncio, numpy as np, pandas as pd
from pipeline_context import PipelineContext
from shared_log import log, LLM_AVAILABLE, OUT_DIR

def enrich_news(ctx: PipelineContext) -> PipelineContext:
    from async_crawler import AsyncNewsFetcher
    from news_engine import analyze_sentiment_llm
    import dart_analyzer
    df_out = ctx.df_out
    LLM_CACHE_TTL_SEC = 6 * 3600
    _llm_cache_path = os.path.join(OUT_DIR, "_llm_cache.json")
    def _load_cache():
        try:
            if os.path.exists(_llm_cache_path):
                with open(_llm_cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                now_ts = time.time()
                return {k: v for k, v in cache.items() if now_ts - v.get("_ts", 0) < LLM_CACHE_TTL_SEC}
        except (json.JSONDecodeError, KeyError, OSError): pass
        return {}
    def _save_cache(cache):
        try:
            with open(_llm_cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=1)
        except OSError: pass
    if LLM_AVAILABLE:
        log("🧠 상위 10개 종목 심층 분석 중 (뉴스 + DART 공시)...")
        dart_key = os.environ.get("DART_API_KEY")
        dart_eng = dart_analyzer.DartAnalyzer(dart_api_key=dart_key)
        if not dart_key: log("⚠️ DART_API_KEY 미설정. 공시 분석 스킵.")
        _sc = "DISPLAY_SCORE" if "DISPLAY_SCORE" in df_out.columns else "FINAL_SCORE"
        df_out[_sc] = pd.to_numeric(df_out[_sc], errors="coerce").fillna(-1)
        tidx = df_out.nlargest(10, _sc).index
        tcodes = [str(df_out.loc[i, "종목코드"]).zfill(6) for i in tidx]
        news_map = {}
        try:
            fetcher = AsyncNewsFetcher(max_concurrent=5)
            if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            news_map = asyncio.run(fetcher.fetch_all(tcodes))
        except Exception as e: log(f"⚠️ 뉴스 수집 중 오류: {e}")
        df_out["NEWS_SCORE"] = 0.0; df_out["NEWS_REASON"] = "특이사항 없음"
        if "AI_COMMENT" not in df_out.columns: df_out["AI_COMMENT"] = ""
        llm_cache = _load_cache(); cache_hits = 0
        for idx in tidx:
            code = str(df_out.loc[idx, "종목코드"]).zfill(6); name = df_out.loc[idx, "종목명"]
            cached = llm_cache.get(code)
            if cached:
                cache_hits += 1; ev = cached["event_val"]; fr = cached["reason"]
                df_out.at[idx, "NEWS_SCORE"] = ev
                df_out.at[idx, "DISPLAY_SCORE"] = np.clip(df_out.at[idx, "FINAL_SCORE"] + ev, 0, 100)
                df_out.at[idx, "NEWS_REASON"] = fr
                if fr and fr != "특이사항 없음":
                    cc = str(df_out.at[idx, "AI_COMMENT"])
                    df_out.at[idx, "AI_COMMENT"] = (cc if cc != "nan" else "") + f" 📢재료: {fr}"
                continue
            headlines = news_map.get(code, [])
            l_sc, l_re = analyze_sentiment_llm(name, headlines) if headlines else (0.0, "")
            d_sc, d_re = 0.0, ""
            if dart_eng.dart:
                try:
                    disc = dart_eng.get_major_disclosures(code, days=3)
                    if disc:
                        rec = disc[0]; d_sc, d_re = dart_eng.analyze_report(rec['rcept_no'], rec['report_nm'])
                        log(f"   📄 {name} DART 분석: {rec['report_nm']} -> {d_sc}점")
                except Exception as e: log(f"   ⚠️ {name} DART 분석 오류: {type(e).__name__}: {e}")
            ev = np.clip(l_sc + d_sc, -10, 10)
            df_out.at[idx, "NEWS_SCORE"] = ev
            df_out.at[idx, "DISPLAY_SCORE"] = np.clip(df_out.at[idx, "FINAL_SCORE"] + ev, 0, 100)
            reasons = [f"[공시]{d_re}" for d_re in [d_re] if d_re] + \
                      [f"[뉴스]{l_re}" for l_re in [l_re] if l_re and l_re != "뉴스없음"]
            fr = " / ".join(reasons) if reasons else "특이사항 없음"
            df_out.at[idx, "NEWS_REASON"] = fr
            if reasons:
                cc = str(df_out.at[idx, "AI_COMMENT"])
                df_out.at[idx, "AI_COMMENT"] = (cc if cc != "nan" else "") + f" 📢재료: {fr}"
            llm_cache[code] = {"event_val": float(ev), "reason": fr, "_ts": time.time()}
        _save_cache(llm_cache)
        if cache_hits > 0: log(f"💾 LLM 캐시: {cache_hits}/{len(tidx)}건 재활용 (TTL={LLM_CACHE_TTL_SEC//3600}h)")
    else:
        log("ℹ️ LLM 설정(API Key)이 없어 심층 분석을 건너뜁니다.")
        df_out["NEWS_SCORE"] = 0.0; df_out["NEWS_REASON"] = "N/A"
        if "AI_COMMENT" not in df_out.columns: df_out["AI_COMMENT"] = ""
    ctx.df_out = df_out
    return ctx
