# -*- coding: utf-8 -*-
"""
pipeline_news.py — Stage 4: 뉴스/DART/LLM 감성분석
═══════════════════════════════════════════════════════
[v20.1] collector.py 분해 — 상위 10종목 뉴스크롤링 + DART + Gemini + 캐시
"""

import os
import json
import time
import asyncio
import numpy as np
import pandas as pd

from pipeline_context import PipelineContext


def enrich_news(ctx: PipelineContext) -> PipelineContext:
    """Stage 4: 뉴스 크롤링 + DART 공시 + LLM 감성분석 → NEWS_SCORE 반영"""
    from collector import (
        log, LLM_AVAILABLE, OUT_DIR, AsyncNewsFetcher,
    )
    from news_engine import analyze_sentiment_llm
    import dart_analyzer

    df_out = ctx.df_out

    LLM_CACHE_TTL_SEC = 6 * 3600
    _llm_cache_path = os.path.join(OUT_DIR, "_llm_cache.json")

    def _load_llm_cache():
        try:
            if os.path.exists(_llm_cache_path):
                with open(_llm_cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                now_ts = time.time()
                return {k: v for k, v in cache.items()
                        if now_ts - v.get("_ts", 0) < LLM_CACHE_TTL_SEC}
        except (json.JSONDecodeError, KeyError, OSError):
            pass
        return {}

    def _save_llm_cache(cache):
        try:
            with open(_llm_cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    if LLM_AVAILABLE:
        log("🧠 상위 10개 종목 심층 분석 중 (뉴스 + DART 공시)...")

        dart_key = os.environ.get("DART_API_KEY")
        dart_eng = dart_analyzer.DartAnalyzer(dart_api_key=dart_key)
        if not dart_key:
            log("⚠️ DART_API_KEY 미설정. 공시 분석 스킵.")

        _score_col = "DISPLAY_SCORE" if "DISPLAY_SCORE" in df_out.columns else "FINAL_SCORE"
        df_out[_score_col] = pd.to_numeric(df_out[_score_col], errors="coerce").fillna(-1)
        target_indices = df_out.nlargest(10, _score_col).index
        target_codes = [str(df_out.loc[i, "종목코드"]).zfill(6) for i in target_indices]

        news_map = {}
        try:
            fetcher = AsyncNewsFetcher(max_concurrent=5)
            if os.name == 'nt':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            news_map = asyncio.run(fetcher.fetch_all(target_codes))
        except Exception as e:
            log(f"⚠️ 뉴스 수집 중 오류: {e}")

        df_out["NEWS_SCORE"] = 0.0
        df_out["NEWS_REASON"] = "특이사항 없음"
        if "AI_COMMENT" not in df_out.columns:
            df_out["AI_COMMENT"] = ""

        llm_cache = _load_llm_cache()
        cache_hits = 0

        for idx in target_indices:
            code = str(df_out.loc[idx, "종목코드"]).zfill(6)
            name = df_out.loc[idx, "종목명"]

            cached = llm_cache.get(code)
            if cached:
                cache_hits += 1
                event_val = cached["event_val"]
                final_reason = cached["reason"]
                df_out.at[idx, "NEWS_SCORE"] = event_val
                old_final = df_out.at[idx, "FINAL_SCORE"]
                df_out.at[idx, "DISPLAY_SCORE"] = np.clip(old_final + event_val, 0, 100)
                df_out.at[idx, "NEWS_REASON"] = final_reason
                if final_reason and final_reason != "특이사항 없음":
                    cur_comment = str(df_out.at[idx, "AI_COMMENT"])
                    df_out.at[idx, "AI_COMMENT"] = (cur_comment if cur_comment != "nan" else "") + f" 📢재료: {final_reason}"
                continue

            # (A) 뉴스 감성 분석
            headlines = news_map.get(code, [])
            l_score, l_reason = analyze_sentiment_llm(name, headlines) if headlines else (0.0, "")

            # (B) DART 공시 분석
            d_score, d_reason = 0.0, ""
            if dart_eng.dart:
                try:
                    disclosures = dart_eng.get_major_disclosures(code, days=3)
                    if disclosures:
                        recent = disclosures[0]
                        d_score, d_reason = dart_eng.analyze_report(recent['rcept_no'], recent['report_nm'])
                        log(f"   📄 {name} DART 분석: {recent['report_nm']} -> {d_score}점")
                except Exception as e:
                    log(f"   ⚠️ {name} DART 분석 오류: {type(e).__name__}: {e}")

            # (C) 점수 통합
            event_val = np.clip(l_score + d_score, -10, 10)
            df_out.at[idx, "NEWS_SCORE"] = event_val
            old_final = df_out.at[idx, "FINAL_SCORE"]
            df_out.at[idx, "DISPLAY_SCORE"] = np.clip(old_final + event_val, 0, 100)

            reasons = [f"[공시]{d_reason}" for d_reason in [d_reason] if d_reason] + \
                      [f"[뉴스]{l_reason}" for l_reason in [l_reason] if l_reason and l_reason != "뉴스없음"]
            final_reason = " / ".join(reasons) if reasons else "특이사항 없음"
            df_out.at[idx, "NEWS_REASON"] = final_reason

            if reasons:
                cur_comment = str(df_out.at[idx, "AI_COMMENT"])
                df_out.at[idx, "AI_COMMENT"] = (cur_comment if cur_comment != "nan" else "") + f" 📢재료: {final_reason}"

            llm_cache[code] = {
                "event_val": float(event_val),
                "reason": final_reason,
                "_ts": time.time(),
            }

        _save_llm_cache(llm_cache)
        if cache_hits > 0:
            log(f"💾 LLM 캐시: {cache_hits}/{len(target_indices)}건 재활용 (TTL={LLM_CACHE_TTL_SEC//3600}h)")
    else:
        log("ℹ️ LLM 설정(API Key)이 없어 심층 분석을 건너뜁니다.")
        df_out["NEWS_SCORE"] = 0.0
        df_out["NEWS_REASON"] = "N/A"
        if "AI_COMMENT" not in df_out.columns:
            df_out["AI_COMMENT"] = ""

    ctx.df_out = df_out
    return ctx
