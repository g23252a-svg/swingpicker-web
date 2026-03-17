# -*- coding: utf-8 -*-
"""pipeline_score.py — Stage 3: 스코어링 + 라우팅 + 전략 팩토리 [v20.6.3]
═══════════════════════════════════════════════════════════════════════
[v20.6.3] SSOT Complete + Deterministic Engine
  - Trigger Score: ThreadPoolExecutor 병렬 배치 (종목별 OHLCV 병렬 처리)
  - ROUTE: _vec_determine_state_dynamic 벡터 판정 (apply 제거)
  - 예외 처리 등급화: [AXIS:*] 태그 warning 로깅 → run_health에서 후속 평가
  - 점수 설명: generate_score_reasons(macro_risk=) 장세 연동 임계치
"""
import os, gc, logging, numpy as np, pandas as pd
from pipeline_context import PipelineContext
from shared_log import log, OUT_DIR, safe_quantile
from collector_config import Route
from scoring_engine import (
    build_global_score,
    _vec_determine_state_dynamic,
    generate_score_reasons,
)
from macro_filter import compute_market_breadth, label_market_temp

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  [v20.6.3] Trigger Score 병렬 배치 처리
# ═══════════════════════════════════════════════════

def _batch_trigger_scores(df: pd.DataFrame, ohlcv_map: dict,
                           max_workers: int = 4) -> list:
    """
    [v20.6.3] ThreadPoolExecutor 병렬 배치 처리.
    종목별 OHLCV 길이가 다르므로 순수 벡터화는 불가 → 병렬 I/O로 해결.
    100종목 기준 sequential 대비 ~3x 속도 향상 (GIL-free pandas rolling).
    개별 종목 실패 시 warning 로깅 (silent fail 방지).

    [v20.6.3] collector import를 _calc_one 내부로 지연(lazy) 처리.
    → 빈 ohlcv_map 테스트 시 collector 의존성 불필요.
    → 테스트 용이성 + 모듈 경계 개선.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    codes = df['종목코드'].astype(str).str.zfill(6).tolist()

    def _calc_one(code):
        ohlcv_df = ohlcv_map.get(code)
        if ohlcv_df is None or ohlcv_df.empty:
            return code, 0.0, None
        try:
            # [v20.6.3] lazy import: collector는 실제 계산이 필요할 때만 로드
            from trigger_engine import calculate_trigger_score  # [v20.6.5] 직접 import
            return code, float(calculate_trigger_score(ohlcv_df)), None
        except Exception as e:
            return code, 0.0, str(e)

    # 순서 보존을 위해 dict로 수집
    score_map = {}
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_calc_one, c): c for c in codes}
        for fut in as_completed(futures):
            code, score, err = fut.result()
            score_map[code] = score
            if err is not None:
                fail_count += 1
                if fail_count <= 3:
                    logger.warning(f"⚠️ [AXIS:TRIGGER] {code} 트리거 계산 실패: {err}")

    if fail_count > 0:
        logger.warning(
            f"⚠️ [AXIS:TRIGGER] 총 {fail_count}/{len(codes)}건 실패 "
            f"→ 0점 처리됨 (투자 판단 품질 저하 가능)"
        )

    # 원본 DataFrame 순서 유지
    return [score_map.get(c, 0.0) for c in codes]


def run_scoring(ctx: PipelineContext) -> PipelineContext:
    from collector import (classify_big_sector,
        add_sector_momentum, ml_engine)
    from investor_flow import fetch_investor_net_buying  # [v20.6.5] 직접 import
    from trigger_engine import calculate_trigger_score    # [v20.6.5] 직접 import
    df_raw = ctx.df_out

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 수급 보정
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if ctx.inv_maps:
        map_ant = ctx.inv_maps.get("ant", {})
    else:
        try:
            _map_frg, _map_inst, map_ant = fetch_investor_net_buying(ctx.trade_ymd)
            df_raw["외인순매수"] = df_raw["종목코드"].map(_map_frg).fillna(0)
            df_raw["기관순매수"] = df_raw["종목코드"].map(_map_inst).fillna(0)
            df_raw["메이저순매수"] = df_raw["외인순매수"] + df_raw["기관순매수"]
        except Exception as e:
            logger.warning(f"⚠️ [AXIS:FLOW] 수급 데이터 수집 실패: {e}")
            map_ant = {}

    df_raw["개인순매수"] = df_raw["종목코드"].map(map_ant).fillna(0)
    if "거래대금(원)" not in df_raw.columns:
        tv = pd.to_numeric(df_raw.get("거래대금(억원)", 0), errors="coerce").fillna(0.0)
        df_raw["거래대금(원)"] = (tv * 100_000_000).astype(float)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. 업종 + 온도
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        if "업종" in df_raw.columns:
            df_raw["업종_상세"] = df_raw["업종"]
            df_raw["업종_대분류"] = df_raw.apply(
                lambda r: classify_big_sector(
                    str(r.get("종목명","")), str(r.get("업종",""))),
                axis=1
            )
        df_raw, _ = add_sector_momentum(df_raw, "업종_대분류")
    except Exception as e:
        logger.warning(f"⚠️ [AXIS:SECTOR] 섹터 분류 실패: {e}")

    ctx.breadth = compute_market_breadth(df_raw)
    mkt_temp = label_market_temp(ctx.breadth.get("ALL", np.nan))
    log(f"🌡 시장 온도: {mkt_temp} (Breadth: {ctx.breadth.get('ALL', 0)}%) -> 동적 가중치 적용")

    try:
        from stop_logic import get_config as _get_stop_cfg
        _scfg = _get_stop_cfg()
        _scfg.market_breadth = ctx.breadth.get('ALL', 50.0)
        log('Stop config: breadth=%.1f adaptive=%s' % (_scfg.market_breadth, _scfg.adaptive_stop))
    except Exception as _e:
        logger.debug(f'Stop config breadth set failed: {_e}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. ML + Trigger + 통합 스코어링
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log("🧠 AI 엔진 동기화 및 통합 스코어링 시작...")

    # [v20.8] Feature Contract 사전 검증 — 불일치 시 ML 비활성
    _fc_ok = True
    try:
        from feature_contract import FEATURE_CONTRACT, validate_features
        log(f"📋 Feature Contract: v={FEATURE_CONTRACT.schema_version}, "
            f"n={FEATURE_CONTRACT.n_features}, hash={FEATURE_CONTRACT.schema_hash}")
        # ml_engine의 FEATURE_COLS와 Contract 동기화 확인
        from ml_engine import FEATURE_COLS as _ml_cols
        _cols_ok, _cols_errs = validate_features(
            pd.DataFrame(columns=_ml_cols), "pipeline_score→ml_engine"
        )
        if not _cols_ok:
            logger.warning(f"⚠️ [AXIS:ML] Feature Contract 불일치: {_cols_errs}")
            _fc_ok = False
    except ImportError:
        _fc_ok = True  # contract 없으면 기존 로직 유지
        logger.debug("feature_contract not available — skipping pre-check")

    # ML Score
    try:
        df_out = ml_engine.apply_ml_score(df_raw, ctx.ohlcv_map)
        df_out["ML_SCORE"] = pd.to_numeric(
            df_out.get("ML_SCORE", 0.0), errors='coerce'
        ).fillna(0.0).clip(0, 100)
    except Exception as e:
        logger.warning(f"⚠️ [AXIS:ML] ML 스코어 실패 → 0점 fallback: {e}")
        df_out = df_raw.copy()
        df_out["ML_SCORE"] = 0.0

    # [v20.6] Trigger Score — 배치 처리 (iterrows 제거)
    df_out['TRIGGER_SCORE'] = _batch_trigger_scores(df_out, ctx.ohlcv_map)
    df_out['RAW_TRIGGER_SCORE'] = df_out['TRIGGER_SCORE']

    ctx.ohlcv_map.clear()
    gc.collect()

    # 통합 스코어 (SSOT: scoring_engine.build_global_score 단일 경로)
    df_out = build_global_score(df_out, ctx.macro_risk)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. Hard Block
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        from validation import apply_hard_blocks, block_summary
        df_out, df_blocked = apply_hard_blocks(df_out)
        if len(df_blocked) > 0:
            _bs = block_summary(df_blocked)
            log(f"🚫 Hard Block: {_bs['total_blocked']}건 제외 {_bs['by_rule']}")
            df_blocked.to_csv(
                os.path.join(OUT_DIR, f"blocked_{ctx.trade_ymd}.csv"),
                index=False, encoding="utf-8-sig"
            )
    except Exception as _hb:
        logger.warning(f"⚠️ [AXIS:VALIDATION] Hard Block 스킵: {_hb}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. ROUTE — [v20.6] 벡터 판정 (apply 제거)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    th = {
        "range_q75": safe_quantile(df_out.get("Range_Pos"), 0.75, 0.8),
        "vol_q75":   safe_quantile(df_out.get("Vol_Quality"), 0.75, 1.2),
    }
    df_out["ROUTE"] = _vec_determine_state_dynamic(df_out, th)
    df_out["상태"] = df_out["ROUTE"]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. 전략 팩토리
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    df_out["STRATEGY"] = "default"
    try:
        from strategies import StrategyFactory
        _ba = ctx.breadth.get("ALL", 50.0)
        _cands = StrategyFactory.select(ctx.macro_risk, _ba)
        log(f"🎯 활성 전략: {_cands} (breadth={_ba:.1f}, macro={ctx.macro_risk})")
        if _cands:
            _all = []
            for _sn, _w in _cands:
                _s = StrategyFactory.create(_sn)
                _fi = _s.filter(df_out)
                _sc = _s.score(_fi)
                _k = max(2, round(5 * _w))
                _p = _s.rank_and_pick(_sc, top_k=_k)
                _all.append(_p)
                log(f"   {'✅' if len(_p) else '⬜'} {_sn}(w={_w:.2f}): "
                    f"필터 {len(_fi)}건 → 선정 {len(_p)}건")
            if _all:
                _sd = pd.concat(_all, ignore_index=True)
                _key = "종목코드" if "종목코드" in _sd.columns else _sd.index.name
                if _key and _key in df_out.columns:
                    _mc = [c for c in ["STRATEGY","STRATEGY_SCORE","STRATEGY_HORIZON"]
                           if c in _sd.columns]
                    if _mc:
                        _sm = _sd.drop_duplicates(_key).set_index(_key)[_mc]
                        for c in _mc:
                            df_out[c] = df_out[_key].map(_sm[c]).fillna(
                                df_out.get(c, "default"))
                log(f"🎯 전략 분류 완료: 매칭 {len(_sd)}건")
    except Exception as _st:
        logger.warning(f"⚠️ [AXIS:STRATEGY] 전략 팩토리 스킵: {_st}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. [v20.6] 점수 설명 컬럼
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    try:
        df_out = generate_score_reasons(df_out, macro_risk=ctx.macro_risk)
        log(f"📝 점수 설명 생성 완료: SCORE_REASON_TOP1/TOP2, SCORE_RISK, ROUTE_REASON")
    except Exception as e:
        logger.warning(f"⚠️ 점수 설명 생성 실패 (무해): {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8. 정예군 편성
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ebs = pd.to_numeric(df_out.get("EBS", 0), errors='coerce').fillna(0)
    struct = pd.to_numeric(df_out.get("STRUCT_SCORE", 0), errors='coerce').fillna(0)
    ms = ~df_out["ROUTE"].isin([Route.OVERHEAT, Route.EXIT_WARNING])
    mq = (ebs >= ctx.pass_ebs) & (struct >= 60) & ms
    dp = df_out[mq].copy().sort_values(["TIMING_SCORE","AI_SCORE"], ascending=False)
    dn = df_out[~mq].copy().sort_values("FINAL_SCORE", ascending=False)
    df_out = pd.concat([dp.head(120), dn, dp.iloc[120:]], ignore_index=True)
    df_out["기준일"] = ctx.trade_ymd
    df_out["시총기준일"] = ctx.mcap_ymd
    ctx.df_out = df_out
    return ctx
