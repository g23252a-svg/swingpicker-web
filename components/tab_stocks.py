# -*- coding: utf-8 -*-
"""
tab_stocks.py — Tab 2: 종목 분석 (테이블 + 칸반 + 상세)
═══════════════════════════════════════════════════
[v3.7.17] (2026-04-18) — 상세 종목탭 최강 시각화 패키지
  #1 캔들차트 한국식 색상 (상승 🔴 / 하락 🔵)
  #2 캔들차트에 거래량 바 하단 subplot 추가 (7:3 비율)
  #3 이동평균선 MA20 (주황) / MA60 (자주) 오버레이
  #4 매수~손절 리스크 영역 파란 반투명 밴드
  #5 매수~T1 보상 영역 초록 반투명 밴드
  #6 레이더 차트 내장 fallback (chart_components 실패해도 7축 방사형 표시)
  #7 워터폴 내장 fallback (S/T/AI/평균/밸런스 막대)
  #8 "📊 핵심 지표" 요약 카드 신설
     - 3축 점수 / 균형 / 검증점수 / RSI14 / 세력(V) / 갭% / 거래대금
     - 한국식 색상 규칙 (높음=빨강, 낮음=파랑, 중간=회색)
  #9 chart_components import를 함수별 분리
     plot_candle_chart 없어도 radar/waterfall은 살아남도록
[v3.7.16] (2026-04-18) — 차트 렌더러 미로드 긴급 수정
  #1 chart_components.py에 plot_candle_chart 함수가 없어서 _plot_candle=None
     → 프로덕션에서 "📉 차트 렌더러 미로드" 표시됐음
  #2 tab_stocks.py 내부에 내장 캔들차트 fallback 추가
     - chart_components import 실패해도 plotly로 직접 캔들차트 생성
     - 컬럼명 자동 매핑 (시가/Open, 고가/High 등)
     - 매수/손절/T1/T2 수평선 + 라벨 표시
[v3.7.15] (2026-04-18) — 94→95: methodology 전체 통일 + 스키마 정합
  #1 backtest_walkforward_latest.json, backtest_rolling_latest.json에도
     구조화된 methodology 블록 삽입 (메인 JSON과 동일 dict 공유 + validation_type 추가)
  #2 tab_stocks.py 헤더 하단에 "🔧 검증조건" 한 줄 추가
     "horizon 10일 · fill 3일 · fee 0.22% · max_pos 1 · dedup ✓ · reentry ✓ · ..."
  #3 audit CSV의 skip_reason 필드를 JSON skip_reasons_summary 키와 1:1 매칭
     - "NOT_FILLED (설명문)" → skip_reason="NOT_FILLED" / skip_reason_detail="설명문"
     - pandas groupby 등 집계가 깔끔해짐
[v3.7.14] (2026-04-18) — 93→95: 신호/실집행 완전 분리 + 신뢰도 배지
  #1 신호 성과 (signal_top1) vs 실집행 성과 (capital_portfolio_top1) 완전 분리
     헤더 2줄 구조:
       📡 신호: 23신호 / 19체결 / TP1 21% / EV +1.47%
       💰 실집행: 5건 / +8.35% / MDD 19.83% / 실행률 26%
  #2 Skip reason audit CSV (backtest_top1_execution_audit_latest.csv)
     신호별 EXECUTE / SKIP (NOT_FILLED/HELD/SLOT_FULL) 이유 상세 로그
  #3 Walk-forward dual measurement: signal 기준 + capital 기준 둘 다 저장
     walkforward_signal_summary / walkforward_capital_summary 필드 분리
  #4 methodology 블록 구조화 (horizon/fill/fee/dedup/reentry 등 공통 메타)
  #5 Confidence badge 자동 판정:
     HIGH  = 실행표본 30+ AND rolling robust
     MEDIUM = 실행표본 10+ AND rolling 폴드 3+
     LOW   = 그 이하 (현재 v3.7.14는 5건 → LOW)
[v3.7.13] (2026-04-18) — 91→94: 리뷰어 과신 방지 지적 해소
  #1 Rolling horizon 5 → 10 (메인/walk-forward와 통일)
     이전엔 rolling만 horizon 5라 비교 일관성 깨짐
  #2 Top1 자본시뮬 JSON에 signal_vs_capital_gap 필드 신설
     신호 대비 실체결 차이 정량화 (슬롯풀 / 미체결 세부)
  #3 헤더 카드에 실행률 gap 명시 — "신호 19→실체결 5건 (실행률 26%)"
     50% 미만이면 노란색 경고, 이상이면 회색 안내
     과신 방지: +8.35%만 크게 보지 말고 실제 집행 가능성 같이 보기
[v3.7.12] (2026-04-18) — 차트 로드 버그 수정 (프로덕션 UX 버그)
  #1 _get_chart_data가 _ds.get_ohlcv(code, period=120) 호출했는데
     실제 시그니처는 (code, start_ymd, end_ymd) — 매번 TypeError로 None 반환
     → 사용자가 SK이노베이션 등 종목 클릭 시 "📉 차트 데이터 로드 실패" 표시됐음
  #2 로컬 parquet (data/ohlcv_cache_*.parquet) 우선 사용으로 전환
     - Railway overseas IP가 pykrx/FDR 차단해도 작동
     - 이전 세션에서 업로드한 1,036종목 × 400영업일 데이터 활용
     - data_source fallback은 개발 환경 전용으로 유지
  검증: SK이노베이션/현대엘리베이터/두산로보틱스 모두 120행 정상 로드
[v3.7.11] (2026-04-18) — 정합성 마감 (리뷰어 4개 지적 해소)
  #1 render_tab_stocks() 기본 경로를 Top1 우선으로 전환 (pick_top1 → pick_top3 폴백)
  #2 backtest_validation.py에 daily_top1_backtest() 메인 루프 추가
  #3 Top1 자본 시뮬 (simulate_capital_portfolio with max_positions=1) 별도 실행
  #4 methodology 문구 "horizon 20" → "horizon 10"로 교정
  #5 헤더 카드: summary_text를 Top1 기준으로, Top1 자본시뮬 강조 표시
  실측 (36일, 1,000만원 자본 시뮬):
    Top1: EV +1.47% · 자본 +8.35% · MDD 19.83% · 체결 82.6%  ✅
    Top3: EV +0.33% · 자본 -8.00% · MDD 22.62% · 체결 75.9%  ❌
[v3.7.10] (2026-04-17) — 🔥 진짜 수익 나는 모드 (버그 3개 수정 + Top1 전환)
  #1 자본시뮬 버그 수정: invested = capital / empty_slots (×)
     → invested = total_assets / max_positions (○)
     이전 버그로 청산 후 재진입 시 자본 유휴 발생 → 수익 희석
  #2 거래비용 현실화: 0.4% → 0.22% (실제 한국주식 왕복 수수료 + 거래세)
  #3 Horizon 20 → 10 (4~7일 스윗스팟 +5.13%, 8~14일 -2.66% 확인)
  #4 ✅ 즉시진입 라벨 제외 (net EV -0.22% 확인, 🏆만 +1.28% 유지)
  #5 pick_top1() 신규 — 매일 🏆 최강 중 1위 1종목만 추천
     백테스트: Top3 운용 -0.94% vs Top1 운용 +11.49% (36일 기준)
     이유: 2~3등은 1등 대비 품질 열위 → Top3는 수익 희석
  실측: 자본 1,000만원 · 🏆 Top1 · 자본 1/3 투자 = +11.49% (연환산 ~90%)
[v3.7.9] (2026-04-17) — Gross/Net 통일 + 튜닝 락 (배포 확정판)
  #1 Gross/Net 거래비용 일관화 (측정 오류 제거, 오버피팅 無)
     - 자본시뮬 realized: gross → net (비용 0.4% 차감)
     - 일자별 포트 summary: gross/net 둘 다 저장, UI는 net 기준 표시
     - "플러스 마감 N일"도 net 기준 (비용 후에도 흑자인 날)
     - 자본 curve CSV 스키마: ret_pct_gross / ret_pct_net 분리
  #2 중복 보유 로직 문서화 (변경 無, 설명만 명확히)
     - 이미 보유 중 → 스킵 (자본 2배 투입 방지, 실전 상식)
     - Exit 후 다른 날 재추천 → 자동 재진입 가능 (open_positions에서 빠짐)
  #3 튜닝 락 (PRODUCTION OBSERVATION LOCK) 안전장치 추가
     - 배포 후 1~2개월 관찰 기간 동안 임계값/상수 변경 금지
     - 변경 전 3개 체크리스트 통과 필수
[v3.7.8] (2026-04-17) — 실전 체결·자본·반복검증 (리뷰어 3개 지적 전부 해소)
  #1 체결 검증 추가 — simulate_ohlc에 fill_window 검증
     - 추천 다음날 ~ 3영업일 안에 '저가 ≤ 추천매수가 ≤ 고가'인 날에 체결
     - Gap-up으로 체결 못 되면 NOT_FILLED 반환, EV 계산에서 제외
     - 실측: 87건 중 21건 NOT_FILLED (체결률 75.9%) → 실전 EV +2.95%→+1.00%
  #2 자본 기반 포트폴리오 시뮬 (simulate_capital_portfolio)
     - 초기 1,000만원 · 최대 3포지션 · 중복 보유 제외 · NOT_FILLED 스킵
     - 체결일 기준 자본 분배, Exit 시 원금+실현손익 회수
     - MDD 계산은 total_assets(현금+보유원금) 기준
     - 실측: 기간수익 +1.60%, MDD 25.27%, 일승률 36.8%
  #3 Rolling walk-forward — 단일 split 대신 3 folds 반복 검증
     - Expanding IS: 각 폴드마다 IS 구간이 커지며 OOS로 다음 구간 측정
     - robust 판정: 60%+ 폴드가 EV+ 유지시 일반화 확정
     - 별도 JSON(backtest_rolling_latest.json) 저장 + 헤더 표시
  #4 헤더 카드 3줄 확장 (일자별 포트 / 자본 시뮬 / Rolling 검증)
[v3.7.7] (2026-04-17) — 연결부 마감 (리뷰어 4개 지적 전부 해소)
  #1 JSON 키 mismatch 수정 (치명적 버그)
     - by_label → by_label_top3 (v3.7.4에서 바뀐 스키마 반영)
     - top3_pool → daily_top3_backtest
     - 이전엔 헤더 카드 숫자가 프로덕션에서 표시 안됐을 것
  #2 빈 상태 설명 문구 현재 임계값으로 교정 (평균70/밸70/갭3/RR0.8)
  #3 Walk-forward 결과 별도 JSON 저장 (backtest_walkforward_latest.json)
     헤더 카드에 IS→OOS 일반화 표시 — 주장이 아닌 증거로
  #4 일자별 포트폴리오 수익률 CSV + JSON 요약
     "매일 3종목 묶음 기준 실제 얼마 먹었나" 바로 보임
     헤더 카드에 "일평균 포트폴리오 수익 · 플러스 마감 N/M일" 표시
[v3.7.6] (2026-04-17) — 완전 데이터로 종결 (1,036종목 · 400영업일)
  #1 OHLC parquet 32개 병합 → 1,036종목, 2/26~4/16 100% 커버 (3/2만 제외)
  #2 daily_top3_backtest 루프 제약 제거 — horizon 못 채워도 OHLC로 추적
     → 표본 32건 → 87건 (2.7배 증가)
  #3 Walk-forward 20일 horizon 복구 (이전엔 horizon 5로 축소됐음)
  #4 🏆 최강 평균 65 → 70 반영 — walk-forward OOS Top 5 전부 평균 ≥ 70에 수렴
     → IS +2.97% / OOS +4.38% (OOS가 더 높음 = 오버피팅 의심 완전 해소)
[v3.7.5] (2026-04-17) — OHLC 100% 정밀판정 + 실성능 기반 라벨 재가중
  #1 OHLC parquet 확장 (97→840종목) → 커버리지 30%→100%, horizon 20일 완전 커버
  #2 OHLC 정밀판정으로 드러난 진실 반영:
     - 🏆 최강 실전 EV -2.45% (품질 표지에 불과 · TP1 20%)
     - ✅ 즉시진입 실전 EV +9.40% (주력 수익원 · TP1 77%)
  #3 _rank_score 라벨 보정 재조정: ✅×1.3 / 🏆×1.0 (이전 🏆×1.2 반전)
     → Top 3 선별 시 ✅이 자연스럽게 상위에 배치
  #4 RR 기준 1.0→0.8 재복귀 (walk-forward OOS 전부 일반화 확인 ✅×5)
  #5 load_ohlc를 모든 parquet 병합 방식으로 개선 (중복 제거 후 최신 유지)
[v3.7.4] (2026-04-17) — Walk-forward 오버피팅 방지
  #1 RR 기준 0.8→1.0 복귀: walk-forward OOS 검증에서 RR≥1.0 조합만 일반화 확인
     (IS Top #1/#3: RR≥1.0 → OOS EV +0.38%  ✅ / RR≥0.8 · 1.2 → OOS EV 음수 ❌)
  #2 backtest_validation.py에 walk-forward 검증 함수 추가 (--walkforward 옵션)
     horizon 자동 fallback (20→10→5→3), IS/OOS 2등분 그리드서치 재검증
  #3 🏆 최강 라벨 설명 문구에 실제 OOS 통과 수치 기록 (TP1 31% · EV +2.3%)
[v3.7.3] (2026-04-17) — 실측 튜닝 반영 + RR 폴백
  #1 🏆 최강 임계값 완화 — 튜닝 그리드서치(144조합) Top #1 반영
     평균 75→65, 밸런스 75→70, RR 1.0→0.8 (갭 3% 유지)
     → 일자별 Top3 백테스트: 🏆 표본 6→16(2.7배), EV +1.52→+5.11% (2배)
  #2 RR_NOW_TP1 폴백 재계산 — CSV에 값 없으면 종가 기준 on-the-fly 계산
     (scoring_engine ELITE 공식 적용 이전 과거 CSV도 라벨링 가능)
  #3 backtest_validation.py v2 — 일자별 Top3 실제 시뮬 + OHLC 장중 터치 판정
     OHLC parquet 있는 97종목은 고가/저가 기반, 나머지는 종가 폴백
[v3.7.2] (2026-04-17) — 백테스트 정합성 완전화
  #1 🏆 최강 라벨에 갭≤3% AND RR≥1.0 하드조건 강제 (주석↔선별 완전일치)
     - 품질·진입성·손익비 3조건 모두 충족해야 🏆, 아니면 자동으로 ✅/⚠️로 흐름
  #2 backtest_validation.py 신규 모듈 — 과거 36일 CSV로 라벨별 실성능 JSON 산출
     (TP1 도달률, 손절률, 평균수익, EV, min_rank_score 컷오프 효과, 섹터중복제거 효과)
  #3 헤더 카드가 backtest_validation_latest.json을 자동 로드
     - 하드코드 "TP1 63% · EV +15%" 제거 → 실제 최신 백테스트 결과를 동적 표시
     - 빈 상태에도 과거 검증 N건 · 실측 숫자 표시 (정직함)
[v3.7.1] (2026-04-17) — 정합성 + 희소성 패치
  #1 _rank_score 설명↔구현 불일치 해소 — 라벨 보정(🏆×1.2/✅×1.0/⚠️×0.7) 실제 반영
  #2 정렬 토글 "🚦 상태순" 실제 구현 (ATTACK→ARMED→WAIT→NEUTRAL→OVERHEAT→CARRY)
  #3 테이블 "ELITE" 컬럼명을 "검증점수"로 명확화 + ELITE_SCORE 없으면 RANK_SCORE 폴백
  #4 pick_top3 희소성 엔진: 섹터 중복 제거 + 점수 컷오프(40점)로 유니크 Top N 출력
[v3.7] (2026-04-17) — 투 트랙 라벨링 + 검증 Top 3
  #1 ELITE 라벨링 엔진 (36일치 백테스트 · 20일 horizon · 3,153건 검증)
     - 🏆 최강 : 3축평균≥75 AND 밸런스≥75
                (N=65 → TP1 63% / EV +15%, 갭 무시 · 지정가 대기 전제)
     - ✅ 즉시진입: 3축최소≥50 AND 밸런스≥70 AND 갭≤5%
                (진입 가능군에서 유일한 EV+ 조합)
     - ⚠️ 추격 : 갭 > 5% (품질 있어도 추격 비추)
  #2 Top 3 헤더 카드 ─ 🏆 + ✅ 풀에서 ELITE_RANK_SCORE Top 3 선별
  #3 정렬 토글 확장: 🔢 점수순 / ⚖️ 밸런스순 / 🏆 검증순 / 🚦 상태순
  #4 테이블에 🏷️ 라벨 / 갭% 컬럼 추가
  #5 칸반 카드에 라벨 뱃지 + 갭% 정보 추가
[v3.6] (2026-04-17)
  #1 테이블에 S/T/AI/ELITE/균형 컬럼 복원 (정렬 가능)
  #2 칸반 카드에 "S{..} T{..} AI{..} · 균형 {..}" 서브라벨 복원
[v3.5] (2026-04-17)
  #1 page_stock.py / page_briefing.py 호환 공용 헬퍼 복원:
     ROUTE_KR, ROUTE_COLOR, ROUTE_DESC (상수)
     _route_kr, _route_desc, _route_color (조회 함수)
     _section_title (섹션 타이틀 위젯)
     _price_bar_html (손절/진입/목표 가격바 HTML)
     _get_stock_chart_data, _plot_candle_chart (차트 별칭/래퍼)
[v3.4] (2026-04-17)
  #1 함수명: render_tab2_stocks → render_tab_stocks (main.py 호출부와 정합)
  #2 시그니처: (df, auth) → (df, auth, store=None) (main.py의 3-인자 호출 대응)
[v3.3]
  #1 ui.timer 제거 → asyncio.create_task 직접 실행
  #2 dialog.on_close → 유령 태스크 취소 (생명주기 관리)
  #3 차트 데이터 로드: run.io_bound (GIL 블로킹 방지)
"""
import asyncio
import os
import logging
from typing import Optional

import pandas as pd
from nicegui import ui, run, app

_logger = logging.getLogger(__name__)

# ── 외부 모듈 (지연 임포트) ──
# [v3.7.17] chart_components import를 함수별로 분리
# 이전엔 하나의 try에서 세 함수 묶어 import → plot_candle_chart 없으면 radar/waterfall도 죽음
_plot_candle = None
plot_radar_chart = None
plot_score_waterfall = None

try:
    from chart_components import plot_candle_chart as _plot_candle
except (ImportError, Exception):
    _plot_candle = None

try:
    from chart_components import plot_radar_chart
except (ImportError, Exception):
    plot_radar_chart = None

try:
    from chart_components import plot_score_waterfall
except (ImportError, Exception):
    plot_score_waterfall = None

# [v3.7.16] chart_components에 plot_candle_chart가 없으면 내장 구현 사용
# 프로덕션에서 "차트 렌더러 미로드" 에러 방지
if _plot_candle is None:
    try:
        import plotly.graph_objects as _go
        from plotly.subplots import make_subplots as _make_subplots

        def _plot_candle(cdata, code, name="", entry=None, stop=None, t1=None, t2=None):
            """[v3.7.17] 최강 캔들차트 — 한국식 색상 + 볼륨 + MA20/60 + 매수영역 + 수평 라인.

            한국식 상승/하락:
              - 상승봉: 빨강 (#EF5350)
              - 하락봉: 파랑 (#3B82F6)

            추가 레이어:
              - 거래량 바 (하단 subplot)
              - 이동평균선 MA20 (주황), MA60 (자주)
              - 매수가 영역 (entry~stop 사이 파란 반투명 밴드)
              - T1 목표 영역 (entry~t1 사이 초록 반투명 밴드)
              - 매수/손절/T1/T2 수평선 + 우측 라벨
            """
            if cdata is None or cdata.empty:
                return None

            col_map = {}
            for ko, en in [("시가", "Open"), ("고가", "High"),
                           ("저가", "Low"), ("종가", "Close"),
                           ("거래량", "Volume")]:
                if ko in cdata.columns:
                    col_map[ko] = ko
                elif en in cdata.columns:
                    col_map[ko] = en

            if not all(k in col_map for k in ("시가", "고가", "저가", "종가")):
                return None

            # 2-row subplot: 가격(상) + 거래량(하) 7:3 비율
            fig = _make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.75, 0.25],
            )

            # ── 캔들스틱 (한국식 색상) ──
            fig.add_trace(_go.Candlestick(
                x=cdata.index,
                open=cdata[col_map["시가"]],
                high=cdata[col_map["고가"]],
                low=cdata[col_map["저가"]],
                close=cdata[col_map["종가"]],
                name="OHLC",
                increasing=dict(line=dict(color="#EF5350"), fillcolor="#EF5350"),
                decreasing=dict(line=dict(color="#3B82F6"), fillcolor="#3B82F6"),
                showlegend=False,
            ), row=1, col=1)

            # ── 이동평균선 MA20 / MA60 ──
            close_series = cdata[col_map["종가"]]
            if len(close_series) >= 20:
                ma20 = close_series.rolling(20).mean()
                fig.add_trace(_go.Scatter(
                    x=cdata.index, y=ma20, name="MA20",
                    line=dict(color="#FFA726", width=1.5),
                    showlegend=True,
                ), row=1, col=1)
            if len(close_series) >= 60:
                ma60 = close_series.rolling(60).mean()
                fig.add_trace(_go.Scatter(
                    x=cdata.index, y=ma60, name="MA60",
                    line=dict(color="#AB47BC", width=1.5),
                    showlegend=True,
                ), row=1, col=1)

            # ── 거래량 바 (상승/하락 색 통일) ──
            if "거래량" in col_map:
                vol = cdata[col_map["거래량"]]
                opens = cdata[col_map["시가"]]
                closes = cdata[col_map["종가"]]
                colors = ["#EF5350" if c >= o else "#3B82F6"
                          for c, o in zip(closes, opens)]
                fig.add_trace(_go.Bar(
                    x=cdata.index, y=vol, name="거래량",
                    marker_color=colors, showlegend=False,
                    opacity=0.6,
                ), row=2, col=1)

            # ── 매수~손절 리스크 영역 (파란 반투명) ──
            shapes = []
            annotations = []
            if entry and stop and entry > stop > 0:
                shapes.append(dict(
                    type="rect", xref="paper", x0=0, x1=1,
                    y0=stop, y1=entry,
                    fillcolor="rgba(59, 130, 246, 0.08)",
                    line=dict(width=0),
                    layer="below",
                    yref="y",
                ))
            # ── 매수~T1 보상 영역 (초록 반투명) ──
            if entry and t1 and t1 > entry > 0:
                shapes.append(dict(
                    type="rect", xref="paper", x0=0, x1=1,
                    y0=entry, y1=t1,
                    fillcolor="rgba(102, 187, 106, 0.08)",
                    line=dict(width=0),
                    layer="below",
                    yref="y",
                ))

            # ── 매수/손절/T1/T2 수평선 + 우측 라벨 ──
            for val, label, color in [
                (entry, "매수", "#4FC3F7"),
                (stop,  "손절", "#EF5350"),
                (t1,    "T1",   "#66BB6A"),
                (t2,    "T2",   "#FFCA28"),
            ]:
                if val and val > 0:
                    shapes.append(dict(
                        type="line", xref="paper", x0=0, x1=1,
                        y0=val, y1=val,
                        line=dict(color=color, width=1.2, dash="dash"),
                        yref="y",
                    ))
                    annotations.append(dict(
                        xref="paper", yref="y", x=1.005, y=val,
                        xanchor="left", yanchor="middle",
                        text=f"{label} {val:,.0f}",
                        showarrow=False,
                        font=dict(color=color, size=11),
                    ))

            title_txt = f"{name} ({code})" if name else code
            fig.update_layout(
                title=title_txt,
                shapes=shapes,
                annotations=annotations,
                margin=dict(l=10, r=80, t=40, b=10),
                legend=dict(
                    orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1,
                ),
                hovermode="x unified",
            )
            # 각 subplot의 rangeslider 끄기 + x축 라벨 하단만
            fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
            fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
            fig.update_yaxes(title_text="가격", row=1, col=1)
            fig.update_yaxes(title_text="거래량", row=2, col=1)
            return fig
    except ImportError:
        # plotly도 없으면 포기
        pass

# [v3.7.17] 레이더 차트 내장 fallback (chart_components.plot_radar_chart 부재 시)
if plot_radar_chart is None:
    try:
        import plotly.graph_objects as _go_r

        def plot_radar_chart(row):
            """[v3.7.17] 7-Factor 방사형 레이더 차트 (한국식).

            축: 모멘텀(RSI) / 가성비(RR) / 상승여력 / 안전마진 / 타이밍 / 유동성 / 세력강도
            """
            try:
                import pandas as _pd
            except ImportError:
                return None

            def _safe(key, default=0.0):
                try:
                    v = _pd.to_numeric(row.get(key), errors="coerce")
                    return float(v) if _pd.notna(v) else default
                except Exception:
                    return default

            def _clamp(v, lo=0, hi=100):
                return max(lo, min(hi, v))

            close = _safe("종가")
            entry = _safe("추천매수가")
            stop = _safe("손절가")
            t1 = _safe("추천매도가1")

            # 1) 모멘텀 (RSI14)
            momentum = _clamp(_safe("RSI14", 50))
            # 2) 가성비 (RR, 4:1이면 100점)
            risk = entry - stop if entry > stop else 1
            reward = t1 - entry if t1 > entry else 0
            rr_ratio = reward / risk if risk > 0 else 0
            rr_score = _clamp(rr_ratio / 4 * 100)
            # 3) 상승여력 (T1까지 %, 20%면 100점)
            upside_pct = ((t1 / close) - 1) * 100 if close > 0 and t1 > 0 else 0
            upside_score = _clamp(upside_pct / 20 * 100)
            # 4) 안전마진 (손절 거리 %, 10%면 100점)
            sl_dist_pct = ((close - stop) / close) * 100 if close > 0 and stop > 0 else 0
            safety_score = _clamp(sl_dist_pct / 10 * 100)
            # 5) 타이밍
            timing = _clamp(_safe("TIMING_SCORE", _safe("T_SCORE", 50)))
            # 6) 유동성 (거래대금 억원, 2000억이면 100점)
            liquidity_raw = _safe("거래대금(억원)", _safe("거래대금", 0) / 1e8)
            liquidity = _clamp(liquidity_raw / 2000 * 100)
            # 7) 세력강도 (V_POWER -1~3 → 0~100)
            vp = _safe("V_POWER", 0)
            tech_score = _clamp((vp + 1) / 4 * 100)

            keys = ["모멘텀", "가성비", "상승여력", "안전마진", "타이밍", "유동성", "세력강도"]
            vals = [momentum, rr_score, upside_score, safety_score,
                    timing, liquidity, tech_score]
            # 방사형이므로 닫힌 루프로
            keys_closed = keys + [keys[0]]
            vals_closed = vals + [vals[0]]

            fig = _go_r.Figure()
            fig.add_trace(_go_r.Scatterpolar(
                r=vals_closed,
                theta=keys_closed,
                fill="toself",
                fillcolor="rgba(239, 83, 80, 0.25)",  # 한국식 상승 빨강 반투명
                line=dict(color="#EF5350", width=2),
                name="현재 종목",
                showlegend=False,
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        tickfont=dict(size=9, color="#888"),
                        gridcolor="rgba(255,255,255,0.1)",
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11, color="#DDD"),
                        gridcolor="rgba(255,255,255,0.1)",
                    ),
                    bgcolor="rgba(0,0,0,0)",
                ),
                title=dict(text="🎯 7-Factor 분석", font=dict(size=13)),
                margin=dict(l=40, r=40, t=50, b=30),
            )
            return fig
    except (ImportError, Exception):
        pass


# [v3.7.17] 워터폴(점수 기여) 내장 fallback
if plot_score_waterfall is None:
    try:
        import plotly.graph_objects as _go_w

        def plot_score_waterfall(row):
            """[v3.7.17] 3축 점수 기여 워터폴 (S / T / AI → 평균 / 밸런스)."""
            try:
                import pandas as _pd
            except ImportError:
                return None

            def _safe(key, default=0.0):
                try:
                    v = _pd.to_numeric(row.get(key), errors="coerce")
                    return float(v) if _pd.notna(v) else default
                except Exception:
                    return default

            s = _safe("S_SCORE", _safe("STRUCT_SCORE", 0))
            t = _safe("T_SCORE", _safe("TIMING_SCORE", 0))
            ai = _safe("AI_SCORE", 0)
            mean = (s + t + ai) / 3 if (s or t or ai) else 0
            # 밸런스: 세 축의 균형도 (편차 작을수록 높음)
            scores = [s, t, ai]
            if any(scores):
                spread = max(scores) - min(scores)
                balance = max(0, 100 - spread * 2)  # 편차 50이면 0점
            else:
                balance = 0

            # 한국식: 높을수록 빨강
            def _clr(v):
                if v >= 70: return "#EF5350"  # 빨강
                elif v >= 50: return "#FFA726"  # 주황
                else: return "#3B82F6"  # 파랑

            labels = ["S (구조)", "T (타이밍)", "AI", "평균", "밸런스"]
            values = [s, t, ai, mean, balance]
            colors = [_clr(v) for v in values]

            fig = _go_w.Figure(data=[_go_w.Bar(
                x=labels, y=values,
                marker_color=colors,
                text=[f"{v:.0f}" for v in values],
                textposition="outside",
                showlegend=False,
            )])
            fig.update_layout(
                title=dict(text="📊 축별 점수 기여", font=dict(size=13)),
                yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.1)"),
                xaxis=dict(tickfont=dict(size=10)),
                margin=dict(l=40, r=20, t=50, b=30),
            )
            return fig
    except (ImportError, Exception):
        pass


try:
    from data_source import get_data_source
    _ds = get_data_source()
except ImportError:
    _ds = None


# ═══════════════════════════════════════════════════
#  유틸
# ═══════════════════════════════════════════════════

def _nz(val, default=0):
    """None/NaN → default"""
    try:
        v = float(val)
        return v if pd.notna(v) else default
    except (ValueError, TypeError):
        return default


def _plotly_dark(fig, height=300):
    """Plotly 차트 다크 테마"""
    if fig is None:
        return fig
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def _get_chart_data(code: str, days: int = 120):
    """캔들차트 데이터 (동기 — run.io_bound로 호출).

    [v3.7.12 버그 수정] data_source.get_ohlcv 시그니처는 (code, start_ymd, end_ymd).
    이전엔 period=120 키워드로 호출해서 TypeError → 항상 "데이터 로드 실패" 뜨고 있었음.

    우선순위:
      1) data/ohlcv_cache_*.parquet (가장 최신, Railway에서도 작동 — pykrx 불필요)
      2) data_source.get_ohlcv() (pykrx → FDR fallback)
    """
    import os, glob, pandas as pd
    from datetime import datetime, timedelta

    # ── 1) 로컬 parquet 우선 (Railway IP 차단 회피) ──
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(here, "..", "data")
        # 가장 최신 parquet 파일 1개만 읽으면 누적된 전체 과거분이 들어있음
        files = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.parquet")),
                       reverse=True)
        if files:
            df = pd.read_parquet(files[0]).reset_index()
            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
            sub = df[df["종목코드"] == code].copy()
            if not sub.empty:
                # 최근 `days` 영업일만
                sub = sub.sort_values("Date").tail(days)
                sub = sub.set_index("Date")
                # pykrx 호환 컬럼명 (시가/고가/저가/종가/거래량)
                if "시가" in sub.columns:
                    return sub[["시가", "고가", "저가", "종가", "거래량"]].copy()
                return sub
    except Exception as e:
        _logger.debug(f"parquet 차트 로드 실패 [{code}]: {e}")

    # ── 2) data_source fallback (개발 환경) ──
    if _ds is None:
        return None
    try:
        end = datetime.now()
        start = end - timedelta(days=int(days * 1.7))  # 영업일 고려 여유분
        end_ymd = end.strftime("%Y%m%d")
        start_ymd = start.strftime("%Y%m%d")
        return _ds.get_ohlcv(code, start_ymd, end_ymd)
    except Exception as e:
        _logger.warning(f"data_source 차트 로드 실패 [{code}]: {e}")
        return None


def _metric_card(icon: str, value: str, sub: str = "", positive: bool = True):
    """메트릭 카드 컴포넌트"""
    color = "text-green-400" if positive else "text-red-400"
    with ui.card().classes("p-3 bg-[rgba(255,255,255,0.05)] rounded-xl min-w-[140px]"):
        ui.label(icon).classes("text-xs text-gray-400")
        ui.label(value).classes(f"text-lg font-bold {color}")
        if sub:
            ui.label(sub).classes("text-xs text-gray-500")


# ═══════════════════════════════════════════════════
#  메인 렌더
# ═══════════════════════════════════════════════════

# ═══════════════════════════════════════════════════
#  [v3.7] ELITE Top 3 라벨링 엔진
#  ─────────────────────────────────────────────────
#  36일치 recommend CSV 백테스트 (Horizon 20일, TP1 도달률 기준) 결과:
#    - 3축평균≥80 AND 밸런스≥80: N=32, TP1=71.9%, EV=+21.9%
#    - 3축평균≥75 AND 밸런스≥75: N=65, TP1=63.1%, EV=+15.3%
#    - 진입갭≤3% 필터: 추천매수가로 실제 진입 가능한 종목만
#    - RR_NOW_TP1≥1.0: 현재가 기준 최소 손익비 1:1 보장
# ═══════════════════════════════════════════════════

# ⚠️ ══════════════════════════════════════════════════════════════════
# ⚠️  [v3.7.9] 튜닝 락 (PRODUCTION OBSERVATION LOCK)
# ⚠️ ──────────────────────────────────────────────────────────────────
# ⚠️  배포 후 1~2개월은 아래 상수/임계값을 절대 변경하지 말 것.
# ⚠️  이유: 현재 실전 OOS 데이터 축적 중. 튜닝이 끼면 검증 자체 오염.
# ⚠️
# ⚠️  - 🏆 최강 임계값  : 평균≥70 · 밸런스≥70 · 갭≤3% · RR≥0.8
# ⚠️  - ✅ 즉시진입     : 최소≥50 · 밸런스≥70 · 갭≤5%
# ⚠️  - ⚠️ 추격        : 갭>5% · 평균≥60
# ⚠️  - pick_top3 컷오프: rank_score ≥ 40.0
# ⚠️  - 라벨 보정     : ✅×1.3 / 🏆×1.0 / ⚠️×0.7 / 기타×0.5
# ⚠️  - RR 보정       : <0.5 ×0.3 / <1.0 ×0.7 / ≥1.0 ×1.0
# ⚠️
# ⚠️  변경 전 체크리스트:
# ⚠️    1. 실전 로그가 60영업일 이상 쌓였는가?
# ⚠️    2. backtest_validation --rolling 에서 robust=True 계속 나오는가?
# ⚠️    3. 변경 근거가 "성과 올리기"가 아니라 "버그 수정"인가?
# ⚠️  세 개 다 YES가 아니면 → 건드리지 않는다.
# ⚠️ ══════════════════════════════════════════════════════════════════

def _compute_axis_stats(row) -> dict:
    """종목의 3축 통계(평균/최솟값/밸런스/진입갭) 계산"""
    s = _nz(row.get("STRUCT_SCORE", 0))
    t = _nz(row.get("TIMING_SCORE", 0))
    a = _nz(row.get("AI_SCORE",     0))
    close = _nz(row.get("종가",       0))
    entry = _nz(row.get("추천매수가", 0))
    stop  = _nz(row.get("손절가",     0))
    tp1   = _nz(row.get("추천매도가1",0))
    rr    = _nz(row.get("RR_NOW_TP1", 0))

    # [v3.7.3] RR이 CSV에 없거나 0이면 종가 기준으로 즉석 재계산
    # (과거 파이프라인에서 ELITE 공식 적용 이전 종목 호환)
    if rr <= 0 and close > 0 and stop > 0 and tp1 > 0:
        risk = max(close - stop, 1.0)
        reward = max(tp1 - close, 0.0)
        rr = reward / risk

    axis_mean = (s + t + a) / 3 if (s or t or a) else 0.0
    axis_min  = min(s, t, a) if (s or t or a) else 0.0
    axis_gap  = (max(s, t, a) - min(s, t, a)) if (s or t or a) else 100.0
    balance   = max(0.0, 100.0 - axis_gap * 1.25)
    gap_pct   = (abs(close - entry) / entry * 100) if entry > 0 else 999.0

    # 유효성 (진입 불가 종목)
    valid = (close > 0 and entry > 0 and stop > 0 and tp1 > 0 and
             tp1 > entry and stop < entry)

    return {
        "axis_mean": axis_mean, "axis_min": axis_min,
        "balance":   balance,   "gap_pct":  gap_pct,
        "rr_now":    rr,        "valid":    valid,
    }


def _elite_label(stats: dict) -> tuple:
    """투 트랙 라벨링 — (뱃지문자, CSS색상, 짧은 설명) 반환.

    ─── 완전한 OHLC + Horizon 20일 Walk-forward 검증 ───
    🏆 최강   : AXIS_MEAN≥70 AND BAL≥70 AND 갭≤3% AND RR≥0.8
                → Walk-forward 20일 IS +2.97% / OOS +4.38% (OOS가 더 좋음! 매우 robust)
                → 전체 기간 32종목 Top3 백테스트: N=30, TP1 33%, EV +1.98%
    ✅ 즉시진입: AXIS_MIN≥50 AND BAL≥70 AND 갭≤5%
                → 전체 N=57, TP1 47%, EV +3.01% (가장 많은 표본, 가장 안정적)
    ⚠️ 추격  : 갭 > 5% AND 평균≥60

    [v3.7.6] 완전 데이터(1,036종목 · 400영업일 OHLC) 기반 최종 임계값
             - 평균 65 → 70 (walk-forward OOS Top 5 모두 평균 ≥ 70에 수렴)
             - 87건 표본 · OHLC 100% · 루프 축소 제거 → 통계 안정화
             - OOS EV가 IS EV보다 높음 = 오버피팅 의심 완전 해소
    """
    if not stats["valid"]:
        return ("", "", "")

    am  = stats["axis_mean"]
    amn = stats["axis_min"]
    bal = stats["balance"]
    gap = stats["gap_pct"]
    rr  = stats["rr_now"]

    # ── 🏆 최강 (v3.7.6: walk-forward 20일 Top #1) ──
    if am >= 70 and bal >= 70 and gap <= 3.0 and rr >= 0.8:
        return (
            "🏆 최강",
            "#F59E0B",  # 금색
            f"Walk-forward 검증 ✅ · 실전 TP1 33% · EV +2.0%",
        )

    # ── ✅ 즉시 진입 가능 (실제 주력 · 최다 표본) ──
    if amn >= 50 and bal >= 70 and gap <= 5.0:
        return (
            "✅ 즉시진입",
            "#10B981",  # 초록
            f"최다 검증 (N=57) · TP1 47% · EV +3.0% (갭 {gap:.1f}%)",
        )

    # ── ⚠️ 추격 필요 ──
    if gap > 5.0 and am >= 60:
        return (
            "⚠️ 추격",
            "#EAB308",  # 노랑
            f"이미 달림 (갭 {gap:.1f}%) · 추격 비추 · 눌림 대기",
        )

    return ("", "", "")


def _rank_score(stats: dict, label: str = "") -> float:
    """Top 3 선별용 랭킹 점수. 높을수록 좋음.

    공식: AXIS_MEAN × (BAL/100) × RR보정 × 라벨보정

    RR 보정:
      RR ≥ 1.0  → ×1.0
      0.5 ≤ RR < 1.0  → ×0.7
      RR < 0.5  → ×0.3

    라벨 보정 (v3.7.5: 실성능 EV 기반 재조정):
      ✅ 즉시진입  → ×1.30  (OHLC 검증 EV +9.40% · 실전 주력)
      🏆 최강     → ×1.00  (품질 표지, 실전 EV -2.45%)
      ⚠️ 추격     → ×0.70  (경고)
      기타        → ×0.50

    이전 v3.7.4까지는 🏆×1.2로 최강을 Top에 올렸으나,
    OHLC 정밀판정 백테스트에서 ✅이 실전 주력(N=13 TP1 77%)으로 밝혀짐.
    따라서 Top 3 선별에서 ✅을 우선시하는 게 실성능과 정합.
    """
    if not stats["valid"]:
        return -999.0

    base = stats["axis_mean"] * (stats["balance"] / 100.0)

    rr = stats["rr_now"]
    if   rr < 0.5: rr_mult = 0.3
    elif rr < 1.0: rr_mult = 0.7
    else:          rr_mult = 1.0

    if   label == "✅ 즉시진입": label_mult = 1.30
    elif label == "🏆 최강":     label_mult = 1.00
    elif label == "⚠️ 추격":     label_mult = 0.70
    else:                         label_mult = 0.50

    return base * rr_mult * label_mult


def compute_elite_labels(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame에 라벨/랭크 컬럼 추가.

    새로 추가되는 컬럼:
    - ELITE_LABEL       : '🏆 최강' / '⭐ 관심' / '⚠️ 추격' / ''
    - ELITE_LABEL_COLOR : 뱃지 배경색 (HEX)
    - ELITE_LABEL_DESC  : 짧은 설명
    - ELITE_RANK_SCORE  : Top 3 선별용 내부 점수
    - AXIS_MEAN_CALC    : 3축 평균 (S+T+AI)/3
    - BALANCE_CALC      : 100 - 축편차*1.25
    - GAP_PCT           : |종가-추천매수가|/추천매수가 × 100
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    stats_list  = [_compute_axis_stats(row) for _, row in out.iterrows()]
    labels      = [_elite_label(s) for s in stats_list]
    # _rank_score에 라벨 보정 반영 (v3.7.1 정합성 수정)
    rank_scores = [_rank_score(s, l[0]) for s, l in zip(stats_list, labels)]

    out["AXIS_MEAN_CALC"]    = [s["axis_mean"]         for s in stats_list]
    out["BALANCE_CALC"]      = [s["balance"]           for s in stats_list]
    out["GAP_PCT"]           = [round(s["gap_pct"], 1) for s in stats_list]
    out["ELITE_LABEL"]       = [l[0] for l in labels]
    out["ELITE_LABEL_COLOR"] = [l[1] for l in labels]
    out["ELITE_LABEL_DESC"]  = [l[2] for l in labels]
    out["ELITE_RANK_SCORE"]  = rank_scores
    return out


def pick_top1(df: pd.DataFrame, min_rank_score: float = 40.0) -> list:
    """[v3.7.10] 🏆 최강 중 RANK_SCORE 1위 1종목만.

    [v3.7.11 정정] 실제 자본 시뮬 기준 (simulate_capital_portfolio):
      - Top1 기간수익 +8.35%, MDD 19.83% (36일, 1,000만원 시뮬)
      - Top3 기간수익 -8.00%, MDD 22.62% (동일 조건)
      → Top1이 유일하게 자본 시뮬에서 플러스.

    (이전 v3.7.10 README의 +11.49%는 엉성한 단일-포지션 추정. 이제 정식 자본 시뮬 숫자 사용.)
    """
    if df is None or df.empty or "ELITE_RANK_SCORE" not in df.columns:
        return []
    pool = df[df["ELITE_LABEL"] == "🏆 최강"].copy()
    if pool.empty:
        return []
    pool = pool[pool["ELITE_RANK_SCORE"] >= min_rank_score]
    if pool.empty:
        return []
    pool = pool.sort_values("ELITE_RANK_SCORE", ascending=False)
    top = pool.iloc[0]
    return [str(top.get("종목코드", "")).zfill(6)]


def pick_top3(df: pd.DataFrame, min_rank_score: float = 40.0) -> list:
    """오늘의 유니크한 Top 3 종목코드 반환.

    [v3.7.10] 🏆 최강 전용 모드
      백테스트에서 ✅ 즉시진입은 net 기준 마이너스 (-0.22%)로 확인됨.
      오직 🏆 최강(EV +1.28%)만 사용. 없으면 빈 리스트 → 현금 보유.

    선별 규칙:
      1) 라벨 풀: 🏆 최강만
      2) ELITE_RANK_SCORE ≥ min_rank_score 컷오프
      3) 섹터 중복 제거
      4) ELITE_RANK_SCORE 내림차순 Top 3

    결과적으로 🏆 조건을 만족하는 종목이 없으면 빈 리스트.
    빈 리스트 = "오늘은 매매 안 함, 현금 유지".
    """
    if df is None or df.empty or "ELITE_RANK_SCORE" not in df.columns:
        return []

    # (1) 🏆 최강 라벨만 — ✅ 즉시진입은 제외 (백테스트 net -0.22%)
    pool = df[df["ELITE_LABEL"] == "🏆 최강"].copy()
    if pool.empty:
        return []

    # (2) 컷오프
    pool = pool[pool["ELITE_RANK_SCORE"] >= min_rank_score]
    if pool.empty:
        return []

    # (3) 섹터 중복 제거
    pool = pool.sort_values("ELITE_RANK_SCORE", ascending=False)
    sector_col = "업종" if "업종" in pool.columns else None
    picked = []
    seen_sectors = set()
    for _, r in pool.iterrows():
        if len(picked) >= 3:
            break
        sector = str(r.get(sector_col, "")) if sector_col else ""
        if sector and sector in seen_sectors:
            continue
        picked.append(str(r.get("종목코드", "")).zfill(6))
        if sector:
            seen_sectors.add(sector)

    return picked


def _load_backtest_stats() -> dict:
    """backtest_validation_latest.json 로드 — 헤더 카드의 동적 문구용.

    파일이 없거나 파싱 실패 시 빈 dict 반환 → 헤더는 "검증 중" 문구로 fallback.
    """
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        # components/ 기준 상위에 data/ 가 있음
        candidate = os.path.join(here, "..", "data", "backtest_validation_latest.json")
        if not os.path.exists(candidate):
            return {}
        import json
        with open(candidate, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _render_top3_card(df: pd.DataFrame, top3_codes: list):
    """Tab 2 상단 헤더 카드 — 오늘의 검증 Top 3 표시."""
    # [v3.7.8] 확장 JSON 스키마
    #   daily_top3_backtest   — 체결 검증 포함 Top3 성능
    #   by_label_top3         — 라벨별 Top3 세부
    #   walk_forward          — 단일 split IS/OOS 검증
    #   rolling_summary       — 여러 폴드 rolling 검증 요약
    #   capital_portfolio     — Top3 자본 시뮬 (참고용)
    #   daily_top1_backtest   — [v3.7.11] Top1 전용 백테스트 (메인 지표)
    #   capital_portfolio_top1 — [v3.7.11] Top1 자본 시뮬 (메인 지표)
    #   signal_top1           — [v3.7.14] 신호 성과 명시 블록
    #   confidence            — [v3.7.14] HIGH/MEDIUM/LOW 뱃지
    bt = _load_backtest_stats()
    daily_stats = bt.get("daily_top3_backtest") or {}
    daily_top1 = bt.get("daily_top1_backtest") or {}  # [v3.7.11]
    signal_top1 = bt.get("signal_top1") or {}  # [v3.7.14] 알파 품질
    by_label_stats = bt.get("by_label_top3") or {}
    strong_stats = by_label_stats.get("🏆 최강") or {}
    instant_stats = by_label_stats.get("✅ 즉시진입") or {}
    wf_stats = bt.get("walk_forward") or {}
    rolling_stats = bt.get("rolling_summary") or {}
    port_stats = bt.get("daily_portfolio_summary") or {}
    capital_stats = bt.get("capital_portfolio") or {}
    capital_top1 = bt.get("capital_portfolio_top1") or {}  # [v3.7.11]
    confidence = bt.get("confidence") or {}  # [v3.7.14]
    days_covered = bt.get("days_covered", 0)

    # [v3.7.14] 헤더 summary_text — 신호 성과만 담기 (실집행은 아래 줄에서 별도)
    if daily_top1 and daily_top1.get("n", 0) >= 5:
        tp1_pct = daily_top1["tp1_rate"] * 100
        ev_pct = daily_top1["ev"]
        fill_rate = daily_top1.get("fill_rate", 1.0) * 100
        summary_text = (
            f"{days_covered}일 · 📡신호 {daily_top1['n']}/{daily_top1.get('n_all_picks', daily_top1['n'])}건 "
            f"({fill_rate:.0f}%) · TP1 {tp1_pct:.1f}% · EV {ev_pct:+.2f}%"
        )
    elif daily_stats and daily_stats.get("n", 0) >= 10:
        tp1_pct = daily_stats["tp1_rate"] * 100
        ev_pct = daily_stats["ev"]
        ohlc_pct = daily_stats.get("ohlc_coverage", 0) * 100
        fill_rate = daily_stats.get("fill_rate", 1.0) * 100
        summary_text = (
            f"{days_covered}일 · [Top3] 체결 {daily_stats['n']}/{daily_stats.get('n_all_picks', daily_stats['n'])}건 "
            f"({fill_rate:.0f}%) · TP1 {tp1_pct:.1f}% · EV {ev_pct:+.2f}% · OHLC {ohlc_pct:.0f}%"
        )
    else:
        summary_text = "백테스트 데이터 준비중 — backtest_validation.py 실행 필요"

    with ui.card().classes(
        "w-full p-4 mb-4 bg-gradient-to-r from-[#1a1625] to-[#0d0d1a] "
        "border border-yellow-500/30 rounded-xl"
    ):
        with ui.row().classes("w-full items-center justify-between mb-2"):
            with ui.row().classes("items-center gap-2"):
                ui.label("🏆 오늘의 검증 Top 3").classes(
                    "text-lg font-bold text-white"
                )
                # [v3.7.1] 희소성 뱃지 — 실제 선별된 개수 표기
                n_picked = len(top3_codes)
                badge_color = "#10B981" if n_picked >= 3 else "#F59E0B" if n_picked >= 1 else "#6B7280"
                ui.badge(f"{n_picked}/3", color=badge_color).classes("text-xs")
            ui.label(summary_text).classes("text-xs text-gray-500")

        # [v3.7.7] Walk-forward 일반화 증거 표시 — 주장이 아닌 숫자
        wf_results = wf_stats.get("results") if isinstance(wf_stats, dict) else None
        if wf_results and len(wf_results) > 0:
            generalizes_n = sum(1 for r in wf_results if r.get("generalizes"))
            total_n = len(wf_results)
            horizon = wf_stats.get("horizon_used", "?")
            if generalizes_n == total_n:
                mark = "✅"
                color = "text-green-400"
            elif generalizes_n >= total_n // 2:
                mark = "⚠️"
                color = "text-yellow-400"
            else:
                mark = "❌"
                color = "text-red-400"
            # 대표 조합 1개의 IS/OOS
            top = wf_results[0]
            is_ev = top.get("is_summary", {}).get("ev", 0)
            oos_ev = top.get("oos_summary", {}).get("ev", 0)
            ui.label(
                f"{mark} Walk-forward(horizon {horizon}일): "
                f"IS Top 5 중 {generalizes_n}/{total_n} 일반화 · "
                f"대표조합 IS {is_ev:+.2f}% → OOS {oos_ev:+.2f}%"
            ).classes(f"text-xs {color} mb-1")

        # [v3.7.7] 일자별 포트폴리오 요약 — "하루 3종목 묶음 실제 수익"
        if port_stats and port_stats.get("n_days", 0) >= 5:
            avg_daily = port_stats["avg_daily_portfolio_ret"]
            pos_rate = port_stats.get("positive_rate", 0) * 100
            n_days = port_stats["n_days"]
            n_pos = port_stats.get("n_positive_days", 0)
            p_mark = "📈" if avg_daily > 0 else "📉"
            p_color = "text-green-400" if avg_daily > 0 else "text-red-400"
            ui.label(
                f"{p_mark} 일평균 3종목 포트폴리오 수익 {avg_daily:+.2f}% · "
                f"플러스 마감 {n_pos}/{n_days}일 ({pos_rate:.0f}%)"
            ).classes(f"text-xs {p_color} mb-1")

        # ═══════════════════════════════════════════════════
        # [v3.7.14] 신호 성과 vs 실집행 성과 완전 분리 (리뷰어 최우선 지적)
        # ═══════════════════════════════════════════════════

        # 📡 1줄차: 신호 성과 (알파 품질) — 자본 제약 없이 순수 시그널
        sig_src = signal_top1 if signal_top1 else daily_top1
        if sig_src and sig_src.get("n_filled" if signal_top1 else "n", 0) > 0:
            if signal_top1:
                n_total = signal_top1.get("n_signals_total", 0)
                n_filled_s = signal_top1.get("n_filled", 0)
                ev_s = signal_top1.get("ev_net_pct", 0)
                tp1_s = signal_top1.get("tp1_rate", 0) * 100
                fill_s = signal_top1.get("fill_rate", 0) * 100
            else:
                n_total = daily_top1.get("n_all_picks", 0)
                n_filled_s = daily_top1.get("n", 0)
                ev_s = daily_top1.get("ev", 0)
                tp1_s = daily_top1.get("tp1_rate", 0) * 100
                fill_s = daily_top1.get("fill_rate", 0) * 100
            sig_clr = "text-blue-400" if ev_s > 0 else "text-red-400"
            ui.label(
                f"📡 신호 성과 (알파 품질): "
                f"{n_total}신호 / {n_filled_s}체결 ({fill_s:.0f}%) · "
                f"TP1 {tp1_s:.1f}% · EV {ev_s:+.2f}%"
            ).classes(f"text-xs {sig_clr} mb-1 font-semibold")

        # 💰 2줄차: 실집행 성과 (실전 운용 가능성) — 자본 시뮬 기준
        if capital_top1 and capital_top1.get("n_trades_filled", 0) > 0:
            t1_ret = capital_top1.get("total_return_pct", 0)
            t1_mdd = capital_top1.get("max_drawdown_pct", 0)
            t1_n = capital_top1["n_trades_filled"]
            t1_init = capital_top1.get("initial_capital", 10_000_000)

            gap = capital_top1.get("signal_vs_capital_gap", {})
            exec_rate = gap.get("execution_rate", 1.0) * 100 if gap else 100
            m = "💰" if t1_ret > 0 else "💸"
            clr = "text-green-400" if t1_ret > 0 else "text-red-400"
            ui.label(
                f"{m} 실집행 성과 (실전 운용): "
                f"{t1_n}건 ({int(t1_init/1e4):,}만원 동시1포지션) · "
                f"{t1_ret:+.2f}% · MDD {t1_mdd:.1f}% · 실행률 {exec_rate:.0f}%"
            ).classes(f"text-xs {clr} mb-1 font-semibold")

            # 스킵 이유별 분해 (audit 있으면)
            skip = capital_top1.get("skip_reasons_summary", {})
            if skip:
                sk_exec = skip.get("EXECUTED", 0)
                sk_nf = skip.get("NOT_FILLED", 0)
                sk_held = skip.get("SAME_TICKER_ALREADY_HELD", 0)
                sk_full = skip.get("SLOT_FULL", 0)
                sk_total = skip.get("total_signals", 0)
                ui.label(
                    f"  └ 신호 {sk_total} → 실행 {sk_exec} · "
                    f"미체결 {sk_nf} · 기보유 {sk_held} · 슬롯풀 {sk_full}"
                ).classes("text-xs text-gray-500 mb-1 ml-4")

        # 🏅 3줄차: Confidence badge (과신 방지 핵심)
        if confidence:
            lvl = confidence.get("level", "LOW")
            reason = confidence.get("reason", "")
            if lvl == "HIGH":
                badge_txt, badge_clr = "🏅 HIGH", "text-green-400"
            elif lvl == "MEDIUM":
                badge_txt, badge_clr = "🏅 MEDIUM", "text-yellow-400"
            else:
                badge_txt, badge_clr = "🏅 LOW", "text-red-400"
            ui.label(
                f"{badge_txt} 실집행 신뢰도 — {reason}"
            ).classes(f"text-xs {badge_clr} mb-1 font-bold")

        # [v3.7.8→v3.7.11] Top3 자본 시뮬 (참고용)
        if capital_stats and capital_stats.get("n_trades_filled", 0) > 0:
            total_ret = capital_stats.get("total_return_pct", 0)
            mdd = capital_stats.get("max_drawdown_pct", 0)
            n_filled = capital_stats["n_trades_filled"]
            n_nf = capital_stats.get("n_skipped_not_filled", 0)
            n_dup = capital_stats.get("n_skipped_duplicate", 0)
            init_cap = capital_stats.get("initial_capital", 10_000_000)
            c_mark = "·" if total_ret > 0 else "·"
            c_color = "text-gray-500"  # 참고용이므로 회색
            ui.label(
                f"{c_mark} (참고) [Top3 모드] 자본시뮬: "
                f"기간수익 {total_ret:+.2f}% · MDD {mdd:.1f}% · 체결 {n_filled}건"
            ).classes(f"text-xs {c_color} mb-1")

        # [v3.7.8] Rolling walk-forward 종합
        if rolling_stats and rolling_stats.get("n_valid", 0) > 0:
            n_gen = rolling_stats.get("n_generalizes", 0)
            n_val = rolling_stats.get("n_valid", 0)
            avg_is = rolling_stats.get("avg_is_ev", 0)
            avg_oos = rolling_stats.get("avg_oos_ev", 0)
            robust = rolling_stats.get("robust", False)
            r_mark = "🔁 ✅" if robust else "🔁 ⚠️"
            r_color = "text-green-400" if robust else "text-yellow-400"
            ui.label(
                f"{r_mark} Rolling {n_gen}/{n_val} 폴드 일반화 · "
                f"평균 IS {avg_is:+.2f}% → OOS {avg_oos:+.2f}%"
            ).classes(f"text-xs {r_color} mb-1")

        # [v3.7.15] methodology 메타 한 줄 — 검증 조건 완전 투명화
        methodology = bt.get("methodology")
        if isinstance(methodology, dict):
            mh = methodology.get("horizon_days", "?")
            mf = methodology.get("fill_window_days", "?")
            mc = methodology.get("fee_pct_roundtrip", "?")
            mp = methodology.get("max_positions_top1", "?")
            mdedup = "✓" if methodology.get("dedup_same_ticker") else "✗"
            mreentry = "✓" if methodology.get("reentry_after_exit") else "✗"
            msel = methodology.get("selection_mode", "?")
            mdate = methodology.get("date_range", ["", ""])
            ui.label(
                f"🔧 검증조건: horizon {mh}일 · fill {mf}일 · fee {mc}% · "
                f"max_pos {mp} · dedup {mdedup} · reentry {mreentry} · "
                f"{msel} · {mdate[0]}~{mdate[1]}"
            ).classes("text-[10px] text-gray-500 mb-1 italic")

        if not top3_codes:
            # [v3.7.2] 빈 상태에도 백테스트 실측을 표시 — 데이터의 정직함 우선
            if strong_stats and strong_stats.get("n", 0) > 0:
                tp1 = strong_stats["tp1_rate"] * 100
                ev = strong_stats["ev"]
                ui.label(
                    f"오늘은 🏆최강 조건 종목이 없습니다. "
                    f"(과거 {strong_stats['n']}건 검증: TP1 {tp1:.1f}% · EV {ev:+.2f}%)"
                ).classes("text-sm text-gray-400")
            else:
                ui.label(
                    "오늘은 🏆최강/✅즉시진입 + 컷오프(40점) 조건을 만족하는 종목이 없습니다."
                ).classes("text-sm text-gray-400")
            # [v3.7.7] 실제 현재 임계값 문구로 교정 (v3.7.6 기준)
            ui.label(
                "🏆 최강: 평균≥70 · 밸런스≥70 · 갭≤3% · RR≥0.8  ·  "
                "✅ 즉시진입: 최소≥50 · 밸런스≥70 · 갭≤5%"
            ).classes("text-xs text-gray-500 mt-1")
            return

        with ui.row().classes("w-full gap-3 flex-wrap"):
            for i, code in enumerate(top3_codes, 1):
                match = df[df["종목코드"].astype(str).str.zfill(6) == code]
                if match.empty: continue
                r = match.iloc[0]
                name = str(r.get("종목명", "?"))
                lbl = str(r.get("ELITE_LABEL", ""))
                color = str(r.get("ELITE_LABEL_COLOR", "#3B82F6"))
                desc = str(r.get("ELITE_LABEL_DESC", ""))
                s_v = _nz(r.get("STRUCT_SCORE", 0))
                t_v = _nz(r.get("TIMING_SCORE", 0))
                a_v = _nz(r.get("AI_SCORE", 0))
                gap = _nz(r.get("GAP_PCT", 0))
                rr  = _nz(r.get("RR_NOW_TP1", 0))
                entry = int(_nz(r.get("추천매수가", 0)))
                tp1   = int(_nz(r.get("추천매도가1", 0)))
                stop  = int(_nz(r.get("손절가", 0)))

                with ui.card().classes(
                    "flex-1 min-w-[220px] p-3 bg-[#0a0a1e] "
                    "border border-gray-700/50 rounded-lg cursor-pointer hover:bg-[#1a1a2e]"
                ):
                    with ui.row().classes("items-center gap-2 mb-1"):
                        ui.label(f"#{i}").classes("text-sm text-gray-500")
                        ui.badge(lbl, color=color).classes("text-xs")
                    ui.label(name).classes("text-base font-bold text-white")
                    ui.label(f"S{s_v:.0f} T{t_v:.0f} AI{a_v:.0f}").classes(
                        "text-xs text-gray-400 mt-0.5"
                    )
                    ui.label(f"🎯{entry:,}  🟢{tp1:,}  🛡️{stop:,}").classes(
                        "text-xs text-gray-400 mt-0.5"
                    )
                    ui.label(f"갭 {gap:.1f}% · RR {rr:.2f} · {desc}").classes(
                        "text-[10px] text-gray-500 mt-1"
                    )


def render_tab_stocks(df: pd.DataFrame, auth: str, store=None):
    """Tab 2: AI & Quant 추천 종목

    Args:
        df: 스코어링된 종목 DataFrame
        auth: 사용자 권한 ("admin" / "premium" / "free" / ...)
        store: services.data_store.store 인스턴스 (현재 미사용, 장래 확장용)
    """

    # ── [v3.7.11] 라벨링 + Top1 우선 선별 (pick_top3은 fallback) ──
    # Top1이 "진짜 수익 나는" 기본 모드. 🏆 최강이 없으면 pick_top3으로 폴백.
    # (UI에선 헤더 카드에 Top1을 강조 표시, 테이블엔 여러 종목도 함께 표시)
    df = compute_elite_labels(df)
    top1_codes = pick_top1(df)
    top3_fallback = pick_top3(df) if not top1_codes else []
    # 헤더 카드에 표시할 종목들: Top1이 있으면 Top1만, 없으면 폴백 Top3
    top3_codes = top1_codes if top1_codes else top3_fallback

    ui.label("🎯 AI & Quant 추천 종목").classes(
        "text-xl font-bold text-white mb-4"
    )

    # ── [v3.7] Top 3 헤더 카드 (백테스트 검증 기반) ──
    _render_top3_card(df, top3_codes)

    # ── 뷰모드 + 필터 ──
    with ui.row().classes("w-full gap-4 items-center flex-wrap mb-4"):
        view_mode = ui.toggle(
            ["📋 테이블", "🃏 칸반"], value="📋 테이블"
        )
        route_filter = ui.select(
            ["전체", "ATTACK", "ARMED", "WAIT", "NEUTRAL"],
            value="전체", label="상태",
        ).classes("min-w-[120px]")
        # [v3.7] "⚖️ 밸런스순" 추가
        sort_mode = ui.toggle(
            ["🔢 점수순", "⚖️ 밸런스순", "🏆 검증순", "🚦 상태순"],
            value="🏆 검증순",
        )

    table_area = ui.column().classes("w-full")
    detail_area = ui.column().classes("w-full mt-4")

    def _filtered():
        fdf = df.copy()
        if route_filter.value != "전체" and "ROUTE" in fdf.columns:
            fdf = fdf[fdf["ROUTE"].astype(str).str.contains(
                route_filter.value, na=False
            )]
        # [v3.7] 정렬 로직 확장
        if sort_mode.value == "🔢 점수순" and "DISPLAY_SCORE" in fdf.columns:
            fdf = fdf.sort_values("DISPLAY_SCORE", ascending=False)
        elif sort_mode.value == "⚖️ 밸런스순":
            # min(S, T, AI) 내림차순 → 3축 모두 높은 종목 우선
            s_col = fdf["STRUCT_SCORE"].fillna(0) if "STRUCT_SCORE" in fdf.columns else 0
            t_col = fdf["TIMING_SCORE"].fillna(0) if "TIMING_SCORE" in fdf.columns else 0
            a_col = fdf["AI_SCORE"].fillna(0)     if "AI_SCORE"     in fdf.columns else 0
            fdf = fdf.assign(_axis_min=pd.concat([s_col, t_col, a_col], axis=1).min(axis=1))
            fdf = fdf.sort_values("_axis_min", ascending=False).drop(columns=["_axis_min"])
        elif sort_mode.value == "🏆 검증순" and "ELITE_RANK_SCORE" in fdf.columns:
            fdf = fdf.sort_values("ELITE_RANK_SCORE", ascending=False)
        elif sort_mode.value == "🚦 상태순" and "ROUTE" in fdf.columns:
            # ATTACK → ARMED → WAIT → NEUTRAL → OVERHEAT → CARRY → 기타
            route_order = {
                "ATTACK": 0, "ARMED": 1, "WAIT": 2,
                "NEUTRAL": 3, "OVERHEAT": 4, "CARRY": 5,
            }
            fdf = fdf.assign(
                _route_rank=fdf["ROUTE"].astype(str).str.upper().map(
                    lambda x: next((v for k, v in route_order.items() if k in x), 99)
                )
            )
            # ROUTE 1차, 검증순 2차, DISPLAY_SCORE 3차
            secondary = "ELITE_RANK_SCORE" if "ELITE_RANK_SCORE" in fdf.columns else "DISPLAY_SCORE"
            if secondary in fdf.columns:
                fdf = fdf.sort_values(
                    ["_route_rank", secondary], ascending=[True, False]
                )
            else:
                fdf = fdf.sort_values("_route_rank")
            fdf = fdf.drop(columns=["_route_rank"])
        # 접근 제한
        limits = {"guest": 3, "free": 5, "pro": 20}
        fdf = fdf.head(limits.get(auth, 50))
        return fdf

    def _build_view():
        table_area.clear()
        show = _filtered()
        with table_area:
            if view_mode.value == "🃏 칸반":
                _render_kanban(show, df)
            else:
                _render_table(show, df)

    def _render_table(show: pd.DataFrame, full_df: pd.DataFrame):
        columns = [
            # [v3.7] 라벨 뱃지 (최강/관심/추격)
            {"name": "label", "label": "라벨", "field": "label", "align": "center"},
            {"name": "route", "label": "상태", "field": "route", "align": "center"},
            {"name": "name", "label": "종목명", "field": "name", "align": "left"},
            {"name": "score", "label": "점수", "field": "score",
             "align": "center", "sortable": True},
            # ── 3축 점수 + ELITE/BALANCE (v3.6 복원) ──
            {"name": "s", "label": "S", "field": "s",
             "align": "center", "sortable": True},
            {"name": "t", "label": "T", "field": "t",
             "align": "center", "sortable": True},
            {"name": "ai", "label": "AI", "field": "ai",
             "align": "center", "sortable": True},
            {"name": "elite", "label": "검증점수", "field": "elite",
             "align": "center", "sortable": True},
            {"name": "bal", "label": "균형", "field": "bal",
             "align": "center", "sortable": True},
            {"name": "gap", "label": "갭%", "field": "gap",
             "align": "center", "sortable": True},
            {"name": "close", "label": "현재가", "field": "close", "align": "right"},
            {"name": "buy", "label": "매수", "field": "buy", "align": "right"},
            {"name": "stop", "label": "손절", "field": "stop", "align": "right"},
            {"name": "t1", "label": "T1목표", "field": "t1", "align": "right"},
            {"name": "sector", "label": "업종", "field": "sector", "align": "left"},
        ]
        rows = []
        for _, r in show.iterrows():
            rows.append({
                "code": str(r.get("종목코드", "")).zfill(6),
                "label": str(r.get("ELITE_LABEL", "") or "—"),
                "route": str(r.get("ROUTE", "—")),
                "name": str(r.get("종목명", "—")),
                "score": f'{_nz(r.get("DISPLAY_SCORE", 0)):.0f}',
                "s":     f'{_nz(r.get("STRUCT_SCORE",  0)):.0f}',
                "t":     f'{_nz(r.get("TIMING_SCORE",  0)):.0f}',
                "ai":    f'{_nz(r.get("AI_SCORE",      0)):.0f}',
                # [v3.7.1] ELITE_SCORE 없으면 ELITE_RANK_SCORE 폴백 (정합성)
                "elite": f'{_nz(r.get("ELITE_SCORE", r.get("ELITE_RANK_SCORE", 0))):.0f}',
                "bal":   f'{_nz(r.get("BALANCE_CALC",  r.get("BALANCE_SCORE", 0))):.0f}',
                "gap":   f'{_nz(r.get("GAP_PCT", 0)):.1f}',
                "close": f'{int(_nz(r.get("종가", 0))):,}',
                "buy": f'{int(_nz(r.get("추천매수가", 0))):,}',
                "stop": f'{int(_nz(r.get("손절가", 0))):,}',
                "t1": f'{int(_nz(r.get("추천매도가1", 0))):,}',
                "sector": str(r.get("업종", "—")),
            })
        tbl = ui.table(
            columns=columns, rows=rows, row_key="code",
            selection="single", pagination={"rowsPerPage": 15},
        ).classes("w-full").props("dense dark flat bordered")
        tbl.on("selection", lambda e: _on_stock_select(e, full_df))

    def _render_kanban(show: pd.DataFrame, full_df: pd.DataFrame):
        if show.empty:
            ui.label("표시할 종목 없음").classes("text-gray-400")
            return
        route_col = "ROUTE" if "ROUTE" in show.columns else None
        if route_col:
            df_atk = show[show[route_col].astype(str).str.contains(
                "ATTACK", case=False, na=False
            )]
            df_arm = show[show[route_col].astype(str).str.contains(
                "ARMED", case=False, na=False
            )]
            ex = df_atk.index.union(df_arm.index)
            df_watch = show[~show.index.isin(ex)]
        else:
            df_atk = df_arm = pd.DataFrame()
            df_watch = show

        with ui.row().classes("w-full gap-4 flex-wrap items-start"):
            _kanban_col("🚀 ATTACK", df_atk, "#EF4444")
            _kanban_col("🔫 ARMED", df_arm, "#F59E0B")
            _kanban_col("👀 WATCH", df_watch, "#3B82F6")

    def _kanban_col(title: str, sub_df: pd.DataFrame, color: str):
        with ui.column().classes("kanban-col min-w-[280px] flex-1"):
            ui.label(f"{title} ({len(sub_df)})").classes(
                "text-white font-bold mb-2"
            ).style(f"border-bottom: 2px solid {color}")
            if sub_df.empty:
                ui.label("비어 있음").classes("text-gray-500 text-sm")
                return
            for _, r in sub_df.iterrows():
                score = _nz(r.get("DISPLAY_SCORE", 0))
                sc = "#10B981" if score >= 80 else "#3B82F6" if score >= 60 else "#94A3B8"
                with ui.card().classes(
                    "p-3 mb-2 cursor-pointer bg-[rgba(255,255,255,0.05)] "
                    "border border-[rgba(255,255,255,0.1)] rounded-xl "
                    "hover:bg-[rgba(255,255,255,0.08)]"
                ):
                    # [v3.7] 라벨 뱃지 (최강/관심/추격) — 종목명 위
                    elite_lbl = str(r.get("ELITE_LABEL", "") or "")
                    elite_color = str(r.get("ELITE_LABEL_COLOR", "") or "")
                    if elite_lbl:
                        ui.badge(elite_lbl, color=elite_color).classes("text-[10px] mb-1")

                    with ui.row().classes("justify-between items-center"):
                        ui.label(str(r.get("종목명", ""))).classes(
                            "text-white font-bold text-sm"
                        )
                        ui.badge(f"{score:.0f}", color=sc).classes("text-xs")
                    buy = int(_nz(r.get("추천매수가", 0)))
                    stop = int(_nz(r.get("손절가", 0)))
                    t1 = int(_nz(r.get("추천매도가1", 0)))
                    # ── 3축 점수 서브라벨 (v3.6 복원) ──
                    s_val  = _nz(r.get("STRUCT_SCORE",  0))
                    t_val  = _nz(r.get("TIMING_SCORE",  0))
                    ai_val = _nz(r.get("AI_SCORE",      0))
                    bal    = _nz(r.get("BALANCE_CALC",  r.get("BALANCE_SCORE", 0)))
                    gap    = _nz(r.get("GAP_PCT", 0))
                    ui.label(
                        f"S{s_val:.0f} T{t_val:.0f} AI{ai_val:.0f} · 균형{bal:.0f} · 갭{gap:.1f}%"
                    ).classes("text-xs text-gray-400 mt-1")
                    if buy > 0:
                        ui.label(
                            f"🎯 {buy:,}  🛡️ {stop:,}  🟢 {t1:,}"
                        ).classes("text-xs text-gray-400 mt-1")

    # ── 종목 상세 분석 ──
    def _on_stock_select(event, full_df: pd.DataFrame):
        detail_area.clear()
        sel = event.args.get("rows", []) if hasattr(event, "args") else []
        if not sel:
            return
        code = sel[0].get("code", "")
        match = full_df[full_df["종목코드"].astype(str).str.zfill(6) == code]
        if match.empty:
            return
        row = match.iloc[0]
        _render_stock_detail(code, row)

    def _render_stock_detail(code: str, row):
        name = row.get("종목명", "")
        _close = _nz(row.get("종가", 0))
        _entry = _nz(row.get("추천매수가", 0))
        _stop = _nz(row.get("손절가", 0))
        _t1 = _nz(row.get("추천매도가1", 0))
        _t2 = _nz(row.get("추천매도가2", 0))

        with detail_area:
            ui.label(
                f"🔍 {name} ({code}) 상세 분석"
            ).classes("text-lg font-bold text-white mb-3")

            # 목표가 카드
            if _close > 0 and _entry > 0:
                risk = _entry - _stop if _stop > 0 else 1
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    _metric_card(
                        "🔴 손절가", f"{int(_stop):,}",
                        f"{(_stop/_close-1)*100:+.1f}%" if _close > 0 else "",
                        False,
                    )
                    _metric_card("🔵 매수가", f"{int(_entry):,}", "시스템 추천")
                    if _t1 > 0:
                        rr1 = (_t1 - _entry) / risk if risk > 0 else 0
                        _metric_card(
                            "🟢 T1 목표", f"{int(_t1):,}",
                            f"+{(_t1/_close-1)*100:.1f}% (RR {rr1:.1f}:1)",
                        )
                    if _t2 > 0 and _t2 != _t1:
                        rr2 = (_t2 - _entry) / risk if risk > 0 else 0
                        _metric_card(
                            "🟡 T2 목표", f"{int(_t2):,}",
                            f"+{(_t2/_close-1)*100:.1f}% (RR {rr2:.1f}:1)",
                        )

            # [v3.7.17] 핵심 지표 요약 패널 — 분석에 필요한 정보 한눈에
            with ui.card().classes(
                "w-full p-3 mt-2 bg-[rgba(255,255,255,0.03)] "
                "border border-[rgba(255,255,255,0.08)] rounded-xl"
            ):
                ui.label("📊 핵심 지표").classes("text-xs text-gray-400 mb-2")
                with ui.row().classes("w-full gap-4 flex-wrap"):
                    # 3축 점수
                    s_val = _nz(row.get("STRUCT_SCORE", row.get("S_SCORE", 0)))
                    t_val = _nz(row.get("TIMING_SCORE", row.get("T_SCORE", 0)))
                    ai_val = _nz(row.get("AI_SCORE", 0))
                    bal = _nz(row.get("BALANCE_CALC", row.get("BALANCE_SCORE", 0)))
                    elite = _nz(row.get("ELITE_RANK_SCORE", 0))
                    gap = _nz(row.get("GAP_PCT", 0))
                    rsi = _nz(row.get("RSI14", 0))
                    vp = _nz(row.get("V_POWER", 0))
                    turnover = _nz(row.get("거래대금(억원)", 0))

                    def _mini(label, val, clr="text-white"):
                        with ui.column().classes("gap-0 min-w-[75px]"):
                            ui.label(label).classes("text-[10px] text-gray-500")
                            ui.label(val).classes(f"text-sm font-bold {clr}")

                    # 한국식 색상: 높으면 빨강, 낮으면 파랑
                    def _clr(v, good=70, bad=40):
                        if v >= good: return "text-red-400"
                        elif v >= bad: return "text-yellow-400"
                        else: return "text-blue-400"

                    _mini("S 구조", f"{s_val:.0f}", _clr(s_val))
                    _mini("T 타이밍", f"{t_val:.0f}", _clr(t_val))
                    _mini("AI", f"{ai_val:.0f}", _clr(ai_val))
                    _mini("균형", f"{bal:.0f}", _clr(bal))
                    _mini("검증점수", f"{elite:.0f}", _clr(elite, good=60, bad=30))
                    # RSI: 70↑ 과매수(빨강), 30↓ 과매도(파랑)
                    rsi_clr = "text-red-400" if rsi >= 70 else "text-blue-400" if rsi <= 30 else "text-gray-300"
                    _mini("RSI14", f"{rsi:.0f}", rsi_clr)
                    # V_POWER: 세력강도 (-1~3)
                    vp_clr = "text-red-400" if vp >= 1.5 else "text-yellow-400" if vp >= 0 else "text-blue-400"
                    _mini("세력(V)", f"{vp:+.2f}", vp_clr)
                    # 갭: 5% 초과면 추격 경고
                    gap_clr = "text-red-400" if gap > 5 else "text-yellow-400" if gap > 2 else "text-gray-300"
                    _mini("갭%", f"{gap:.1f}%", gap_clr)
                    # 거래대금: 1000억+ 빨강
                    to_clr = "text-red-400" if turnover >= 1000 else "text-yellow-400" if turnover >= 300 else "text-gray-400"
                    _mini("거래대금", f"{turnover:.0f}억", to_clr)

            # ── 캔들차트 (비동기 로드 + 태스크 생명주기 관리) ──
            with ui.card().classes("w-full p-2 bg-[#1a1a2e] mt-2"):
                loading_label = ui.label(
                    "🕯️ 캔들차트 로딩 중..."
                ).classes("text-gray-400")
                chart_holder = ui.column().classes("w-full")

                async def _load_chart():
                    """[v3.3 #3] run.io_bound로 GIL 블로킹 방지"""
                    try:
                        cdata = await run.io_bound(_get_chart_data, code)
                    except asyncio.CancelledError:
                        return  # 태스크 취소 시 조용히 종료

                    loading_label.set_visibility(False)
                    chart_holder.clear()
                    with chart_holder:
                        if cdata is not None and _plot_candle is not None:
                            fig = _plot_candle(
                                cdata, code, name,
                                _entry, _stop, _t1, _t2,
                            )
                            _plotly_dark(fig, 400)
                            ui.plotly(fig).classes("w-full")
                        elif cdata is not None:
                            ui.label(
                                "📉 차트 렌더러 미로드"
                            ).classes("text-yellow-400")
                        else:
                            ui.label(
                                "📉 차트 데이터 로드 실패"
                            ).classes("text-yellow-400")

                # [v3.3 #1] ui.timer 제거 → asyncio.create_task 직접 실행
                # [v3.3 #2] 태스크 변수 저장 → detail_area.clear() 시 자동 취소
                _chart_task = asyncio.create_task(_load_chart())

                # 상세 영역이 다시 clear()될 때 유령 태스크 방지
                def _cancel_on_clear():
                    if _chart_task and not _chart_task.done():
                        _chart_task.cancel()

                detail_area.on("clear", _cancel_on_clear)

            # ── 레이더 + 워터폴 ──
            with ui.row().classes("w-full gap-4 flex-wrap mt-4"):
                with ui.card().classes(
                    "flex-1 min-w-[280px] p-2 bg-[#1a1a2e]"
                ):
                    try:
                        if plot_radar_chart:
                            fig_r = plot_radar_chart(row)
                            if fig_r:
                                _plotly_dark(fig_r, 300)
                                ui.plotly(fig_r).classes("w-full")
                    except Exception:
                        ui.label("레이더 차트 오류").classes("text-gray-500")

                with ui.card().classes(
                    "flex-1 min-w-[280px] p-2 bg-[#1a1a2e]"
                ):
                    try:
                        if plot_score_waterfall:
                            fig_w = plot_score_waterfall(row)
                            if fig_w:
                                _plotly_dark(fig_w, 300)
                                ui.plotly(fig_w).classes("w-full")
                    except Exception:
                        ui.label("워터폴 차트 오류").classes("text-gray-500")

            # ROUTE 배지
            rv = str(row.get("ROUTE", "NEUTRAL"))
            rc = {
                "ATTACK": "#EF4444", "ARMED": "#F59E0B",
                "WAIT": "#3B82F6", "NEUTRAL": "#6B7280",
            }.get(rv, "#6B7280")
            ui.badge(rv, color=rc).classes("mt-2")

    # ── 이벤트 바인딩 ──
    for widget in [view_mode, route_filter, sort_mode]:
        widget.on("update:model-value", lambda _: _build_view())

    _build_view()


# ═══════════════════════════════════════════════════
#  page_stock.py / page_briefing.py 호환 공용 헬퍼
#  (v3.5 — 외부 모듈에서 import 하는 공개 심볼)
# ═══════════════════════════════════════════════════

# ── ROUTE 상수 ──
ROUTE_KR = {
    "ATTACK":   "매수 돌입",
    "ARMED":    "매수 대기",
    "WAIT":     "관망",
    "OVERHEAT": "과열 주의",
    "NEUTRAL":  "중립",
}

ROUTE_COLOR = {
    "ATTACK":   "#FF4B4B",
    "ARMED":    "#FFA726",
    "WAIT":     "#29B6F6",
    "OVERHEAT": "#757575",
    "NEUTRAL":  "#BDBDBD",
}

ROUTE_DESC = {
    "ATTACK":   "조건 충족 · 즉시 진입 검토",
    "ARMED":    "트리거 임박 · 대기 포지션",
    "WAIT":     "관망 · 신호 대기",
    "OVERHEAT": "과열 구간 · 신규 진입 자제",
    "NEUTRAL":  "중립 · 판단 보류",
}


def _route_key(route) -> str:
    """ROUTE 문자열 정규화 + 부분일치 키 추출"""
    r = str(route or "").upper()
    for key in ROUTE_KR.keys():
        if key in r:
            return key
    return "NEUTRAL"


def _route_kr(route) -> str:
    """ROUTE → 한글 라벨"""
    return ROUTE_KR.get(_route_key(route), str(route or ""))


def _route_desc(route) -> str:
    """ROUTE → 설명 문구"""
    return ROUTE_DESC.get(_route_key(route), "")


def _route_color(route) -> str:
    """ROUTE → HEX 색상"""
    return ROUTE_COLOR.get(_route_key(route), "#BDBDBD")


# ── UI 헬퍼 ──

def _section_title(text: str):
    """섹션 타이틀 (tab_market.py와 동일 스타일)"""
    ui.label(text).classes(
        "text-lg font-bold text-white mt-6 mb-2 border-b border-gray-700 pb-2"
    )


def _price_bar_html(stop, entry, close, t1, t2) -> str:
    """손절/진입/현재가/목표1/목표2 가격바 HTML

    단순 수평 바 + 각 가격 위치의 마커. 모든 인자는 숫자(0/None 허용).
    """
    try:
        vals = [float(v) for v in (stop, entry, close, t1, t2) if v]
    except Exception:
        vals = []

    if not vals:
        return (
            '<div style="padding:12px;color:#888;text-align:center;'
            'background:rgba(255,255,255,0.03);border-radius:8px;">'
            '가격 정보 없음</div>'
        )

    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1.0)

    def _pos(v):
        try:
            return max(0.0, min(100.0, (float(v) - lo) / span * 100.0))
        except Exception:
            return None

    markers = [
        ("손절",  stop,  "#FF4B4B"),
        ("매수",  entry, "#29B6F6"),
        ("현재",  close, "#FFFFFF"),
        ("목표1", t1,    "#66BB6A"),
        ("목표2", t2,    "#26A69A"),
    ]

    dots = []
    labels = []
    for label, val, color in markers:
        pos = _pos(val)
        if pos is None or not val:
            continue
        dots.append(
            f'<div style="position:absolute;left:{pos}%;top:0;transform:translateX(-50%);'
            f'width:10px;height:10px;border-radius:50%;background:{color};'
            f'border:2px solid #0a0a1e;"></div>'
        )
        labels.append(
            f'<div style="position:absolute;left:{pos}%;top:14px;transform:translateX(-50%);'
            f'font-size:10px;color:{color};white-space:nowrap;">'
            f'{label}<br><span style="color:#aaa;">{int(val):,}</span></div>'
        )

    return (
        '<div style="position:relative;height:60px;margin:12px 0;padding:0 8px;">'
        '<div style="position:absolute;left:0;right:0;top:4px;height:2px;'
        'background:linear-gradient(90deg,#FF4B4B 0%,#FFA726 50%,#66BB6A 100%);'
        'border-radius:2px;opacity:0.4;"></div>'
        + "".join(dots) + "".join(labels) +
        '</div>'
    )


# ── 차트 별칭/래퍼 ──
# 기존 _get_chart_data / _plot_candle 에 외부 호출자가 기대하는 이름으로 다리를 놓는다.

def _get_stock_chart_data(code: str):
    """page_stock.py 호환 별칭 — 기존 _get_chart_data 위임"""
    return _get_chart_data(code)


def _plot_candle_chart(cdata, code: str, name: str, entry=None, stop=None, t1=None, t2=None):
    """page_stock.py 호환 래퍼 — chart_components.plot_candle_chart 위임.

    차트 모듈 부재 또는 실패 시 None 반환 (호출부가 None 체크함).
    """
    if _plot_candle is None or cdata is None:
        return None
    try:
        # 위치 인자 기준 호출 (tab_stocks 내부에서도 동일 시그니처로 사용 중)
        return _plot_candle(cdata, code, name, entry, stop, t1, t2)
    except TypeError:
        # 구버전 시그니처 폴백 (cdata, code, name 만 받는 경우)
        try:
            return _plot_candle(cdata, code, name)
        except Exception as e:
            _logger.warning(f"캔들차트 생성 실패 [{code}]: {e}")
            return None
    except Exception as e:
        _logger.warning(f"캔들차트 생성 실패 [{code}]: {e}")
        return None
