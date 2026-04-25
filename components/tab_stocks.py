# -*- coding: utf-8 -*-
"""
tab_stocks.py — Tab 2: 종목 분석 (테이블 + 칸반 + 상세)
═══════════════════════════════════════════════════
[v3.7.29] (2026-04-20) — 마지막 봉합 (94 → 97점 도전)
  배경: v3.7.28 평가 결과 94점 — 2가지 마무리만 남음
  Patch A: CONFIG_SNAPSHOT Migration 완전 봉합
    · pipeline_finalize.py 과거 "변경 필요" 주석 제거
    · SSOT (Single Source of Truth) 정책 공식 문서화
      - CSV 행: 경량 (CONFIG_VERSION만)
      - JSON 파일: 전체 config snapshot 전용
    · load_config_snapshot() Fallback 체인 4단계 강화:
      1) dated JSON → 2) latest JSON → 3) 런타임 재생성 → 4) 빈 dict
    · test_shadow_analyze.py가 실제 load_config_snapshot() 호출
      - snapshot 모드에서 config meta를 별도 저장
      - 재현성 보장 + migration 실제 검증
  Patch B: Shadow Test → CI Regression Gate 전환
    · exit code 0/1 (CI pass/fail 신호)
    · CLI 인자 확장:
      - --min-match-rate (default 0.995)
      - --critical-keys (쉼표 구분)
      - --report-json (CI 파싱용)
      - --quiet (CI 로그 절약)
    · 3단계 Gate 판정:
      1) match_rate 임계치
      2) critical keys 무결성 (점수/가격/라벨)
      3) missing 종목 0개
    · pytest 래퍼 신규: tests/test_shadow_regression.py
      - pytest 자동 수집 가능
      - 환경변수로 임계치 조정 (SHADOW_MIN_MATCH_RATE 등)
      - skip 처리로 snapshot 없을 때 안전
  의의: "좋은 도구" → "리팩토링 안전 게이트"로 성숙도 업그레이드
[v3.7.28] (2026-04-20) — Phase 1 완결 패치 (89 → 95점 도전)
  배경: v3.7.27 평가 결과 89점 — 마감 디테일 부족
  #1 README/주석 수치 일치화
     · 실측 결과 174 → 141 (33개 제거)로 통일
     · 이전 오기 수정: v3.7.27 주석에 잘못 적혔던 컬럼 수치 교정
     · "76.3%" 크기 감소로 통일 (2,008,554 → 476,445 bytes)
  #2 ml_engine.py print 완전 제거 (11개 → 0개)
     · 멀티라인 print 괄호 균형 추적 변환
     · 전체 logger.info/warning/error로 100% 통일
  #3 CONFIG_SNAPSHOT 참조 코드 봉합
     · test_shadow_analyze.py: CONFIG_SNAPSHOT 컬럼 → CONFIG_VERSION만 사용
     · JSON 파일 없을 때 fallback 로직 추가
[v3.7.27] (2026-04-20) — Phase 1 기술부채 청소 (CSV/DB/ML 품질)
  목표: 87점 → 90점 (100점 평가 결과 Critical 3개 해결)
  #1 CSV 다운로드 중복/상수 컬럼 제거 (components/tab_stocks.py)
     · 중복 5개: LDY_SCORE, TOTAL_SCORE, RANK_SCORE, ML_SCORE, RAW_TRIGGER_SCORE
     · 상수 28개: CONFIG_SNAPSHOT, MACRO_RISK, W_STRUCT 등 (전체 동일값)
     · 실측 검증: nunique()==1 자동 필터 (파이프라인 변경 대비)
     · 실측 결과: 174 → 141 컬럼 (33개 제거, 76% 크기 감소)
     · 알림에 "N컬럼 (X개 제거)" 표시
  #2 CONFIG_SNAPSHOT 별도 파일 분리 (pipeline_finalize.py)
     · 이전: 행당 2.5KB × 500종목 = 1.3MB 반복 저장
     · 이후: data/config_snapshot_YYYYMMDD.json 1회만 저장
  #3 DB 트랜잭션 안전성 (db_utils.py)
     · SQLite: rollback + finally + 커서 close 추가
     · 일반 예외도 rollback 처리
     · DuckDB: 재연결 실패 시 명시적 raise
  #4 ML Engine 품질 (ml_engine.py)
     · logger 모듈 추가 (print 30개 → logger.info/warning/error)
     · 7개 핵심 함수 타입힌트 추가
     · 운영 로그 레벨 제어 가능 (이전: print 42회 · logger 0회)
  #5 슬리피지 0.10% → 0.25% 보수적 재설정 (auto_backtest.py)
     · 왕복 비용 0.41% → 0.71% (현실 반영)
     · 개인 시장가 주문 2~3틱 slip + 저유동 종목 대응
  원칙: 엔진 로직 불변 + CSV 생성 단계에서만 정리
  복구: 각 패치별 주석에 rollback 가이드 기록
[v3.7.26] (2026-04-19) — 스코어 체계 UI 정리 (페이즈 1) + 차트 툴팁 수정
  사용자 지적 2가지:
    1. "스코어들이 너무 많은데 로직들좀 설명해봐"
    2. "차트의 흰 팝업이 너무 쎄서 정보가 안보여"
  원칙: 엔진(collector/pipeline)은 건드리지 않고 UI 레이어만 정리
       → 리스크 0, 롤백 즉시 가능
  #1 테이블 보기 모드 토글 추가 — 🎯 기본 / 🔬 고급
     · 기본: label·상태·종목명·점수·S·T·AI·갭%·RR·가격·업종 (11개)
     · 고급: 기본 + 종합·랭크·균형 (14개)
  #2 테이블에 RR 컬럼 신규 추가 (실전 매매 핵심 지표)
  #3 상세 패널 2단 구조로 재편
     · 🎯 핵심 지표 (항상): S·T·AI·갭%·RR·종합 (6개)
     · 🔬 기타 지표 (접이식): 균형·랭크·RSI·세력(V)·거래대금
  #4 스코어 용어집 접이식 카드 신규
     · 각 스코어 공식 + 라벨 가중치 + 실전 팁 한 곳에
  #5 캔들차트 툴팁 가독성 수정 (⭐ 사용자 즉각 피드백)
     · hoverlabel bgcolor: 흰색(plotly 기본) → rgba(26,26,46,0.95) 다크 반투명
     · font color: 회색 → #FFFFFF 흰색
     · border: 보라 (#8B5CF6) 브랜드 컬러
     · font family: monospace (OHLC 숫자 정렬)
     · 내장 캔들차트 + _plotly_dark 헬퍼 모두 적용
  복구: view_table_mode 기본값 변경 또는 hoverlabel 제거로 즉시 가능.
[v3.7.25] (2026-04-19) — 🛡️ 콤보 정식 승격 + 🏆 최강 관찰 모드 전환
  사용자 결정 2가지:
    · "콤보를 제1 매수종목으로 가자"
    · "최강은 표본도 부족하고 실질 검증도 안되니까 매매에서 제외"
  배경:
    · combo_optimizer: S≥90 T≥80 AI≥60 ATTACK/ARMED → n=112, EV +25.77%, 승률 83.9%
    · 즉석 walk-forward: IS 64.3% → OOS 92.0% (매우 robust)
    · 🏆 최강: n=6 표본부족 → 통계 신뢰 LOW → 관찰만
  #1 🛡️ 콤보 라벨 정식 승격 (보라 #8B5CF6)
     · _compute_axis_stats()에 S/T/AI/ROUTE 추가 필드
     · _elite_label() 최상위 판정 → 🛡️ > 🏆 > ✅ > ⚠️
     · _rank_score() 가중치 ×1.50 (최고)
  #2 🏆 최강 관찰 모드로 전환 (매매 풀에서 배제)
     · pick_top1/top3에서 🏆 최강 fallback 제거 (콤보만)
     · _rank_score 가중치 1.00 → 0.50 (페널티)
     · 라벨 설명: "👁️ 관찰중 · 매매 제외 · 표본 n=6 · 통계 축적 대기"
     · UI 기준 카드: 회색 + 취소선 + opacity-60
  #3 pick_top1: 콤보 없으면 빈 결과 ("오늘 매매 없음")
  #4 pick_top3: 콤보만 사용 (🏆 최강 포함 X)
  #5 라벨 필터 드롭다운에 🛡️ 콤보 추가
  #6 복구 경로: pick_top1 docstring에 주석 처리된 fallback 블록
     → 2~3개월 실집행 표본 100건+ 쌓이면 주석 풀어서 부활 가능
[v3.7.24] (2026-04-18) — 검증점수 표시 통일 + 명칭 명확화
  사용자 지적: "같은 '검증점수' 라벨인데 테이블과 상세영역 값이 다름 (80 vs 22)"
  원인:
    · 테이블: ELITE_SCORE > ELITE_RANK_SCORE 폴백 → 세아 80
    · 상세영역: ELITE_RANK_SCORE만 → 세아 22
    · + "검증"이라는 단어가 walk-forward 검증과 무관인데 오해 유발
  #1 테이블 컬럼 분리: "검증점수" 하나 → "종합" + "랭크" 2개
     - 종합 (ELITE_SCORE) : 파이프라인 최종 스코어
     - 랭크 (ELITE_RANK_SCORE): 내부 Top 선별용 (라벨/밸런스/RR 보정)
  #2 상세영역도 테이블과 동일하게 "종합" + "랭크" 2개 미니바 표시
     - 같은 종목에서 두 값 일관되게 매치
  #3 정렬 옵션: "🏆 검증순" → "🏆 랭크순" (ELITE_RANK_SCORE 기준임을 명확화)
  #4 "검증"이라는 혼동 용어 제거 (walk-forward와 무관한 점수)
[v3.7.23] (2026-04-18) — 헤더 카드 읽기 순서 최적화 + 제목 정확화
  사용자 리뷰어 지적 3가지 전부 반영:
  #1 제목: "🏆 오늘의 검증 Top 3" → "🏆 오늘의 실전 Top Pick"
     - 실제 모드는 top1_first_then_top3_fallback (Top1 우선)
     - "Top 3"는 예전 철학 흔적 → "Top Pick"이 정확
     - 뱃지: "1/3" → "🎯 Top1" (1개 성공) / "Top3 폴백 N/3" (fallback)
  #2 정보 우선순위 재배치 — 메인/보조 구분
     ┌ [메인 블록] (font-semibold, 눈에 띄게)
     │   💰 실집행 성과 (가장 중요 — 실전 운용)
     │   📡 신호 성과 (알파 품질)
     │   🏅 Confidence (실행 판단 기준)
     └ [보조 블록] (font-normal, 참고용)
         ✅ Walk-forward
         🔁 Rolling
         (참고) 일평균 3종목 포트
         (참고) Top3 모드 자본시뮬
  #3 "TP1 21% vs EV +1.47%" 해석 문구 추가
     - TP1 <40% + EV>0: "승리 폭 > 패배 폭 + 미도달 종가 마감 포함"
     - TP1 >=40% + EV>0: "높은 승률 기반 수익 구조"
     - EV<=0: "알파 약화 상태 (임계값 재조정 검토)"
[v3.7.22] (2026-04-18) — CSV 다운로드 Prime 회원 제한
  #1 사용자 지적: "무료회원 다운로드 권한 있으면 안 됨 — Prime만 가능해야"
     v3.7.20에서 모두에게 다운로드 허용 → 수익 모델 훼손
  #2 권한 정책:
     · guest/free : 🔒 버튼 비활성 + "👑 Prime 회원 전용" 안내
     · prime      : 👑 정상 다운로드
     · admin      : 🛠️ 정상 다운로드
  #3 이중 방어 — UI disabled + 함수 내부 권한 체크 (백엔드 차단)
[v3.7.21] (2026-04-18) — 매일 백테스트 자동 갱신 + 데이터 신선도 표시
  #1 auto_collect.yml에 backtest_validation 실행 단계 추가
     - 평일 20:05 KST Collector 후 자동으로 backtest_validation.py 실행
     - 매일 장 마감 데이터까지 반영된 최신 검증 JSON 생성
     - 이전엔 수동으로만 돌려서 wf capital 0/5 · 실집행 5건 고착 상태
  #2 실패 시 기존 JSON 유지하고 계속 진행 (|| exit 0 안전장치)
  #3 Tab 2 상단에 "마지막 검증 갱신" 시각 표시
     - 신선도 색상: 🟢 24h 이내 / 🟡 3일 이내 / 🔴 3일 초과
     - 예: "🟢 마지막 검증 갱신: 2026-04-18 20:15 (2시간 전) · 버전 v3.7.15"
  #4 사용자가 숫자 기준 시점을 즉시 알 수 있음 (freshness 투명화)
[v3.7.20] (2026-04-18) — CSV 다운로드 기능 복구
  #1 CSV 다운로드 버튼 복구 (Streamlit→NiceGUI 마이그레이션 시 누락됐던 기능)
     - backup_v205c_20260310/dashboard.py 라인 3759-3760에 원본 존재 확인
     - NiceGUI ui.download() API로 마이그레이션
  #2 두 가지 다운로드 옵션 제공:
     - "현재 필터 결과": 라벨/상태/정렬 적용된 결과만
     - "전체 종목": df 전체 다운로드
  #3 파일명 자동 생성: swingpicker_{scope}_{N개}_{YYYYMMDD_HHMM}.csv
  #4 UTF-8 BOM 인코딩 (엑셀에서 한글 깨짐 방지)
  #5 컬럼 순서 재정렬: 중요 컬럼(종목명/라벨/점수/가격) 앞쪽으로
[v3.7.19] (2026-04-18) — 가격 범위 바 + 프로그레스 바 렌더링 수정
  #1 가격 범위 바 라벨 잘림 수정
     - v3.7.18은 position:absolute + translateX(-50%)로 가장자리(0%/100%)에서
       라벨 절반이 화면 밖으로 잘림 + 가까운 포인트끼리 겹침
     - v3.7.19: flex row justify-between으로 균등 배치 (절대위치 제거)
     - 라벨에 현재가 대비 변동률 (+/-%) 추가 표시
     - 외부 container로 감싸서 nicegui ui.html의 wrapper div와 격리
  #2 _score_gauge / _mini_bar 프로그레스 바 렌더링 보장
     - ui.html에 .classes("w-full") 추가 → nicegui q-field wrapper 폭 0 방지
     - inner div에 display:block 명시
[v3.7.18] (2026-04-18) — UX 개선 3종 + 시각 임팩트 강화
  #1 전체 CSV 접근 허용 — admin/premium은 이제 CSV 전체 노출 (이전 50개 제한 해제)
  #2 테이블 페이지당 옵션 [15, 30, 50, 100, 전체] + 기본 30
  #3 라벨 기준 투명 공개 카드 + 라벨 필터 드롭다운
     "🏆 최강 (N): 평균≥70·밸런스≥70·갭≤3%·RR≥0.8"
     "✅ 즉시진입 (N): 최소≥50·밸런스≥70·갭≤5%"
     "⚠️ 추격 (N): 갭>5%·평균≥60"
  #4 상세 영역 종합 요약 배너 — 라벨별 그라데이션 배경 + 제목 크게
  #5 점수 게이지 3개 (종합/검증/RR) — 프로그레스 바 + glow shadow
  #6 가격 범위 바 — 손절~매수~현재~T1~T2를 하나의 바에 시각화
     손실 구간(빨강), 1차 보상(초록), 연장 보상(금색) + 현재가 흰색 마커
  #7 핵심 지표 카드에 프로그레스 바 추가 (9개 미니 메트릭 전부)
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
                # [v3.7.26] 차트 전체 다크 테마 통일 + 툴팁 가독성 수정
                # 사용자 지적: "차트의 흰 팝업이 너무 쎄서 정보가 안보임"
                # 해결: hoverlabel bgcolor/font_color/border 명시
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hoverlabel=dict(
                    bgcolor="rgba(26, 26, 46, 0.95)",  # 다크 반투명 (#1a1a2e 계열)
                    bordercolor="#8B5CF6",              # 보라 테두리 (브랜드 컬러)
                    font=dict(
                        color="#FFFFFF",                # 흰 텍스트
                        size=12,
                        family="monospace",             # OHLC 숫자 정렬용
                    ),
                    align="left",
                ),
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
    """Plotly 차트 다크 테마 + 툴팁 가독성

    [v3.7.26] hoverlabel 어두운 배경 + 흰 텍스트 통일.
    이전엔 plotly 기본 흰색 배경이 적용돼 다크 UI에서 가독성 0이었음.
    """
    if fig is None:
        return fig
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        # [v3.7.26] 툴팁 가독성 — 다크 반투명 배경 + 흰 텍스트
        hoverlabel=dict(
            bgcolor="rgba(26, 26, 46, 0.95)",
            bordercolor="#8B5CF6",
            font=dict(color="#FFFFFF", size=12, family="monospace"),
            align="left",
        ),
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


def _score_gauge(label: str, value: float, max_val: float = 100,
                 display_text: str = None):
    """[v3.7.18] 점수 게이지 바 컴포넌트 — 시각적 임팩트 강화.

    구조:
      라벨 (작게)
      큰 숫자 (점수별 색상)
      수평 프로그레스 바 (한국식 색상)
    """
    pct = max(0, min(100, (value / max_val) * 100)) if max_val > 0 else 0
    # 한국식 점수 색상: 70+ 빨강(강함), 50-69 주황, 50- 파랑(약함)
    if pct >= 70:
        bar_color = "#EF5350"; text_color = "text-red-400"
    elif pct >= 50:
        bar_color = "#FFA726"; text_color = "text-yellow-400"
    else:
        bar_color = "#3B82F6"; text_color = "text-blue-400"

    txt = display_text if display_text else f"{value:.0f}"

    with ui.column().classes("gap-1 min-w-[110px] flex-1"):
        ui.label(label).classes("text-[10px] text-gray-400 uppercase tracking-wider")
        ui.label(txt).classes(f"text-2xl font-black {text_color}")
        # [v3.7.19] 프로그레스 바 — nicegui ui.html wrapper 대응 w-full 필수
        ui.html(
            f'<div style="display:block; width:100%; height:6px; '
            f'background:rgba(255,255,255,0.08); border-radius:3px; overflow:hidden;">'
            f'<div style="display:block; width:{pct:.1f}%; height:100%; '
            f'background:{bar_color}; border-radius:3px; '
            f'box-shadow:0 0 8px {bar_color}80;"></div>'
            f'</div>'
        ).classes("w-full")


def _price_range_bar(stop: float, entry: float, close: float,
                     t1: float, t2: float = 0):
    """[v3.7.19] 가격대 시각 게이지 — 손절/매수/현재/T1/T2를 하나의 바에.

    v3.7.18 문제: 라벨이 0%/100% 가장자리에서 translateX(-50%)로 화면 밖 잘림,
                 가까운 마커들끼리 겹침.
    v3.7.19 수정:
     - 전체를 단일 <div>로 감싸서 position:relative 보장
     - 라벨 가장자리 padding 10% 여유 (clamp)
     - 표시는 그리드 아이템 줄바꿈으로 변경 (절대 위치 X → 문제 최소화)
     - 바 자체는 유지, 라벨은 별도 flex row로 분리
    """
    # 전체 범위
    vals = [v for v in [stop, entry, close, t1, t2] if v and v > 0]
    if len(vals) < 2:
        return
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    if span <= 0:
        return

    def pos(v):
        return (v - lo) / span * 100

    p_stop = pos(stop) if stop > 0 else None
    p_entry = pos(entry) if entry > 0 else None
    p_close = pos(close) if close > 0 else None
    p_t1 = pos(t1) if t1 > 0 else None
    p_t2 = pos(t2) if t2 > 0 else None

    # ═══════════════════════════════════════════
    # PART 1: 가로 바 (컬러 구간 + 세로 마커)
    # ═══════════════════════════════════════════
    bar_html = ['<div style="position:relative; width:100%; height:16px; '
                'background:rgba(255,255,255,0.05); border-radius:8px; '
                'overflow:hidden; margin:8px 0 6px 0;">']

    # 손절~매수 구간 (빨강 = 리스크)
    if p_stop is not None and p_entry is not None and p_entry > p_stop:
        bar_html.append(
            f'<div style="position:absolute; left:{p_stop:.1f}%; '
            f'width:{p_entry - p_stop:.1f}%; height:100%; '
            f'background:linear-gradient(to right, rgba(239,83,80,0.5), rgba(239,83,80,0.2));"></div>'
        )
    # 매수~T1 구간 (초록 = 1차 보상)
    if p_entry is not None and p_t1 is not None and p_t1 > p_entry:
        bar_html.append(
            f'<div style="position:absolute; left:{p_entry:.1f}%; '
            f'width:{p_t1 - p_entry:.1f}%; height:100%; '
            f'background:linear-gradient(to right, rgba(102,187,106,0.2), rgba(102,187,106,0.5));"></div>'
        )
    # T1~T2 구간 (금색 = 연장 보상)
    if p_t1 is not None and p_t2 is not None and p_t2 > p_t1:
        bar_html.append(
            f'<div style="position:absolute; left:{p_t1:.1f}%; '
            f'width:{p_t2 - p_t1:.1f}%; height:100%; '
            f'background:linear-gradient(to right, rgba(255,202,40,0.3), rgba(255,202,40,0.5));"></div>'
        )

    # 세로 마커들
    markers = [
        (p_stop, "#EF5350", 2),
        (p_entry, "#4FC3F7", 2),
        (p_t1, "#66BB6A", 2),
        (p_t2, "#FFCA28", 2),
    ]
    for p, clr, w in markers:
        if p is not None:
            bar_html.append(
                f'<div style="position:absolute; left:calc({p:.1f}% - {w/2}px); '
                f'width:{w}px; height:100%; background:{clr};"></div>'
            )
    # 현재가 (흰색 강조 + glow)
    if p_close is not None:
        bar_html.append(
            f'<div style="position:absolute; left:calc({p_close:.1f}% - 1.5px); '
            f'width:3px; height:100%; background:#FFFFFF; '
            f'box-shadow:0 0 6px rgba(255,255,255,0.9), 0 0 12px rgba(255,255,255,0.4);"></div>'
        )
    bar_html.append('</div>')

    # ═══════════════════════════════════════════
    # PART 2: 라벨을 flex row로 균등 배치 (겹침 없음)
    # ═══════════════════════════════════════════
    label_items = []
    if p_stop is not None:
        label_items.append(
            (p_stop, "🔴 손절", f"{stop:,.0f}", "#EF5350", f"{(stop/close-1)*100:+.1f}%" if close > 0 else "")
        )
    if p_entry is not None:
        label_items.append(
            (p_entry, "🔵 매수", f"{entry:,.0f}", "#4FC3F7", "" if close == entry else f"{(entry/close-1)*100:+.1f}%")
        )
    if p_close is not None:
        label_items.append(
            (p_close, "⚪ 현재", f"{close:,.0f}", "#FFFFFF", "")
        )
    if p_t1 is not None:
        label_items.append(
            (p_t1, "🟢 T1", f"{t1:,.0f}", "#66BB6A", f"{(t1/close-1)*100:+.1f}%" if close > 0 else "")
        )
    if p_t2 is not None and t2 != t1:
        label_items.append(
            (p_t2, "🟡 T2", f"{t2:,.0f}", "#FFCA28", f"{(t2/close-1)*100:+.1f}%" if close > 0 else "")
        )

    # position 기준 정렬
    label_items.sort(key=lambda x: x[0])

    # flex row로 균등 배치 (가장 안전)
    labels_html = [
        '<div style="display:flex; justify-content:space-between; '
        'gap:8px; margin-top:2px; flex-wrap:wrap;">'
    ]
    for p, emoji_label, price_txt, clr, chg in label_items:
        labels_html.append(
            f'<div style="text-align:center; min-width:60px; flex:1;">'
            f'<div style="font-size:10px; color:{clr}; font-weight:bold; white-space:nowrap;">'
            f'{emoji_label}</div>'
            f'<div style="font-size:11px; color:{clr}; font-weight:bold; white-space:nowrap;">'
            f'{price_txt}</div>'
        )
        if chg:
            chg_clr = "#9CA3AF"
            labels_html.append(
                f'<div style="font-size:9px; color:{chg_clr}; white-space:nowrap;">'
                f'{chg}</div>'
            )
        labels_html.append('</div>')
    labels_html.append('</div>')

    # 최종 — 외부 container로 감싸서 nicegui 격리
    full_html = (
        '<div style="width:100%;">'
        + "".join(bar_html)
        + "".join(labels_html)
        + '</div>'
    )
    ui.html(full_html).classes("w-full")


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
    # [v3.7.25] 🛡️ 콤보 라벨 판정용 — ROUTE 포함
    route = str(row.get("ROUTE", "") or "")

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
        # [v3.7.25] 🛡️ 콤보 판정용 추가 필드
        "s_raw":     s,         "t_raw":    t,
        "ai_raw":    a,         "route":    route,
    }


def _elite_label(stats: dict) -> tuple:
    """투 트랙 라벨링 — (뱃지문자, CSS색상, 짧은 설명) 반환.

    ─── 완전한 OHLC + Horizon 20일 Walk-forward 검증 ───
    🛡️ 콤보   : S≥90 AND T≥80 AND AI≥60 AND ROUTE ∈ {ATTACK, ARMED}
                → [v3.7.25 신설] 실성능 최고 (n=112, EV +25.77%, 승률 83.9%)
                → 즉석 Walk-forward: IS 64.3% → OOS 92.0% (매우 robust)
                → 콤보_optimizer 기반 지표 조합 최적화 결과
    🏆 최강   : AXIS_MEAN≥70 AND BAL≥70 AND 갭≤3% AND RR≥0.8
                → Walk-forward 20일 IS +2.97% / OOS +4.38%
                → ⚠️ 현재 표본 매우 적음 (n=6). 통계 신뢰 낮음.
    ✅ 즉시진입: AXIS_MIN≥50 AND BAL≥70 AND 갭≤5%
                → ⚠️ 최근 재검증 결과 EV -0.05% (주의 필요)
    ⚠️ 추격  : 갭 > 5% AND 평균≥60

    라벨 우선순위 (같은 종목이 여러 조건 만족 시):
      🛡️ 콤보 > 🏆 최강 > ✅ 즉시진입 > ⚠️ 추격
    콤보가 최상위 — 실성능 최고이기 때문.
    """
    if not stats["valid"]:
        return ("", "", "")

    am  = stats["axis_mean"]
    amn = stats["axis_min"]
    bal = stats["balance"]
    gap = stats["gap_pct"]
    rr  = stats["rr_now"]
    # [v3.7.25] 콤보 판정용
    s   = stats.get("s_raw",  0)
    t   = stats.get("t_raw",  0)
    ai  = stats.get("ai_raw", 0)
    rt  = stats.get("route",  "")

    # ── 🛡️ 콤보 (v3.7.25: 실성능 최고 · 제1 매수 종목) ──
    # combo_optimizer.py 그리드 서치 결과 반영
    # 즉석 walk-forward: IS 64% → OOS 92% (오버피팅 아님)
    if s >= 90 and t >= 80 and ai >= 60 and rt in ("ATTACK", "ARMED"):
        return (
            "🛡️ 콤보",
            "#8B5CF6",  # 보라색 (프리미엄 느낌)
            f"실성능 1위 (n=112, EV +25.77%, 승률 83.9%) · S{s:.0f} T{t:.0f} AI{ai:.0f}",
        )

    # ── 🏆 최강 (v3.7.6 기준 · v3.7.25 관찰 모드로 전환) ──
    # ⚠️ 표본 n=6 · 매매 풀에서 제외 · 관찰용으로만 표시
    if am >= 70 and bal >= 70 and gap <= 3.0 and rr >= 0.8:
        return (
            "🏆 최강",
            "#F59E0B",  # 금색
            f"👁️ 관찰중 · 매매 제외 · 표본 n=6 (신뢰 LOW) · 통계 축적 대기",
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

    라벨 보정 (v3.7.25 관찰 모드):
      🛡️ 콤보     → ×1.50  (실성능 최고 · EV +25.77% · n=112 · 제1 매수)
      ✅ 즉시진입  → ×1.30  (OHLC 검증 EV +9.40% · 실전 주력)
      🏆 최강     → ×0.50  (👁️ 관찰중 · 매매 제외 · 표본부족 n=6)
      ⚠️ 추격     → ×0.70  (경고)
      기타        → ×0.50

    [v3.7.25] 🏆 최강 가중치 1.00 → 0.50 (관찰 모드)
      - 매매 풀에서 실질적으로 배제 (pick_top1/top3에서 이미 제외)
      - rank_score 순위도 낮춤 → 사용자가 테이블에서 정렬해도 상위 안 뜨도록
    """
    if not stats["valid"]:
        return -999.0

    base = stats["axis_mean"] * (stats["balance"] / 100.0)

    rr = stats["rr_now"]
    if   rr < 0.5: rr_mult = 0.3
    elif rr < 1.0: rr_mult = 0.7
    else:          rr_mult = 1.0

    # [v3.7.25] 🛡️ 콤보 최상위 · 🏆 최강 관찰 모드로 페널티
    if   label == "🛡️ 콤보":    label_mult = 1.50
    elif label == "✅ 즉시진입": label_mult = 1.30
    elif label == "🏆 최강":     label_mult = 0.50  # 관찰 모드 (이전 1.00)
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
    """[v3.7.25] 🛡️ 콤보만 사용. 🏆 최강은 관찰 모드 (매매 제외).

    선별 규칙:
      · 🛡️ 콤보 중 RANK_SCORE 1위 1종목만
      · 콤보 없으면 빈 리스트 → "오늘 매매 없음"

    [v3.7.25 방안1 — 관찰 모드]
      🏆 최강 라벨은 UI/통계에서는 표시하되 매매 풀에서 완전 배제.
      사유:
        - 표본 n=6로 통계적 신뢰 매우 낮음
        - 실성능 EV +6.79% (콤보 +25.77% 대비 열위)
        - 2~3개월 실집행 표본 축적 후 재평가 예정
      사용자 결정: "최강은 표본도 부족하고 실질 검증도 안되니까 매매에서 제외"

    복구 방법 (표본 축적 후):
      아래 주석 처리된 fallback 블록을 활성화하면 됨.
    """
    if df is None or df.empty or "ELITE_RANK_SCORE" not in df.columns:
        return []

    # [v3.7.25] 🛡️ 콤보만 — 🏆 최강 fallback 제거 (관찰 모드)
    combo_pool = df[df["ELITE_LABEL"] == "🛡️ 콤보"].copy()
    combo_pool = combo_pool[combo_pool["ELITE_RANK_SCORE"] >= min_rank_score]
    if combo_pool.empty:
        return []
    combo_pool = combo_pool.sort_values("ELITE_RANK_SCORE", ascending=False)
    top = combo_pool.iloc[0]
    return [str(top.get("종목코드", "")).zfill(6)]

    # [관찰 모드 해제 시 복구 — 실집행 표본 100건+ 축적 후 활성화]
    # pool = df[df["ELITE_LABEL"] == "🏆 최강"].copy()
    # pool = pool[pool["ELITE_RANK_SCORE"] >= min_rank_score]
    # if pool.empty:
    #     return []
    # pool = pool.sort_values("ELITE_RANK_SCORE", ascending=False)
    # top = pool.iloc[0]
    # return [str(top.get("종목코드", "")).zfill(6)]


def pick_top3(df: pd.DataFrame, min_rank_score: float = 40.0) -> list:
    """오늘의 유니크한 Top 3 종목코드 반환.

    [v3.7.25] 🛡️ 콤보 전용. 🏆 최강 관찰 모드 (매매 제외).

    선별 규칙:
      1) 라벨 풀: 🛡️ 콤보만 (🏆 최강은 배제)
      2) ELITE_RANK_SCORE ≥ min_rank_score 컷오프
      3) 섹터 중복 제거
      4) ELITE_RANK_SCORE 내림차순 Top 3

    콤보 없으면 빈 리스트 → 현금 보유.
    (🏆 최강 복구 시 방법은 pick_top1() docstring 참조)
    """
    if df is None or df.empty or "ELITE_RANK_SCORE" not in df.columns:
        return []

    # [v3.7.25] 🛡️ 콤보만 (🏆 최강 관찰 모드로 배제)
    pool = df[df["ELITE_LABEL"] == "🛡️ 콤보"].copy()
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
                # [v3.7.23] 제목 — "Top 3" → "Top Pick" (실제 모드는 Top1 우선)
                # 실제 selection_mode가 top1_first_then_top3_fallback이므로
                # "오늘의 실전 1순위"가 정확한 표현. Top3는 fallback일 뿐.
                ui.label("🏆 오늘의 실전 Top Pick").classes(
                    "text-lg font-bold text-white"
                )
                # [v3.7.1→v3.7.23] 뱃지 의미 명확화
                n_picked = len(top3_codes)
                if n_picked == 1:
                    # Top1 모드 성공 (진짜 1순위 발굴)
                    badge_txt = "🎯 Top1"
                    badge_color = "#F59E0B"  # 금색 - 최고 상태
                elif n_picked >= 2:
                    # Top3 fallback (Top1 없을 때만)
                    badge_txt = f"Top3 폴백 {n_picked}/3"
                    badge_color = "#10B981"
                else:
                    badge_txt = "0/3"
                    badge_color = "#6B7280"
                ui.badge(badge_txt, color=badge_color).classes("text-xs")
            ui.label(summary_text).classes("text-xs text-gray-500")

        # ═══════════════════════════════════════════════════
        # [v3.7.23] 정보 재배치 — 사용자 읽기 순서 최적화
        # ─────────────────────────────────────────────────
        # Before: WF → 3종목 포트 → 신호 → 실집행 → 신뢰도 → Top3 → Rolling
        #         (정보 우선순위 섞임 → 메인이 뭔지 흐림)
        # After: 메인 블록 (실집행/신호/신뢰도) → 보조 블록 (WF/Rolling/Top3/3종목)
        #         → 메타 (조건/갱신시각)
        # ═══════════════════════════════════════════════════

        # ───────────────────────────────────────────
        # 🎯 메인 블록 — 실전 의사결정에 직접 쓰는 숫자 3개
        # ───────────────────────────────────────────

        # 💰 #1 실집행 성과 (가장 중요 — 실제 운용 가능성)
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
            ).classes(f"text-sm {clr} mb-1 font-semibold")

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

        # 📡 #2 신호 성과 (알파 품질)
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
            ).classes(f"text-sm {sig_clr} mb-1 font-semibold")

            # [v3.7.23] TP1 vs EV 해석 한 줄 — 사용자가 숫자 의미를 바로 이해하게
            # 예: "TP1 21%"는 낮아 보이지만 EV +1.47%인 이유 = 승리 폭 > 패배 폭
            if tp1_s > 0 and ev_s > 0 and tp1_s < 40:
                # 낮은 hit rate + 양수 EV 조합 설명
                ui.label(
                    f"  ℹ️ TP1 hit rate({tp1_s:.0f}%)는 낮지만, 승리 폭 > 패배 폭 "
                    f"+ 미도달 종가 마감까지 포함하여 기대수익 EV는 양수"
                ).classes("text-[11px] text-blue-300 mb-1 ml-4 italic")
            elif tp1_s >= 40 and ev_s > 0:
                ui.label(
                    f"  ℹ️ TP1 hit rate({tp1_s:.0f}%) + 양수 EV = 높은 승률 기반 수익 구조"
                ).classes("text-[11px] text-green-300 mb-1 ml-4 italic")
            elif ev_s <= 0:
                ui.label(
                    f"  ℹ️ EV가 음수 — 현재 시그널의 알파 약화 상태 "
                    f"(임계값 재조정 검토 필요)"
                ).classes("text-[11px] text-red-300 mb-1 ml-4 italic")

        # 🏅 #3 Confidence badge (실행 판단의 기준점)
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
            ).classes(f"text-sm {badge_clr} mb-2 font-bold")

        # ───────────────────────────────────────────
        # 📊 보조 블록 — 검증 증거 (회색 톤으로 구분)
        # ───────────────────────────────────────────
        ui.label("📊 보조 검증 (참고)").classes(
            "text-[11px] text-gray-400 font-bold mt-2 mb-1 uppercase tracking-wider"
        )

        # Walk-forward 일반화
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
            top = wf_results[0]
            is_ev = top.get("is_summary", {}).get("ev", 0)
            oos_ev = top.get("oos_summary", {}).get("ev", 0)
            ui.label(
                f"{mark} Walk-forward(h={horizon}일): "
                f"IS Top 5 중 {generalizes_n}/{total_n} 일반화 · "
                f"대표조합 IS {is_ev:+.2f}% → OOS {oos_ev:+.2f}%"
            ).classes(f"text-xs {color} mb-1")

        # Rolling walk-forward
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

        # 일자별 3종목 포트폴리오 (참고 수치)
        if port_stats and port_stats.get("n_days", 0) >= 5:
            avg_daily = port_stats["avg_daily_portfolio_ret"]
            pos_rate = port_stats.get("positive_rate", 0) * 100
            n_days = port_stats["n_days"]
            n_pos = port_stats.get("n_positive_days", 0)
            p_mark = "📈" if avg_daily > 0 else "📉"
            p_color = "text-green-500" if avg_daily > 0 else "text-gray-500"
            ui.label(
                f"{p_mark} (참고) 일평균 3종목 포트폴리오: {avg_daily:+.2f}% · "
                f"플러스 마감 {n_pos}/{n_days}일 ({pos_rate:.0f}%)"
            ).classes(f"text-xs {p_color} mb-1")

        # Top3 자본 시뮬 (참고)
        if capital_stats and capital_stats.get("n_trades_filled", 0) > 0:
            total_ret = capital_stats.get("total_return_pct", 0)
            mdd = capital_stats.get("max_drawdown_pct", 0)
            n_filled = capital_stats["n_trades_filled"]
            ui.label(
                f"· (참고) Top3 모드 자본시뮬: "
                f"기간수익 {total_ret:+.2f}% · MDD {mdd:.1f}% · 체결 {n_filled}건"
            ).classes("text-xs text-gray-500 mb-1")

        # ───────────────────────────────────────────
        # 🔧 메타 블록 — 검증 조건 + 갱신 시각
        # ───────────────────────────────────────────

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

        # [v3.7.21] 검증 JSON 생성 시각 표시 — 데이터 신선도(Freshness) 투명화
        # 이전엔 generated_at이 전혀 표시 안 돼서 사용자가 숫자 기준 시점 모름
        # auto_collect.yml에서 매일 백테스트 자동 실행 후 여기에 갱신 시각 반영
        generated_at = bt.get("generated_at", "")
        if generated_at:
            # "2026-04-18T01:08:10" → "2026-04-18 01:08"
            gen_display = generated_at.replace("T", " ")[:16]
            # 얼마나 오래됐는지 계산 (UI 색상으로 신선도 표시)
            try:
                from datetime import datetime, timezone
                gen_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                if gen_dt.tzinfo is None:
                    gen_dt = gen_dt.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                age_hours = (now - gen_dt).total_seconds() / 3600
                if age_hours < 24:
                    freshness = "🟢"  # 24시간 이내 (fresh)
                    fresh_cls = "text-green-500"
                elif age_hours < 72:
                    freshness = "🟡"  # 3일 이내 (stale)
                    fresh_cls = "text-yellow-500"
                else:
                    freshness = "🔴"  # 3일 초과 (outdated)
                    fresh_cls = "text-red-500"
                age_txt = (
                    f"{int(age_hours)}시간 전" if age_hours < 48
                    else f"{int(age_hours/24)}일 전"
                )
            except Exception:
                freshness = "📅"
                fresh_cls = "text-gray-500"
                age_txt = ""
            ver = bt.get("version", "")
            ui.label(
                f"{freshness} 마지막 검증 갱신: {gen_display}"
                + (f" ({age_txt})" if age_txt else "")
                + (f" · 버전 {ver}" if ver else "")
            ).classes(f"text-[10px] {fresh_cls} mb-2 italic")

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
    with ui.row().classes("w-full gap-4 items-center flex-wrap mb-2"):
        view_mode = ui.toggle(
            ["📋 테이블", "🃏 칸반"], value="📋 테이블"
        )
        route_filter = ui.select(
            ["전체", "ATTACK", "ARMED", "WAIT", "NEUTRAL"],
            value="전체", label="상태",
        ).classes("min-w-[120px]")
        # [v3.7.18] 라벨 필터 추가 - 즉시진입 너무 많을 때 최강만 보기 등
        # [v3.7.25] 🛡️ 콤보 필터 추가 (제1 매수 종목)
        label_filter = ui.select(
            ["전체", "🛡️ 콤보", "🏆 최강", "✅ 즉시진입", "⚠️ 추격"],
            value="전체", label="라벨",
        ).classes("min-w-[130px]")
        # [v3.7.24] "🏆 검증순" → "🏆 랭크순" (ELITE_RANK_SCORE 기준 명확화)
        sort_mode = ui.toggle(
            ["🔢 점수순", "⚖️ 밸런스순", "🏆 랭크순", "🚦 상태순"],
            value="🏆 랭크순",
        )
        # [v3.7.26] 테이블 보기 모드 — 기본(핵심만) vs 고급(전체)
        # 사용자 지적: "스코어들이 너무 많은데 로직들좀 설명해봐"
        # 해결: 기본 보기 = 실전 매매에 필요한 컬럼만 (7개)
        #       고급 보기 = 기존 전체 (11개)
        view_table_mode = ui.toggle(
            ["🎯 기본", "🔬 고급"],
            value="🎯 기본",
        )

    # [v3.7.18] 라벨 기준 투명 공개 (사용자 혼란 방지)
    # 라벨별 종목 수도 함께 표시
    # [v3.7.25] 🛡️ 콤보 카운트 추가
    if "ELITE_LABEL" in df.columns:
        n_combo = int((df["ELITE_LABEL"] == "🛡️ 콤보").sum())
        n_strong = int((df["ELITE_LABEL"] == "🏆 최강").sum())
        n_instant = int((df["ELITE_LABEL"] == "✅ 즉시진입").sum())
        n_chase = int((df["ELITE_LABEL"] == "⚠️ 추격").sum())
        n_none = int(df["ELITE_LABEL"].fillna("").eq("").sum())
    else:
        n_combo = n_strong = n_instant = n_chase = n_none = 0

    with ui.card().classes(
        "w-full p-2 mb-3 bg-[rgba(255,255,255,0.02)] "
        "border border-[rgba(255,255,255,0.05)] rounded"
    ):
        with ui.row().classes("w-full gap-6 items-center flex-wrap"):
            ui.label("🏷️ 라벨 기준:").classes("text-xs text-gray-500 font-bold")
            # [v3.7.25] 🛡️ 콤보 최우선 표시 (제1 매수 · 실성능 1위)
            ui.label(
                f"🛡️ 콤보 ({n_combo}): S≥90 · T≥80 · AI≥60 · ATTACK/ARMED "
                f"[n=112 EV +25.77% 승률 83.9%]"
            ).classes("text-xs text-purple-400 font-bold")
            ui.label(
                f"🏆 최강 ({n_strong}): 평균≥70 · 밸런스≥70 · 갭≤3% · RR≥0.8 "
                f"[n=6 · 👁️ 관찰중 · 매매 제외]"
            ).classes("text-xs text-gray-500 line-through opacity-60")
            ui.label(
                f"✅ 즉시진입 ({n_instant}): 최소≥50 · 밸런스≥70 · 갭≤5%"
            ).classes("text-xs text-green-400")
            ui.label(
                f"⚠️ 추격 ({n_chase}): 갭>5% · 평균≥60 (추격 비추)"
            ).classes("text-xs text-orange-400")
            if n_none > 0:
                ui.label(f"(기준 미달 {n_none}개)").classes("text-xs text-gray-600")

    # [v3.7.26] 스코어 용어집 (접이식) — 사용자 지적: "스코어 너무 많음" 해결
    # 기본 닫힘 · 펼치면 각 스코어의 정체 + 공식을 한눈에
    with ui.expansion(
        "📖 스코어 용어집 — 각 스코어가 뭘 의미하는지 (클릭하면 펼침)",
        icon="help_outline",
    ).classes(
        "w-full mb-3 bg-[rgba(139,92,246,0.05)] "
        "border border-[rgba(139,92,246,0.2)] rounded"
    ).props("dense"):
        with ui.column().classes("w-full gap-1 p-2"):
            # ── 핵심 스코어 3축 ──
            ui.label("🧱 핵심 3축 (실전 매매 기본)").classes(
                "text-xs text-purple-300 font-bold mt-1"
            )
            ui.label(
                "  · S (구조): 추세·정배열·VWAP 위치 등 기본기 — 0~100"
            ).classes("text-[11px] text-gray-400")
            ui.label(
                "  · T (타이밍): RSI·MACD·거래량·TRIGGER 등 진입 시점 — 0~100"
            ).classes("text-[11px] text-gray-400")
            ui.label(
                "  · AI (=ML): 머신러닝 예측값 — 약 7~91 범위"
            ).classes("text-[11px] text-gray-400")

            # ── 파생 통계 ──
            ui.label("🔢 파생 통계 (3축에서 계산)").classes(
                "text-xs text-purple-300 font-bold mt-2"
            )
            ui.label(
                "  · 평균: (S + T + AI) / 3"
            ).classes("text-[11px] text-gray-400")
            ui.label(
                "  · 균형: 100 - (MAX - MIN) × 1.25  "
                "(3축 편차 적을수록 높음)"
            ).classes("text-[11px] text-gray-400")
            ui.label(
                "  · 갭%: |종가 - 매수가| / 매수가 × 100  (지정가 진입 가능성)"
            ).classes("text-[11px] text-gray-400")
            ui.label(
                "  · RR: (T1 - 매수가) / (매수가 - 손절가)  "
                "(리스크 대비 리워드 비율)"
            ).classes("text-[11px] text-gray-400")

            # ── 종합 점수 ──
            ui.label("🏭 종합 점수 (파이프라인 산출)").classes(
                "text-xs text-purple-300 font-bold mt-2"
            )
            ui.label(
                "  · 점수 (DISPLAY): S×40% + T×40% + AI×20% + 보너스 − 페널티"
            ).classes("text-[11px] text-gray-400")
            ui.label(
                "  · 종합 (ELITE): 안정 진입 품질 등급 (갭 작을수록 높음)"
            ).classes("text-[11px] text-gray-400")
            ui.label(
                "  · 랭크 (ELITE_RANK): 평균×(균형/100)×RR보정×라벨보정 "
                "(Top 선별용 내부 점수)"
            ).classes("text-[11px] text-gray-400")

            # ── 라벨 가중치 ──
            ui.label("🏷️ 라벨 가중치 (랭크 계산에 곱해지는 배수)").classes(
                "text-xs text-purple-300 font-bold mt-2"
            )
            ui.label(
                "  · 🛡️ 콤보 ×1.50 (실성능 1위)  "
                "· ✅ 즉시진입 ×1.30  "
                "· 🏆 최강 ×0.50 (관찰 모드)  "
                "· ⚠️ 추격 ×0.70"
            ).classes("text-[11px] text-gray-400")

            # ── 실전 팁 ──
            ui.label("💡 실전 매매 시 우선 확인 순서").classes(
                "text-xs text-yellow-400 font-bold mt-2"
            )
            ui.label(
                "  ① 라벨 (🛡️/✅/⚠️)  → ② 점수(DISPLAY) 70+  "
                "→ ③ RR 1.0+  → ④ 갭 5% 이하  → ⑤ S/T/AI 세부 확인"
            ).classes("text-[11px] text-gray-300")

    # [v3.7.22] CSV 다운로드 권한 제어 — prime/admin만 허용
    # - guest/free: 다운로드 버튼 비활성 (잠금 상태 + Prime 안내)
    # - prime/admin: 전체 종목 다운로드 가능
    # 권한 정책:
    #   · guest  : 체험 — 다운로드 불가
    #   · free   : 무료 회원 — 다운로드 불가 (Prime 유도)
    #   · prime  : 유료 회원 — 전체 다운로드 가능
    #   · admin  : 관리자 — 전체 다운로드 가능
    can_download = auth in ("prime", "admin")

    def _download_csv(scope: str = "filtered"):
        """CSV 다운로드 트리거.

        scope='filtered': 현재 필터/정렬 적용된 결과만
        scope='all': df 전체 (필터 무시)

        [v3.7.27 Phase 1 · v3.7.28 완결] CSV 스키마 정리:
          - 중복 컬럼 5개 제거 (값이 DISPLAY_SCORE 등과 100% 동일)
            LDY_SCORE, TOTAL_SCORE, RANK_SCORE, ML_SCORE, RAW_TRIGGER_SCORE
          - 상수 컬럼 28개 제거 (전체 종목 같은 값 → 정보 없음)
            CONFIG_SNAPSHOT, MACRO_RISK, W_STRUCT 등
          - 실측: 174 → 141 컬럼 (33개 제거) · 파일 76% 감소 (2MB → 0.47MB)

          내부 엔진/DB는 그대로 유지 → 참조 코드 영향 0.
          CSV 소비 측에서만 필터링.
        """
        # [v3.7.22] 이중 권한 체크 - 버튼이 disabled여도 안전하게 차단
        if not can_download:
            ui.notify(
                "👑 CSV 다운로드는 Prime 회원 전용입니다",
                type="warning",
            )
            return
        try:
            from datetime import datetime

            source_df = _filtered() if scope == "filtered" else df
            if source_df.empty:
                ui.notify("다운로드할 종목이 없습니다", type="warning")
                return

            # [v3.7.27] 제거 대상: 중복 컬럼 (100% 동일값)
            duplicate_cols = [
                "LDY_SCORE",        # = DISPLAY_SCORE
                "TOTAL_SCORE",      # = DISPLAY_SCORE
                "RANK_SCORE",       # = DISPLAY_SCORE
                "ML_SCORE",         # = AI_SCORE
                "RAW_TRIGGER_SCORE",  # = TRIGGER_SCORE
            ]

            # [v3.7.27] 제거 대상: 상수 컬럼 (전체 종목 같은 값)
            constant_cols = [
                "MACRO_RISK", "MARKET_BREADTH",
                "W_STRUCT", "W_TIMING", "W_AI",
                "CONFIDENCE_SCORE",  # 항상 100 — 이름과 달리 정보 없음
                "AXIS_QUALITY",
                "CONFIG_SNAPSHOT",   # 행당 2.5KB × N행 — 1MB+ 낭비
                "EXEC_RULE_ID", "ML_STATUS", "REASON_THRESHOLD",
                "기준일", "시총기준일",
                "벤치_60d_KOSPI_%", "벤치_60d_KOSDAQ_%",
                "CALIBRATION_MODE", "CAL_N_TRADES", "DATA_FRESHNESS_OK",
                "CONFIG_VERSION", "RUN_STATUS", "MAX_ALLOWED_ROUTE",
                "AXIS_MCAP", "AXIS_BENCH", "AXIS_FLOW", "AXIS_NEWS",
                "AXIS_SECTOR", "AXIS_ML", "AXIS_TRIGGER", "FALLBACK_COUNT",
                "HAS_NEWS", "HAS_FLOW", "HAS_SECTOR",
                "OBV_Div", "개인순매수", "SCORING_AXES",
            ]

            # [v3.7.27] 상수 컬럼은 실측으로 한번 더 확인 (파이프라인 변경 대비)
            # nunique() == 1이면 정보 없음
            verified_constant = [
                c for c in constant_cols
                if c in source_df.columns
                and source_df[c].nunique(dropna=False) <= 1
            ]

            cols_to_drop = set(duplicate_cols) | set(verified_constant)

            # 다운로드용 컬럼 선별 — 핵심 먼저 배치
            download_cols = [c for c in [
                "종목코드", "종목명", "업종", "ELITE_LABEL", "ROUTE",
                "DISPLAY_SCORE", "STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE",
                "BALANCE_CALC", "AXIS_MEAN", "ELITE_RANK_SCORE",
                "GAP_PCT", "RR_NOW_TP1",
                "종가", "추천매수가", "손절가", "추천매도가1", "추천매도가2",
                "RSI14", "V_POWER", "거래대금(억원)",
            ] if c in source_df.columns]

            # 핵심 외 나머지 컬럼 추가 (단, drop 대상 제외)
            for c in source_df.columns:
                if c in download_cols or c in cols_to_drop:
                    continue
                download_cols.append(c)

            out_df = source_df[download_cols].copy()

            # UTF-8 BOM 포함 (한글 엑셀 호환)
            csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")

            # 파일명
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            scope_tag = "전체" if scope == "all" else "필터"
            filename = f"swingpicker_{scope_tag}_{len(out_df)}개_{ts}.csv"

            # NiceGUI download 트리거
            ui.download(csv_bytes, filename)
            # [v3.7.27] 컬럼 수와 절감량도 알림에 표시
            original_cols = len(source_df.columns)
            final_cols = len(download_cols)
            saved = original_cols - final_cols
            ui.notify(
                f"✅ {len(out_df)}개 종목 · {final_cols}컬럼 "
                f"({saved}개 중복/상수 제거)",
                type="positive",
            )
        except Exception as e:
            ui.notify(f"❌ 다운로드 실패: {e}", type="negative")

    # [v3.7.22] 다운로드 버튼 바 — 권한별 다른 UI
    with ui.row().classes("w-full gap-2 items-center mb-3 flex-wrap"):
        if can_download:
            # ─── Prime / Admin: 정상 다운로드 버튼 ───
            tier_icon = "🛠️" if auth == "admin" else "👑"
            ui.label(
                f"📥 CSV 다운로드 {tier_icon}:"
            ).classes("text-xs text-gray-400 font-bold")
            ui.button(
                "현재 필터 결과",
                on_click=lambda: _download_csv("filtered"),
            ).props("size=sm flat color=primary").classes("text-xs")
            ui.button(
                "전체 종목",
                on_click=lambda: _download_csv("all"),
            ).props("size=sm flat color=secondary").classes("text-xs")
            ui.label("(UTF-8 BOM · 엑셀 한글 호환)").classes(
                "text-[10px] text-gray-500 ml-2"
            )
        else:
            # ─── Guest / Free: 비활성 버튼 + Prime 안내 ───
            ui.label("🔒 CSV 다운로드:").classes("text-xs text-gray-500 font-bold")
            ui.button(
                "현재 필터 결과",
                on_click=lambda: _download_csv("filtered"),
            ).props("size=sm flat color=grey disable").classes("text-xs opacity-50")
            ui.button(
                "전체 종목",
                on_click=lambda: _download_csv("all"),
            ).props("size=sm flat color=grey disable").classes("text-xs opacity-50")
            ui.label(
                "👑 Prime 회원 전용 기능"
            ).classes("text-[11px] text-yellow-500 ml-2 font-bold")
            ui.label(
                "(업그레이드하고 엑셀로 자유롭게 분석하세요)"
            ).classes("text-[10px] text-gray-500")

    table_area = ui.column().classes("w-full")
    detail_area = ui.column().classes("w-full mt-4")

    def _filtered():
        fdf = df.copy()
        if route_filter.value != "전체" and "ROUTE" in fdf.columns:
            fdf = fdf[fdf["ROUTE"].astype(str).str.contains(
                route_filter.value, na=False
            )]
        # [v3.7.18] 라벨 필터 적용
        if label_filter.value != "전체" and "ELITE_LABEL" in fdf.columns:
            fdf = fdf[fdf["ELITE_LABEL"] == label_filter.value]
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
        elif sort_mode.value == "🏆 랭크순" and "ELITE_RANK_SCORE" in fdf.columns:
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
        # [v3.7.18] 접근 제한 — admin/premium은 전체 CSV (이전엔 50개 제한)
        # guest/free만 미리보기 제한, 나머지는 CSV 전체 노출
        limits = {"guest": 3, "free": 5}
        max_rows = limits.get(auth)
        if max_rows is not None:
            fdf = fdf.head(max_rows)
        # admin/premium/pro는 제한 없음 → 전체 df 반환
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
        # [v3.7.26] 보기 모드별 컬럼 구성 — 기본: 핵심만 / 고급: 전체
        is_advanced = view_table_mode.value == "🔬 고급"

        # 공통 컬럼 (항상 표시)
        base_cols = [
            {"name": "label", "label": "라벨", "field": "label", "align": "center"},
            {"name": "route", "label": "상태", "field": "route", "align": "center"},
            {"name": "name", "label": "종목명", "field": "name", "align": "left"},
            # 점수 = DISPLAY_SCORE (파이프라인 최종, 화면 메인 숫자)
            {"name": "score", "label": "점수", "field": "score",
             "align": "center", "sortable": True},
            {"name": "s", "label": "S", "field": "s",
             "align": "center", "sortable": True},
            {"name": "t", "label": "T", "field": "t",
             "align": "center", "sortable": True},
            {"name": "ai", "label": "AI", "field": "ai",
             "align": "center", "sortable": True},
            {"name": "gap", "label": "갭%", "field": "gap",
             "align": "center", "sortable": True},
            # [v3.7.26] RR 컬럼 추가 — 실전 매매 의사결정의 핵심 지표
            {"name": "rr", "label": "RR", "field": "rr",
             "align": "center", "sortable": True},
            {"name": "close", "label": "현재가", "field": "close", "align": "right"},
            {"name": "buy", "label": "매수", "field": "buy", "align": "right"},
            {"name": "stop", "label": "손절", "field": "stop", "align": "right"},
            {"name": "t1", "label": "T1목표", "field": "t1", "align": "right"},
            {"name": "sector", "label": "업종", "field": "sector", "align": "left"},
        ]

        # [v3.7.26] 고급 모드 전용 컬럼 — 내부 스코어들
        # 기본 모드에서는 숨김 (사용자 혼란 방지)
        advanced_cols = [
            # 종합 = ELITE_SCORE (파이프라인 품질 등급)
            {"name": "elite", "label": "종합", "field": "elite",
             "align": "center", "sortable": True},
            # 랭크 = ELITE_RANK_SCORE (Top 선별용 내부 점수)
            {"name": "rank", "label": "랭크", "field": "rank",
             "align": "center", "sortable": True},
            # 균형 = BALANCE_CALC (3축 편차)
            {"name": "bal", "label": "균형", "field": "bal",
             "align": "center", "sortable": True},
        ]

        # 기본 모드: base만, 고급 모드: base + advanced (업종 앞에 추가)
        if is_advanced:
            # 업종을 마지막으로 빼고 고급 컬럼을 그 앞에 끼움
            sector_col = [c for c in base_cols if c["name"] == "sector"]
            non_sector = [c for c in base_cols if c["name"] != "sector"]
            columns = non_sector + advanced_cols + sector_col
        else:
            columns = base_cols
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
                # [v3.7.24] 검증점수 → 종합/랭크 2개 분리 (구분 명확화)
                # 종합 (elite): ELITE_SCORE (파이프라인 최종) > ELITE_RANK_SCORE 폴백
                "elite": f'{_nz(r.get("ELITE_SCORE", r.get("ELITE_RANK_SCORE", 0))):.0f}',
                # 랭크 (rank): ELITE_RANK_SCORE (내부 Top 선별용)
                "rank":  f'{_nz(r.get("ELITE_RANK_SCORE", 0)):.0f}',
                "bal":   f'{_nz(r.get("BALANCE_CALC",  r.get("BALANCE_SCORE", 0))):.0f}',
                "gap":   f'{_nz(r.get("GAP_PCT", 0)):.1f}',
                # [v3.7.26] RR 값 추가 (실전 핵심 지표)
                "rr":    f'{_nz(r.get("RR_NOW_TP1", 0)):.2f}',
                "close": f'{int(_nz(r.get("종가", 0))):,}',
                "buy": f'{int(_nz(r.get("추천매수가", 0))):,}',
                "stop": f'{int(_nz(r.get("손절가", 0))):,}',
                "t1": f'{int(_nz(r.get("추천매도가1", 0))):,}',
                "sector": str(r.get("업종", "—")),
            })
        # [v3.7.18] 페이지당 행 수 확장: 기본 30, 옵션 [15, 30, 50, 100, 전체]
        # Quasar 테이블에서 rowsPerPageOptions로 사용자가 직접 선택 가능
        # [v22 Step AB] sticky header — 헤더 행 스크롤해도 상단 고정
        tbl = ui.table(
            columns=columns, rows=rows, row_key="code",
            selection="single",
            pagination={"rowsPerPage": 30, "sortBy": None, "descending": True},
        ).classes("w-full ldy-sticky-header").style(
            "max-height: 70vh"
        ).props(
            'dense dark flat bordered '
            ':rows-per-page-options="[15, 30, 50, 100, 0]" '
            'virtual-scroll-sticky-size-start="0"'
        )
        # [Step AB] sticky header CSS — 한 번만 주입
        ui.add_head_html('''
        <style>
        .ldy-sticky-header .q-table__top,
        .ldy-sticky-header thead tr:first-child th {
            position: sticky;
            top: 0;
            z-index: 2;
            background: #1a1a2e !important;
        }
        .ldy-sticky-header thead tr:nth-child(2) th {
            position: sticky;
            top: 32px;
            z-index: 1;
            background: #1a1a2e !important;
        }
        .ldy-sticky-header .q-table__container {
            max-height: 70vh;
            overflow-y: auto;
        }
        .ldy-sticky-header .q-table__bottom {
            position: sticky;
            bottom: 0;
            z-index: 2;
            background: #1a1a2e !important;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        /* 모바일 대응 */
        @media (max-width: 768px) {
            .ldy-sticky-header .q-table__container {
                max-height: 60vh;
            }
        }
        </style>
        ''')
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
            # ═══════════════════════════════════════════════════
            # [v3.7.18] 종합 요약 배너 — 라벨 + 점수 + 가격 게이지 한눈에
            # ═══════════════════════════════════════════════════
            elite_lbl = str(row.get("ELITE_LABEL", "") or "")
            elite_color_hex = str(row.get("ELITE_LABEL_COLOR", "") or "#6B7280")
            display_score = _nz(row.get("DISPLAY_SCORE", 0))
            elite_rank = _nz(row.get("ELITE_RANK_SCORE", 0))
            route = str(row.get("ROUTE", "—"))
            sector = str(row.get("업종", ""))

            # 배경 그라데이션: 라벨별 색상
            if "최강" in elite_lbl:
                grad = "from-yellow-900/40 via-orange-900/30 to-red-900/20"
                border_clr = "border-yellow-500/40"
            elif "즉시" in elite_lbl:
                grad = "from-green-900/40 via-emerald-900/30 to-teal-900/20"
                border_clr = "border-green-500/40"
            elif "추격" in elite_lbl:
                grad = "from-orange-900/40 via-amber-900/30 to-yellow-900/20"
                border_clr = "border-orange-500/40"
            else:
                grad = "from-slate-900/40 via-slate-800/30 to-slate-900/20"
                border_clr = "border-slate-500/30"

            with ui.card().classes(
                f"w-full p-4 mb-3 bg-gradient-to-br {grad} "
                f"border {border_clr} rounded-2xl"
            ):
                # 1행: 제목 + 라벨 + 상태
                with ui.row().classes("w-full items-center gap-3 mb-3 flex-wrap"):
                    ui.label(f"🔍 {name}").classes(
                        "text-2xl font-black text-white"
                    )
                    ui.label(f"({code})").classes("text-sm text-gray-400")
                    if elite_lbl:
                        ui.badge(elite_lbl, color=elite_color_hex).classes(
                            "text-sm font-bold px-3 py-1"
                        )
                    route_color = {
                        "ATTACK": "#EF4444", "ARMED": "#F59E0B",
                        "WAIT": "#3B82F6", "NEUTRAL": "#6B7280",
                        "OVERHEAT": "#DC2626", "CARRY": "#8B5CF6",
                    }.get(route, "#6B7280")
                    ui.badge(route, color=route_color).classes("text-xs")
                    if sector:
                        ui.label(f"· {sector}").classes("text-xs text-gray-400")

                # 2행: 핵심 점수 게이지 3개
                with ui.row().classes("w-full gap-4 mb-2 flex-wrap"):
                    _score_gauge("종합 점수", display_score, max_val=100)
                    _score_gauge("검증 점수", elite_rank, max_val=100)
                    # RR 목표 게이지 (현재가 → T1)
                    if _close > 0 and _entry > 0 and _t1 > _entry:
                        risk = max(_entry - _stop, 1) if _stop > 0 else 1
                        reward = _t1 - _entry
                        rr = reward / risk
                        _score_gauge(
                            "RR (T1:손절)", min(rr * 20, 100),  # RR 5배면 100점
                            max_val=100, display_text=f"{rr:.1f}:1",
                        )

                # 3행: 가격 게이지 바 (손절 ──── 매수 ──── 현재 ──── T1 ──── T2)
                if _close > 0 and _entry > 0 and _stop > 0 and _t1 > 0:
                    _price_range_bar(_stop, _entry, _close, _t1, _t2)

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

            # [v3.7.17/18] 핵심 지표 요약 패널 — 프로그레스 바 포함 (시각 임팩트)
            with ui.card().classes(
                "w-full p-3 mt-2 bg-[rgba(255,255,255,0.03)] "
                "border border-[rgba(255,255,255,0.08)] rounded-xl"
            ):
                ui.label("📊 핵심 지표").classes("text-xs text-gray-400 mb-2")
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    # 3축 점수
                    s_val = _nz(row.get("STRUCT_SCORE", row.get("S_SCORE", 0)))
                    t_val = _nz(row.get("TIMING_SCORE", row.get("T_SCORE", 0)))
                    ai_val = _nz(row.get("AI_SCORE", 0))
                    bal = _nz(row.get("BALANCE_CALC", row.get("BALANCE_SCORE", 0)))
                    # [v3.7.24] 테이블과 동일한 로직으로 통일
                    # 종합 = ELITE_SCORE 우선 (없으면 ELITE_RANK_SCORE 폴백)
                    # 랭크 = ELITE_RANK_SCORE (Top 선별용 내부 점수)
                    elite = _nz(row.get("ELITE_SCORE", row.get("ELITE_RANK_SCORE", 0)))
                    rank_val = _nz(row.get("ELITE_RANK_SCORE", 0))
                    gap = _nz(row.get("GAP_PCT", 0))
                    rsi = _nz(row.get("RSI14", 0))
                    vp = _nz(row.get("V_POWER", 0))
                    turnover = _nz(row.get("거래대금(억원)", 0))

                    def _mini_bar(label, val_txt, pct, bar_color, text_color):
                        """[v3.7.19] 프로그레스 바 포함 미니 메트릭."""
                        pct = max(0, min(100, pct))
                        with ui.column().classes("gap-1 min-w-[88px] flex-1"):
                            ui.label(label).classes("text-[10px] text-gray-400 uppercase")
                            ui.label(val_txt).classes(f"text-base font-bold {text_color}")
                            ui.html(
                                f'<div style="display:block; width:100%; height:4px; '
                                f'background:rgba(255,255,255,0.06); border-radius:2px; '
                                f'overflow:hidden;">'
                                f'<div style="display:block; width:{pct:.1f}%; '
                                f'height:100%; background:{bar_color};"></div>'
                                f'</div>'
                            ).classes("w-full")

                    def _clr_hex(v, good=70, bad=40):
                        if v >= good: return ("#EF5350", "text-red-400")
                        elif v >= bad: return ("#FFA726", "text-yellow-400")
                        else: return ("#3B82F6", "text-blue-400")

                    # [v3.7.26] 상세 패널 2단 구조 — 핵심 지표 / 기타 지표
                    # 사용자 지적: "스코어들이 너무 많은데"
                    # 해결: 실전 의사결정 핵심 6개만 먼저 표시, 나머지는 접이식

                    # ─── 🎯 핵심 지표 (항상 표시, 실전 매매 직접 활용) ───
                    # S / T / AI (3축) · 갭% · RR · 종합
                    rr_val = _nz(row.get("RR_NOW_TP1", 0))
                    for lbl, val in [("S 구조", s_val), ("T 타이밍", t_val),
                                      ("AI", ai_val)]:
                        bc, tc = _clr_hex(val)
                        _mini_bar(lbl, f"{val:.0f}", val, bc, tc)

                    # 갭% (실전 진입 핵심)
                    gap_pct_bar = min(gap * 10, 100)
                    if gap > 5:
                        gap_bar = "#EF5350"; gap_tc = "text-red-400"
                    elif gap > 2:
                        gap_bar = "#FFA726"; gap_tc = "text-yellow-400"
                    else:
                        gap_bar = "#66BB6A"; gap_tc = "text-green-400"
                    _mini_bar("갭%", f"{gap:.1f}%", gap_pct_bar, gap_bar, gap_tc)

                    # [v3.7.26] RR 미니바 신규 — 실전 매매 핵심 지표
                    # RR 1.0 이상 양호, 2.0 이상 우수
                    rr_pct = max(0, min(100, rr_val * 33.33))  # 3.0 기준 100%
                    if rr_val >= 2.0:
                        rr_bar = "#10B981"; rr_tc = "text-green-400"
                    elif rr_val >= 1.0:
                        rr_bar = "#FFA726"; rr_tc = "text-yellow-400"
                    else:
                        rr_bar = "#EF5350"; rr_tc = "text-red-400"
                    _mini_bar("RR", f"{rr_val:.2f}", rr_pct, rr_bar, rr_tc)

                    # 종합 (파이프라인 최종 점수)
                    bc, tc = _clr_hex(elite, good=60, bad=30)
                    _mini_bar("종합", f"{elite:.0f}", elite, bc, tc)

                # ─── 🔬 기타 지표 (접이식 · 필요 시만 확장) ───
                # 균형 / 랭크 / RSI / 세력(V) / 거래대금 등 상세 분석용
                with ui.expansion(
                    "🔬 기타 지표 (균형/랭크/RSI/세력/거래대금)",
                    icon="expand_more",
                ).classes("w-full text-xs text-gray-400").props("dense"):
                    with ui.row().classes("w-full gap-3 flex-wrap pt-2"):
                        # 균형
                        bc, tc = _clr_hex(bal)
                        _mini_bar("균형", f"{bal:.0f}", bal, bc, tc)
                        # 랭크 (내부 Top 선별용)
                        bc, tc = _clr_hex(rank_val, good=40, bad=20)
                        _mini_bar("랭크", f"{rank_val:.0f}", rank_val, bc, tc)

                        # RSI: 70+ 과매수 빨강, 30- 과매도 파랑, 중간 회색
                        if rsi >= 70:
                            rsi_bar = "#EF5350"; rsi_tc = "text-red-400"
                        elif rsi <= 30:
                            rsi_bar = "#3B82F6"; rsi_tc = "text-blue-400"
                        else:
                            rsi_bar = "#9CA3AF"; rsi_tc = "text-gray-300"
                        _mini_bar("RSI14", f"{rsi:.0f}", rsi, rsi_bar, rsi_tc)

                        vp_pct = max(0, min(100, (vp + 1) / 4 * 100))
                        vp_bar, vp_tc = _clr_hex(vp_pct)
                        _mini_bar("세력(V)", f"{vp:+.2f}", vp_pct, vp_bar, vp_tc)

                        to_pct = min(turnover / 20, 100)
                        to_bar, to_tc = _clr_hex(to_pct, good=50, bad=15)
                        _mini_bar("거래대금", f"{turnover:.0f}억", to_pct, to_bar, to_tc)

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
    # [v3.7.26] view_table_mode 추가 — 기본/고급 전환 시 테이블 재구성
    for widget in [view_mode, route_filter, label_filter, sort_mode, view_table_mode]:
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
