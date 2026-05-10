# SwingPicker Update Notes — 2026-05-10

> v22.3.2 핫픽스 4종 + v23 추천 엔진 재설계 시작 (Phase 0 + Phase 1 PR-A)

**작업 일자**: 2026-05-10 (일)
**브랜치**: `release/v22.3` (모두 push 완료)
**커밋 수**: 8개

---

## 한 눈에 — 8개 커밋 요약

```
dfeacdb  feat(v23): PR-A — V23_* 신호 attach (96점, 머지 가능)
ce39de4  docs(v23): Phase 0 완료, 다음 단계 bookmark
f74a7f4  docs(v23): 추천 엔진 4-tier 재설계 GDD v0.3
25a3d0e  chore: ignore *.before_* / *.bak local backup files
a049aac  fix(monotonicity): logger의 dangling 'gap' 참조 제거
d05b7a7  merge: main의 최신 collector 데이터 + 동기화 (162 파일)
dd477dd  fix(monotonicity): v22.3.2 declared/realized 모집단 일치
a41c73f  fix(db): v22.3.2 inquiries v2 마이그레이션 + Gist 컬럼 밀림 핫픽스
```

분류:
- **운영 핫픽스**: 4건 (db / monotonicity / data merge / NameError)
- **운영 정리**: 1건 (백업파일 정리)
- **v23 추천 엔진**: 3건 (GDD + bookmark + Phase 1 PR-A)

---

## 1. v22.3.2 — DB 핫픽스 (a41c73f)

### 문제
- inquiries 테이블에 `None / None` 문의 160건 자동 증식
- Gist 업로드에서 컬럼 밀림 (SELECT * + ALTER TABLE 후 컬럼 순서 어긋남 → JSON 통째 밀림)

### 해결 (5개 영역)
1. **inquiries v2 마이그레이션** — 5컬럼 → 10컬럼 in-place, `schema_versions` 가드
2. **`inquiry_id` UNIQUE INDEX** (부분, NULL 허용)
3. **멱등화** — `_insert_gist_inquiries`에 INSERT OR IGNORE + 빈글 차단
4. **Gist hotfix** — `_do_gist_upload`의 `SELECT *` → 명시 컬럼 SELECT 변경
5. **"None"/"null"/"nan"/"-" 문자열 차단** — `_clean_inquiry_text` 신규
6. **fast path 정리** — `_cleanup_invalid_inquiries` 매 재시작 실행
7. **add_inquiry 60초 자연 중복 차단**
8. **누락 메서드 5개 구현** — get_user_inquiries, get_inquiry_stats, update_inquiry_reply, delete_inquiry, verify_inquiries_health
9. **UTC 통일 4곳** — register_user / record_login_failure / record_payment / grant_all_users_trial
10. **`save_inquiries(force=False)` 가드** 추가

### 영향
- 운영 게시판 안전성 회복
- Gist 데이터 무결성 보장 (컬럼 순서 의존 제거)

---

## 2. v22.3.2 — Monotonicity CI 정상화 (dd477dd → a049aac)

### 문제
CI Monotonicity HARD gate가 영구 FAIL 상태.

원인:
- `declared_wr_top_pick` (오늘 TOP_PICK 평균, ELITE_SCORE 79+) 와
  `realized_wr` (전체 ELITE_SCORE bin 가중평균) 비교 → 영구 양수 gap
- `winrate_table.meta.is_sufficient=false` (n_trades=41 < min_n) 인데도 HARD gate 발화
- TOP_PICK n=2 + 매칭 bin n=21 절대값 비교 → Wilson CI 안에 들어감

### 해결
1. **`_realized_wr_for_score_range(table, lo, hi)` 헬퍼** — declared 모집단의 ELITE_SCORE 범위에 매칭되는 bin만 가중평균
2. **`realized_wr_top_pick` / `realized_wr_active` 신규 키** — declared 모집단과 같은 점수 범위에서 계산, n도 함께 기록
3. **gap 모집단 일치** — `gap_top_pick` = declared_top_pick - realized_top_pick (matched), 호환 alias 유지
4. **HARD 3 게이트 표본 가드** — `MIN_N_FOR_HARD_GAP = 30` 미달 시 HARD SKIP
5. **active fallback 강등** — TOP_PICK 전용 HARD, active는 SOFT WARN으로 분리 (declared_wr_active가 fallback 상수 0.539 위주라 HARD 비교 부적절)
6. **NameError 픽스 (a049aac)** — logger의 dangling `gap` 참조를 `report.get('declared_vs_realized_gap')` alias로 정정

### 검증
- v22 Monotonicity #25 GREEN 확인 (43s, May 10 14:30 KST)
- HARD gate: ROUTE / TP1 / RR_NOW_TP1 / declared_vs_realized_gap_15pp 모두 PASS 또는 SKIP
- SOFT WARN: avg_ret_excess_positive (-5.12%), wilson_monotonicity, declared_vs_realized_gap_active (21.8%, n=316) — 모니터링용

---

## 3. v22.3.2 — main 데이터 동기화 (d05b7a7)

### 문제
release/v22.3 브랜치가 May 1 데이터에 머물러 있어 monotonicity 워크플로우가 옛 데이터로 평가
- 5월 3일 v22.3.1 RR hard gate (`RR_NOW_TP1 >= 1.0`) 머지됨
- 그러나 release/v22.3의 csv는 5월 1일 (RR hard gate 적용 전)
- 결과: 5월 1일 csv의 TOP_PICK에 RR=0.87 종목이 포함되어 게이트 자연 fail

### 해결
`git merge origin/main --no-ff` — May 4-8 데이터 + ldy_trader.db 일제 동기화 (162 파일).

### 영향
- 워크플로우 재실행 시 May 8 csv 사용 → RR hard gate 자연 통과
- positions / per_trade_log / monotonicity_report 모두 최신화

---

## 4. 운영 정리 (25a3d0e)

### 변경
- `components/*.before_*` 백업파일 6개 삭제
- `.gitignore`에 패턴 추가:
  ```
  # Local backup files
  *.before_*
  *.bak
  ```

### 영향
- 향후 자동 무시 — `git status`에 더 이상 안 잡힘

---

## 5. v23 추천 엔진 재설계 — Phase 0 (f74a7f4 + ce39de4)

### 배경
기존 v22.x 추천 시스템의 한계:
- 단일 점수(DISPLAY_SCORE/ELITE_SCORE) 정렬 → 추격 매수 유도
- "내일 갈 종목"과 "1-2주 후 폭발할 종목" 미분리
- 윗꼬리/거래량 클라이맥스/휩쏘 위험 무시
- Top 3 강제 → 진짜 좋은 후보 0건이어도 채워서 표시

### v23 4-Tier 분류 도입
```
🟢 NOW_BUY              — 즉시진입 (trend onset, 5d horizon)
🟡 ACCUMULATION_READY   — 매집 후보 (compression before expansion, 14d horizon)
🟠 PULLBACK_WAIT        — 좋지만 현재가 비쌈 (눌림 대기)
🔴 NO_CHASE             — 추격 금지 (chase risk + whipsaw)
⬜ BLOCKED              — 수식 무결성 실패 (UI 비표시)
```

**TOP_PICK은 직교(orthogonal) 품질 배지** — 액션 등급 아님.

### 임계값 분류 (A/B/C/D)
- **A** 운영 검증값 재사용: RR≥1.0, turnover≥50억, EBS PASS 등
- **B** 수식 무결성: TP1 > CURRENT_PRICE > STOP
- **C** 분포 기반 동적 percentile (당일 active)
- **D** 백테스트 결정 필요 — **5개로 압축**

### Phase 5 진입 조건 (6개)
- **절대**: NOW_BUY 5d wr ≥ 50%, ACCUMULATION 14d wr ≥ 55%, NO_CHASE EV +1%p
- **상대**: NOW_BUY 5d EV ≥ active baseline +1%p, ACCUMULATION 14d EV ≥ baseline +1%p
- **무결성**: tier H1/H2/H3 PASS

**미달 시 분류 자체 폐기.** 자기 가설을 데이터로 reject 할 준비.

### 도입 로드맵 (6주, Phase 5까지 사용자 노출 0)
| Phase | 기간 | 작업 | 위험 |
|---|---|---|---|
| 0 | 0주 | GDD + 컬럼 매핑 분석 ✅ | 0 |
| 1 | 1주 | scoring_engine 신규 컬럼 추가 (PR-A 진행 중) | 0 |
| 2 | 1주 | 4-tier 컬럼 추가 (log only) | 0 |
| 3 | 1주 | monotonicity v23 확장 | 0 |
| 4 | 2주 | 백테스트 — D 카테고리 5개 결정 | 0 |
| 5 | 1주 | UI 4-tier 노출 | 중 |
| 6 | 지속 | fine-tune | 저 |

### 산출물
- `docs/RECOMMENDATION_ENGINE_v23_GDD.md` (517줄, 14 섹션, v0.3)
- `docs/RECOMMEND_v23_COLUMN_MAPPING.md` (199줄, Phase 0 분석)
- `docs/RECOMMEND_v23_NEXT_STEP.md` (bookmark)

### 핵심 발견 (Phase 0 분석)
- `ENTRY_GAP_PCT` 이미 현재가 기준 — derive 불필요
- `Range_Pos` = `close_location` 본질 동일
- ATR / OBV / Upper_Shadow_Ratio 함수는 indicators.py에 있지만 **csv에 컬럼 미저장**
- `per_trade_log.csv` n=271,160 — **백테스트 인프라 즉시 사용 가능** (Phase 4)

---

## 6. v23 Phase 1 PR-A — V23_* 신호 attach (dfeacdb)

### 목적
ATR_Pct / OBV_Slope / Upper_Shadow_Ratio 3개 신호를 `recommend_latest.csv`에 노출.
**기존 추천 로직 한 줄도 안 변경**, **사용자 노출 0**.

### 핵심 발견
ml_engine.py의 `add_technical_features` (line 296)와 `add_technical_features_batch` (line 386)가 이미 위 3개 + 16개 ML feature를 계산. 다만:
- 학습 path는 **single 함수** 사용 (`build_master_dataset:865`)
- 추론 path는 **batch 함수** 사용 (`apply_ml_score:1292`)
- **두 함수 계산식 일부 불일치 발견** (MACD_Hist_Norm / Vol_Ratio_5 / OBV_Slope) → **PR-B 별건 작업**

### 구현
`apply_ml_score` 진입부에 `attach_v23_phase1_signals()` helper 호출 — **학습 코드와 동일한 single 함수 사용**으로 parity 버그 회피.

### 9개 안전 요구사항 모두 충족
1. helper 함수 분리 (`attach_v23_phase1_signals`)
2. ML 모델 로드 실패해도 attach 시도 (모델 체크 전 호출)
3. hardcoded index 제거 — `last_row.get('ATR_Pct')` 이름 기반
4. **V23_ prefix 사용** — 기존 ATR_Pct/OBV_Slope/Upper_Shadow_Ratio 보호
5. `V23_SIGNAL_STATUS` (OK / NO_OHLCV / SHORT_HISTORY / FEATURE_FAIL)
6. coverage 로그 — `[v23 Phase 1] signal coverage: X%`
7. read-only — feature_cache 미건드림
8. **종목코드 zfill(6) 양방향 정규화** — 12345 ↔ 012345 매칭
9. unused helper 정리

### 결과
- `recommend_latest.csv` 컬럼: 182 → 186 (+4)
  - `V23_ATR_Pct`, `V23_OBV_Slope`, `V23_Upper_Shadow_Ratio`, `V23_SIGNAL_STATUS`
- 기존 추천 로직 변경: 0
- ML_SCORE 영향: 0 (별도 add_technical_features 호출, ML 추론 path 분리)
- 사용자 노출: 0 (Phase 5까지 log only)

### 트레이드오프
- v23용 single 함수 + ML용 batch 함수 → 같은 OHLCV에 대해 일부 중복 계산
- PR-A 안전성 우선이라 의도된 trade-off
- PR-B에서 parity 정상화 후 통합 가능

### 검증 (다음 collector run 후 예정)
- coverage 90%+ 기대
- V23_ATR_Pct: 0.005 ~ 0.10 (변동성 0.5-10%)
- V23_OBV_Slope: -2.0 ~ 2.0 (clip 범위)
- V23_Upper_Shadow_Ratio: 0.0 ~ 1.0

---

## 다음 작업 큐

### Priority A — 필요 시
- **PR-B (별건 큰 작업, ML 입력 정합성 복구)**
  - `add_technical_features` (single) vs `add_technical_features_batch` (batch) 계산식 통일
  - 차이 항목: MACD_Hist_Norm (×100 누락), Vol_Ratio_5 (log1p 누락), OBV_Slope (정의 자체 다름)
  - shadow A/B 비교 필요 — Top20 교체율, AI_SCORE 변화, TOP_PICK 변화, EST_WIN_RATE 변화
  - calibration 재측정 또는 모델 재학습 검토
  - **paid 구독자 신호 영향 가능** — 신중

### Priority B — Phase 진행
- **PR-C (Phase 2)** — V23_* 데이터 1주 누적 후 4-tier 분류 컬럼 추가 (log only)
- **PR-D (Phase 3)** — monotonicity_report v23 확장 (tier_calibration with horizon split)
- **PR-E (Phase 4)** — 백테스트로 D 카테고리 임계값 5개 결정
- **PR-F (Phase 5)** — UI 4-tier 노출 (사용자 영향 시작)

### Priority C — UX 개선 (low risk, 시간 날 때)
- daily_briefing.py: "AI 점수" → "종합점수" 문구 정정
- daily_briefing.py: "107종목" → 동적 `len(df)`
- `_norm_route()` / `_is_top_pick()` 헬퍼 (입력 방어성)
- positions.json 구조 확인 후 CLOSED_TP 날짜 필터

---

## 변경 이력

- 2026-05-10 v0.1 — 초안 작성
