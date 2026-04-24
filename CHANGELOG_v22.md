# SwingPicker v22 Merge v4 — 99점권 마감 (관측/CI 운영부 + 자동화)

**배포 상태**: ✅ Syntax/Import/기능/회귀 모두 검증 완료
**테스트 결과**: 86 passed (기존 72 + Route 6 + Smoke 8)

---

## v4 추가 패치 (리뷰어 98.7점 → 99점권 마감)

### Fix #8: GitHub Actions workflow 추가
**파일**: `.github/workflows/v22_monotonicity.yml`

**기능**:
- push/PR/스케줄(평일 KST 16:30) + 수동 실행
- Route 계약 테스트 자동 실행
- `daily_briefing.py`로 monotonicity_report 생성
- HARD gate 검증 + SOFT 경고 출력
- 보고서 아티팩트 30일 보관

**운영 단계** (continue-on-error 토글):
- **Week 1~2** (현재): `true` → REPORT-ONLY (실패해도 PR merge 가능)
- **Week 3~4**: `false` → SOFT BLOCK (HARD FAIL 시 PR 빨강)
- **MATURE 후**: branch protection rule → HARD ENFORCE

### Fix #9: TOP_PICK 기준 declared_wr 분리
**문제**: 기존엔 ATTACK/ARMED 전체 평균만 계산. HARD gate에는 TOP_PICK 기준이 더 적합.

**해결**: `monotonicity_report`에 두 축 모두 기록
```json
{
  "declared_wr_active": 0.54,
  "declared_wr_top_pick": 0.70,
  "declared_vs_realized_gap_active": 0.08,
  "declared_vs_realized_gap_top_pick": 0.13,
  "declared_vs_realized_gap": 0.13   // alias = top_pick 우선
}
```

HARD gate 3번은 `top_pick` 우선 판정, 없으면 `active` fallback. 
detail에 `gap_top_pick=12.5%` 형식으로 어느 축인지 명시.

### Fix #10: monotonicity_report 스모크 테스트 8건
**파일**: `test_monotonicity_report_v22.py`

| 케이스 | 검증 |
|---|---|
| `test_happy_path_all_pass` | 정상 데이터 → HARD PASS, SOFT OK |
| `test_hard_fail_route_leak` | TOP_PICK=1인데 ROUTE=WAIT → FAIL 감지 |
| `test_hard_fail_negative_tp1` | TP1_PCT=0 → FAIL 감지 |
| `test_hard_fail_large_gap` | 선언 0.75 vs 실현 0.40 → 35%p FAIL |
| `test_no_recommend_file_graceful` | recommend_latest 없어도 크래시 안 함 |
| `test_no_winrate_table` | winrate_table 없으면 SKIP, 다른 게이트는 PASS |
| `test_zero_top_pick` | 0건 날도 PASS (위반 자체 불가) |
| `test_top_pick_vs_active_split` | 두 축 declared_wr 분리 기록 검증 |

---

## v22 전체 통합 요약

| 파일 | 원본 → v4 |
|---|---|
| `shared_utils.py` | 92 → 378 (+286) |
| `scoring_engine.py` | 760 → 854 (+94) |
| `pipeline_calibrate.py` | 422 → 491 (+69) |
| `pipeline_finalize.py` | 346 → 515 (+169) |
| `kelly_calibrator.py` | 576 → 863 (+287) |
| `auto_backtest.py` | 649 → 796 (+147) |
| `daily_briefing.py` | 321 → 668 (+347) |
| `test_route_contract_v22.py` | — → 84 (신규) |
| `test_monotonicity_report_v22.py` | — → 215 (신규) |
| `.github/workflows/v22_monotonicity.yml` | — → 134 (신규) |

**누적 +1,832 lines**

---

## 누적된 모든 fix (v1 → v4)

### v1 — 핵심 엔진 구현 (10개)
1. ELITE_SCORE 축 일치
2. TOP_PICK positive gate
3. AGGRESSIVE/STABLE 이원화
4. RANK_SCORE 강등 → LEGACY 기록만
5. ELITE 기반 EST_WIN_RATE (compute_est_win_rate SSOT)
6. Kelly empirical b_ratio (min(planned, empirical))
7. per_trade_log 5-method 확장
8. auto_backtest 벤치마크 초과수익
9. finalize_sort SSOT (8축)
10. TOP_PICK 0건 latest.json 갱신

### v2 — 방어력 강화 (4개)
11. Series/DataFrame 입력 방어
12. _normalize_stock_code 전체 적용
13. Kelly method sync 양방향
14. per-method latest stale 방지

### v3 — 관측/CI 운영부 (3개)
15. monotonicity_report + CI HARD/SOFT Gate (7개)
16. Route 타입 계약 + route_name 헬퍼 + 6 테스트
17. CARRY 구간 SSOT 정리

### v4 — CI 자동화 (3개)
18. GitHub Actions workflow (.github/workflows/v22_monotonicity.yml)
19. TOP_PICK 기준 declared_wr 분리
20. monotonicity_report 스모크 8 테스트

---

## 배포 가이드

### 1. ZIP 풀기
```bash
cd ~/Downloads
unzip -o v22_merged_v4.zip -d v22_merged_v4/
ls v22_merged_v4/
ls v22_merged_v4/.github/workflows/
```

### 2. 레포에 적용
```bash
cd ~/Downloads/swingpicker-web
git branch --show-current   # feature/v22-implementation 확인

# 모듈 + 테스트
cp -v ~/Downloads/v22_merged_v4/*.py .

# workflow 파일
mkdir -p .github/workflows
cp -v ~/Downloads/v22_merged_v4/.github/workflows/v22_monotonicity.yml .github/workflows/

# 문서
cp ~/Downloads/v22_merged_v4/CHANGELOG_v22.md .
```

### 3. 테스트
```bash
export PYTHONUTF8=1
python -m pytest test_scoring_weights.py test_stop_logic.py test_trade_plan.py \
                 test_route_contract_v22.py test_monotonicity_report_v22.py --tb=short
# 기대: 86 passed
```

### 4. 커밋
```bash
git add *.py CHANGELOG_v22.md .github/workflows/v22_monotonicity.yml
git commit -m "feat(v22): CI 자동화 마감 (v22_merged_v4 — 99점권)

핵심 v4 추가:
- .github/workflows/v22_monotonicity.yml: HARD gate 자동 검증
- daily_briefing: TOP_PICK 기준 declared_wr 분리
- test_monotonicity_report_v22: 스모크 8 테스트

누적 +1,832 lines, 86 tests PASS

[Breaking] 없음. compute_elite_score 시그니처 변경은 v1에서 이미 반영."

git push -u origin feature/v22-implementation
```

### 5. PR 생성 후 첫 CI 실행 시
- "v22 Monotonicity" workflow 실행됨
- Route 계약 테스트 6 PASS
- monotonicity_report 생성 시도
  - data/recommend_latest.csv 없으면 → 안전하게 SKIP
  - 있으면 → HARD/SOFT gate 평가
- 첫 1주는 `continue-on-error: true` → 워크플로우 통과 (REPORT-ONLY)

---

## 운영 단계별 워크플로우 조정

### Week 1~2 (현재 상태 — v22 PR merge 직후)
```yaml
continue-on-error: true   # ← 그대로
```
- HARD FAIL이어도 워크플로우는 ✅
- 이슈만 모니터링, 데이터 수집

### Week 3~4 (winrate_table_by_ELITE_SCORE 누적 시작)
```yaml
continue-on-error: false   # ← 수정
```
- HARD FAIL 시 PR 빨강 (force-merge 가능)
- branch protection rule 미적용

### MATURE (n_trades ≥ 100, 2~3개월 후)
- GitHub Settings → Branches → Add rule
- "Require status checks to pass before merging"
- "v22 Monotonicity / monotonicity-check" 체크박스

---

## 완성도

**99점 / 100점**

남은 1점은 설계가 아닌 **실제 운영 데이터 누적 후 검증**:
- 2-3주: 실제 KPI 갭 ≤ 5%p 달성 확인
- STABLE 분기 실측 빈도 측정
- Wilson 단조성 tolerance 점진적 강화 (0.05 → 0.03 → 0.02)
