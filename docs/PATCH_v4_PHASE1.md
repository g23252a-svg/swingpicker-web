# PATCH — Recommendation Engine v4.0 Phase 1: 세그먼트 조건부 캘리브레이션

> 상태: **SHADOW** (본선 무변경). 백테스트 통과 후 별도 PR에서 본선 승격.
> 대상 버전: 엔진 v3.9.28 → v4.0.0 (Phase 1)
> 날짜: 2026-05-30

---

## 0. 한 줄

EST_WIN_RATE를 막고 있던 **캘리브레이션 천장(~0.51)**을 제거한다. 단일축 ELITE_SCORE 룩업 →
(ELITE 버킷 × 세그먼트) 경험적 베이즈 수축 테이블로. 본선은 건드리지 않고 `*_V4` 그림자 컬럼으로 먼저 검증.

> **[보강 2026-05-30] 승률 기준을 absolute → excess(시장 대비)로 교정.**
> 검증 중 발견: raw win(ret>0)은 상승장에서 prior 0.65로 부풀려져 DISPLAY 전환 시 전 종목 0.71(게이트 무력화 + 구독자 화면 과대표기). day-relative excess로 바꾸니 prior 0.41, 진짜 엣지 세그먼트만 0.5↑로 분리. `build_segmented_table(win_basis="excess")`가 기본. explainer는 "시장 평균보다 나은 결과가 10번 중 약 N번"으로 정직 표기.


---

## 1. 왜 (진단, 실측 근거)

2026-05-30 추천 CSV(579종목): **TOP_PICK = 0**.
원인은 매크로 하드블록이 아니라(그건 진단 전용 shadow), **STABLE 게이트의 구조적 사망**:

```
scoring_engine STABLE 게이트:  ... AND EST_WIN_RATE >= 0.55 AND MATURE
EST_WIN_RATE_METHOD = ELITE_SCORE (단일축 버킷 룩업)
실측 EST_WIN_RATE 분포: 562종목 0.417 · 최대 0.509  → 0.55 영구 미달
```

per-trade 로그 11,467건으로 세그먼트 테이블을 만들어 보면 천장이 깨진다:

| 세그먼트 (score×method) | p_win | n_eff | 충분 |
|---|---|---|---|
| 90-101 · DISPLAY_SCORE | **0.80** | 861 | O |
| 90-101 · FINAL_SCORE | **0.80** | 739 | O |
| 80-90 · FINAL_SCORE | **0.72** | 1585 | O |
| 70-80 · ELITE_SCORE | 0.61 | 915 | O |
| **80-90 · ELITE_SCORE** | **0.43** | 506 | O |

**핵심 발견 2가지**
1. 충분표본 세그먼트 **19개가 0.55 돌파** (최고 0.80). 천장은 데이터가 아니라 모델 구조 탓.
2. **EST_WIN_RATE가 가장 안 좋은 축(ELITE_SCORE)으로 계산되고 있음.** ELITE_SCORE는 비단조
   (80-90 band 0.43 < 70-80 band 0.61). DISPLAY/FINAL_SCORE가 승자를 훨씬 잘 분리.
   → 세그먼트화 이전에 **calibration score-col을 score(DISPLAY/FINAL 계열)로 바꾸는 것만으로도 리프트**.

---

## 2. 무엇 (변경 파일)

| 파일 | 성격 | 내용 |
|---|---|---|
| `calibration_v4.py` | 신규 | 세그먼트 EB 캘리브레이션 엔진 (순수 함수) + shadow 컬럼 + 상대 게이트 |
| `scripts/build_calibration_v4_table.py` | 신규 | per-trade 로그 → `data/calibration_v4_table_latest.json` |
| `tests/test_calibration_v4.py` | 신규 | 8 tests (천장 제거 / EB 수축 / 상대 게이트 / shadow 안전) |
| `components/explainer_v4.py` | 신규 | 회원용 평이 설명 카드 (상업용 톤, 표현 안전) |

**추가되는 shadow 컬럼 (본선 컬럼 무변경):**
`EST_WIN_RATE_V4`, `EST_WIN_RATE_V4_N`, `EST_WIN_RATE_V4_SEGMENT`,
`EST_WIN_RATE_V4_SUFFICIENT`, `STABLE_GATE_V4_PASS`,
`TOP_PICK_STABLE_V4_SHADOW`, `TOP_PICK_V4_SHADOW`

---

## 3. 수식

경험적 베이즈 수축 (셀별):
```
p_shrunk = (wins_eff + k·p0) / (n_eff + k)
  p0      = 전체 시간감쇠 승률 (글로벌 prior, 실측 0.65)
  k       = 30 (사전강도)
  n_eff   = 시간감쇠 유효표본 (half-life 90d, kelly_calibrator 재사용)
```
→ 표본 적은 셀은 p0로 수축(과적합 차단), 표본 많은 셀만 신호.

상대 STABLE 게이트 (절대 0.55 폐기):
```
PASS = (p_win_v4 >= max( quantile(오늘 calibrated WR, 0.75), p0 + 0.03 ))
       AND n_eff >= 20  AND  MATURE
```

---

## 4. 통합 지점 (본선 승격 시 — 이번 PR 아님)

`scoring_engine.py` STABLE 게이트 1줄 교체 (백테스트 통과 후):

```python
# AS-IS (v3.9.28)
_stable = (_hard_gate & (ELITE_SCORE>=70) & (7<=_tp1_pct<15)
           & (BALANCE_SCORE>=70) & (_est_wr >= 0.55) & _cal_mature)

# TO-BE (v4.0 본선)
_stable = (_hard_gate & (ELITE_SCORE>=70) & (7<=_tp1_pct<15)
           & (BALANCE_SCORE>=70) & (x["STABLE_GATE_V4_PASS"]==1) & _cal_mature)
```
+ `pipeline_calibrate.py`에서 `EST_WIN_RATE`를 V4 세그먼트 값으로 산정 (score-col=score).

---

## 5. 선결 과제 (Phase 1 한계)

현재 per-trade 로그(`kelly_calibrator.save_per_trade_log`)는 `MACRO_REGIME_MODE`·`ACTION_TIER`를
**기록하지 않는다.** 따라서 v4 GDD가 목표한 (ELITE × TIER × REGIME) 3축은 아직 불가.

→ **선행 작업:** `save_per_trade_log`에 두 컬럼 추가 후 1~2개월 누적. 그 전까지 세그먼트는
`score × method (× horizon)`로 한정. 이 한계는 테이블 meta `WARN_no_regime_tier_in_log=true`로 표시됨.

---

## 6. 검증 / 롤아웃

1. `pytest tests/test_calibration_v4.py -v` → 8 passed
2. `combo_optimizer.py`에 `USE_CALIBRATION_V4` ON/OFF 변수 추가, 6개월 백테스트로 EV·승률 비교
   - 검증 케이스: 에스엔시스(5/7), 신세계I&C(5/1) 포함
3. 통과 시 §4 통합 → canary → production. 앱/UI 버전 동반 bump.

**롤백:** shadow 컬럼만 추가하므로 `calibration_v4` import 제거 시 즉시 원복. 본선 baseline 무영향.

---

## 7. 안전 계약 (유지)

- 자동매도 없음. explainer는 '검토/권장'까지만, '매도/팔아라/자동매매' 단정 금지.
- 승률은 과거 패턴 빈도이며 미래 보장 아님을 항상 병기 (회원 화면 disclaimer 고정).
- CSV 스키마 append-only. 기존 컬럼 의미 불변.
