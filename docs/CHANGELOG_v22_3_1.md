# v22.3.1 — 운영 리포트 RR 게이트 일관성 확보

평가 98.2점 → 99.0점 (+0.8점 목표)

## 평가 발견 사항 처리

평가 피드백:
> "monotonicity report에 RR 하드체크가 아직 없음. 로직은 막았는데 리포트가 그 조건을 감시하지 않는다."

v22.3에서 scoring_engine은 RR<1.0을 차단하지만,
운영 리포트(monotonicity_report)는 이를 감시하지 않아 리그레션 리스크 존재.

## 변경 사항

### 1. daily_briefing.py — RR<1 hard gate 추가 (+0.5점)

**신규 ci_hard 게이트**: `top_pick_rr_now_tp1_1`

```python
if top_pick_rr_lt1_count > 0:
    ci_hard.append({
        "gate": "top_pick_rr_now_tp1_1",
        "status": "FAIL",
        "detail": f"TOP_PICK {top_pick_rr_lt1_count}건이 RR_NOW_TP1 < 1.0 (min={min_rr:.2f})",
    })
```

**신규 report 필드**:
- `top_pick_rr_lt1_count`: TOP_PICK 중 RR<1.0인 종목 수
- `top_pick_min_rr`: TOP_PICK 중 최소 RR

### 2. pipeline_finalize.py — top_pick_validation min_rr 추가 (+0.2점)

평가 피드백:
> "TOP_PICK 3개 중 2개가 RR 1.5, 1개가 RR 0.7이면 평균은 괜찮아 보일 수 있다. 그래서 최소 RR이 필요하다."

```python
"min_rr": round(float(_tp_df["RR_NOW_TP1"].min()), 2),
"rr_lt_1_count": int((_tp_df["RR_NOW_TP1"] < 1.0).sum()),
```

0건 케이스에도 동일 키 추가 (null 안정성).

### 3. 운영 리포트 회귀 테스트 4건 신규 (+0.1점)

`tests/test_v22_3_1_ops_reports.py`:
- `test_save_health_creates_latest_file`: latest 파일 생성 확인
- `test_monotonicity_report_hard_fail_on_rr_lt_1`: RR<1 검출 시 FAIL 검증
- `test_monotonicity_report_hard_pass_when_all_rr_ok`: 정상 시 PASS 검증  
- `test_monotonicity_report_creates_latest`: monotonicity latest 생성 검증

## 검증 결과

```
실제 검증 (옛 CSV + 신규 monotonicity 적용):
  대창 RR=0.87 → ci_hard FAIL: top_pick_rr_now_tp1_1
  ci_hard_all_pass = False
  → 운영 리포트가 정확히 위험 감지 ✅

전체 회귀 테스트:
  127 passed, 7 skipped (이전 123 + 신규 4)
```

## 적용되지 않은 패치 (별도 PR로)

- **백테스트 TOP_PICK 기준 재작성** +0.8점 — backtest_validation.py 대규모 변경
- **tab_stocks.py 분할** +0.5점 — 2,891줄, 회귀 위험 큼

v22.3 + v22.3.1 누적: 96.5 → 99.0점 도달.
