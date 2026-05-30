# 설계 — combo_optimizer에 USE_CALIBRATION_V4 + TP1_BAND 백테스트 끼우기

> 목적: v4 excess 캘리브레이션 + STABLE의 TP1 밴드 완화가 **공식 추천(TOP_PICK) 포트폴리오**의
> EV·승률·개수·MDD를 어떻게 바꾸는지 6개월 백테스트로 측정. 본선 승격 판단 근거.
> 성격: **read-only 백테스트.** scoring_engine / 라이브 추천식은 건드리지 않는다.

---

## 0. 핵심 결정 — 별도 격자 (기존 128콤보와 분리)

`_all_combos()`(S/T/AI/ROUTE 128콤보)에 **넣지 않는다.** 이유:

| | 기존 `_all_combos` | 새 `_v4_gate_combos` |
|---|---|---|
| 탐색 질문 | "어떤 raw 임계값이 EV 최대?" | "v4 + TP1 밴드가 공식픽을 개선하나?" |
| 대상 | 모든 active 종목 | STABLE/공식 TOP_PICK 경로 |
| 측정 | win/ev | + 공식픽 개수, 보류 현금효과, MDD |

S/T/AI는 STABLE 내부에 이미 고정 포함(ELITE≥70 등)이라 재탐색은 중복. 또 128×2×N으로 격자 폭증.
→ **`USE_CALIBRATION_V4` + `TP1_BAND`는 한 격자에 묶되, S/T/AI 격자와는 분리된 새 경로**로 둔다.
(사용자 요구 "TP1 밴드도 같은 격자에" = V4·TP1을 한 격자에 함께 — 충족. S/T/AI와 합치라는 뜻은 아님.)

---

## 1. 끼우는 위치 (combo_optimizer.py, 줄 기준)

```
[42–105]   _load_trade_rows()            ← (A) 컬럼 확장 (STABLE 재현용 필드 추가)
[106–143]  _evaluate_combo()             ← 그대로 (재사용)
[144–157]  _all_combos()                 ← 그대로
... 기존 함수들 ...
[608 끝]   ↓ 여기에 새 섹션 append:
           ── v4 Official-Pick Gate Backtest (신규) ──
           _v4_gate_combos()
           _simulate_official_pick()
           _evaluate_gate_combo()
           run_v4_gate_backtest()        ← IS/OOS 분할은 기존 wf 로직 재사용
```

새 함수는 전부 **append-only**. 기존 함수 시그니처/본문 무변경 → 회귀 위험 0.

---

## 2. (A) `_load_trade_rows` 컬럼 확장

현재 산출: `ret, win, S, T, AI, ROUTE, SCORE(=DISPLAY), trade_date`.
STABLE을 재현하려면 행마다 추가 적재:

```python
"code":        r["종목코드"],
"ELITE":       float(r.get("ELITE_SCORE", 0) or 0),
"BALANCE":     float(r.get("BALANCE_SCORE", 0) or 0),
"TP1_PCT":     float(r.get("TP1_PCT", 0) or 0),
"RR":          float(r.get("RR_NOW_TP1", 0) or 0),
"turnover":    float(r.get("거래대금(억원)", 0) or 0),
"entry_gap":   abs(float(r.get("ENTRY_GAP_PCT", r.get("GAP_PCT", 99)) or 99)),
"MATURE":      str(r.get("CALIBRATION_MODE", r.get("EST_WIN_RATE_MODE",""))).upper() == "MATURE",
"EST_WR":      float(r.get("EST_WIN_RATE", 0) or 0),     # 베이스라인 STABLE WR
"TOP_PICK":    1 if str(r.get("TOP_PICK","0")) in ("1","1.0","True") else 0,
"TOP_PICK_TYPE": str(r.get("TOP_PICK_TYPE","")),          # ★ AGGRESSIVE 재사용 핵심
"DISPLAY":     float(r.get("DISPLAY_SCORE", 0) or 0),     # v4 lookup_col
```

> **핵심 트릭:** AGGRESSIVE는 재유도하지 않고 저장된 `TOP_PICK_TYPE=="AGGRESSIVE"`를 그대로 쓴다.
> 재시뮬 대상은 **STABLE 분기 하나**뿐 → 충실도↑, 코드↓.

(별도 함수 `_load_trade_rows_v4()`로 두거나, 기존에 컬럼만 추가. 컬럼 추가가 기존 호출에 무해하므로 후자 권장.)

---

## 3. 공식픽 시뮬레이터 `_simulate_official_pick`

```python
from calibration_v4 import score_segment, relative_stable_gate, _build_lookup

def _simulate_official_pick(df, use_v4: bool, tp1_band, v4_table=None):
    """주어진 설정에서 'TOP_PICK이었을' 불리언 Series 반환.
       AGGRESSIVE = 저장된 결정 / STABLE = 설정대로 재유도."""
    lo, hi = tp1_band
    aggressive = df["TOP_PICK_TYPE"].eq("AGGRESSIVE")           # 저장값 재사용

    # STABLE 구조 전제 (TP1 밴드만 가변)
    stable_struct = (
        (df["ELITE"] >= 70) & (df["TP1_PCT"] >= lo) & (df["TP1_PCT"] < hi)
        & (df["BALANCE"] >= 70) & df["MATURE"]
        & (df["RR"] >= 1.0) & (df["turnover"] >= 50) & (df["entry_gap"] <= 5.0)  # hard gate
    )

    if use_v4 and v4_table is not None:
        # v4 excess WR + 상대 게이트
        res = df.apply(lambda r: score_segment(r, v4_table), axis=1, result_type="expand")
        p_v4, n_v4 = res[0].astype(float), res[1].astype(float)
        _, _, p0 = _build_lookup(v4_table)
        wr_gate = relative_stable_gate(p_v4, n_v4, p0, mask=df["MATURE"])
    else:
        wr_gate = df["EST_WR"] >= 0.55                          # 베이스라인 절대 게이트

    stable = stable_struct & wr_gate
    return (aggressive | stable)
```

`v4_table` = `data/calibration_v4_table_latest.json` 로드해서 주입.
**주의:** v4 테이블은 미래참조 방지를 위해 *백테스트 시점 이전* 데이터로 빌드돼야 함(§6 참고).

---

## 4. 평가기 `_evaluate_gate_combo`

기존 `_evaluate_combo` 지표 + 공식픽 전용 지표:

```python
def _evaluate_gate_combo(df, use_v4, tp1_band, v4_table, baseline_sel):
    sel = _simulate_official_pick(df, use_v4, tp1_band, v4_table)
    sub = df[sel]
    n = int(sel.sum())

    # 일자별 시장평균 → excess 승률 (v4 캘리브레이션과 동일 기준)
    day_mean = df.groupby("trade_date")["ret"].transform("mean")
    wr_abs = (sub["ret"] > 0).mean() * 100 if n else 0
    wr_exc = (sub["ret"] > day_mean[sel]).mean() * 100 if n else 0

    avg_ret = sub["ret"].mean() if n else 0
    wins, losses = sub[sub.ret>0]["ret"], sub[sub.ret<=0]["ret"]
    aw, al = (wins.mean() or 0), abs(losses.mean() or 0)
    p = wr_abs/100
    ev = p*aw - (1-p)*al

    # MDD: 일자순 누적수익 곡선
    eq = sub.sort_values("trade_date")["ret"].cumsum()
    mdd = float((eq.cummax() - eq).max()) if n else 0

    # 보류 현금효과 (vs baseline 선택집합)
    dropped = df[baseline_sel & ~sel]["ret"]   # 베이스라인은 샀는데 이 설정은 보류
    added   = df[~baseline_sel & sel]["ret"]   # 이 설정만 새로 매수
    avoided_loss = -dropped[dropped < 0].sum() # 보류로 피한 손실(+가 이득)
    missed_gain  = -dropped[dropped > 0].sum() # 보류로 놓친 수익(-가 손해)
    delta_pnl    = added.sum() - dropped.sum() # 순 현금효과 vs baseline

    return dict(use_v4=use_v4, tp1_band=list(tp1_band), n=n,
                win_rate=round(wr_abs,1), win_rate_excess=round(wr_exc,1),
                avg_ret=round(avg_ret,2), ev=round(ev,2), mdd=round(mdd,2),
                avoided_loss=round(avoided_loss,2), missed_gain=round(missed_gain,2),
                delta_pnl_vs_baseline=round(delta_pnl,2),
                n_dropped=int((baseline_sel & ~sel).sum()),
                n_added=int((~baseline_sel & sel).sum()))
```

---

## 5. 격자 + 드라이버 `run_v4_gate_backtest`

```python
def _v4_gate_combos():
    return list(product(
        [False, True],                                  # USE_CALIBRATION_V4
        [(7,15), (5,20), (5,25), (3,30)],               # TP1_BAND (첫째=현행 baseline)
    ))

def run_v4_gate_backtest(data_dir="data", horizon=5, oos_ratio=0.3,
                         v4_table_path="data/calibration_v4_table_latest.json"):
    df = _load_trade_rows(data_dir, horizon)            # (컬럼 확장본)
    v4_table = json.load(open(v4_table_path, encoding="utf-8"))

    # baseline = 현행 (use_v4=False, TP1 7~15) 의 선택집합
    baseline_sel = _simulate_official_pick(df, False, (7,15), v4_table)

    rows = []
    for use_v4, band in _v4_gate_combos():
        rows.append(_evaluate_gate_combo(df, use_v4, band, v4_table, baseline_sel))

    # IS/OOS 분할 재검증 (기존 wf 로직 재사용)
    # → split_date_idx 로 is_df/oos_df 나눠 동일 평가, robustness 판정

    # 검증 케이스 추적 (§7)
    cases = _trace_validation_cases(df, v4_table)

    return {"baseline": rows[0], "combos": rows, "cases": cases,
            "meta": {"horizon": horizon, "total_trades": len(df),
                     "v4_table_meta": v4_table.get("meta", {})}}
```

**판정 규칙(권장):** baseline 대비 `ev↑ AND win_rate_excess↑ AND delta_pnl≥0 AND mdd 비악화`를
IS·OOS 모두에서 만족하는 (use_v4, band) 조합만 "본선 승격 후보".

---

## 6. 미래참조(look-ahead) 방지 — 필수

- v4 테이블은 **각 평가 구간 시작 이전** trade로만 빌드해야 한다. 단일 `calibration_v4_table_latest.json`을
  전 구간에 쓰면 OOS에 미래정보 누수.
- 최소 대응(1차): IS 구간 trade로만 v4 테이블 빌드 → OOS 평가에 사용.
- `build_segmented_table(..., asof_ymd=split_date)`로 청산완료 trade만 반영(기존 kelly_calibrator와 동일 원칙).
- 이 한계를 결과 `meta.lookahead_note`에 명시.

---

## 7. 검증 케이스 훅 (에스엔시스 5/7, 신세계I&C 5/1)

```python
VALIDATION_CASES = [
    {"name": "에스엔시스", "date": "20260507"},   # day-4 진입 케이스
    {"name": "신세계I&C", "date": "20260501"},    # day-10 방치 케이스
]
def _trace_validation_cases(df, v4_table):
    out = []
    for c in VALIDATION_CASES:
        m = df["trade_date"].eq(c["date"]) & df["code"].isin(_codes_for(c["name"]))
        for _, r in df[m].iterrows():
            base = _simulate_official_pick(df.loc[[r.name]], False, (7,15), v4_table).iloc[0]
            v4   = _simulate_official_pick(df.loc[[r.name]], True,  (5,20), v4_table).iloc[0]
            out.append({**c, "ret": round(r["ret"],2),
                        "baseline_picked": bool(base), "v4_picked": bool(v4)})
    return out
```
→ "v4/밴드 완화가 이 두 사고 케이스를 (안)샀는가, 그 결과 손익은?"을 표로 확인.

---

## 8. 출력 / 소비

- 저장: `data/v4_gate_backtest_latest.json` (+ 날짜본).
- 성과탭/관리자 카드에 baseline vs 최적 조합 비교표 + 검증 케이스 표 렌더(별도 작업).
- CLI: `python -m combo_optimizer --v4-gate` 또는 `scripts/run_v4_gate_backtest.py`.

---

## 9. 안전 / 롤백

- read-only: scoring_engine·pipeline_finalize·TOP_PICK 라이브 산식 무변경.
- 새 함수 append-only → 제거 시 즉시 원복.
- 본선 승격은 §5 판정 + §6 look-ahead 통과 후 **별도 PR**에서 `scoring_engine` STABLE 게이트 교체.

---

## 10. 구현 순서 (다음 작업)

1. `_load_trade_rows` 컬럼 확장 + 단위테스트(컬럼 존재/결측 안전)
2. `_simulate_official_pick` + 테스트(AGGRESSIVE 저장값 재사용, STABLE 재현 일치)
3. `_evaluate_gate_combo` + 테스트(보류 현금효과 부호, MDD 비음수)
4. `run_v4_gate_backtest` + IS/OOS look-ahead 빌드
5. 검증 케이스 코드 매핑(`_codes_for`)
6. 67일 recommend × 58 snapshot으로 실행 → baseline 대비 델타 표
