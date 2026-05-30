# V4 Gate Backtest Follow-up — TP1 (6,18) 분해 결과

> 날짜: 2026-05-30
> 성격: read-only 분석 (production 변경 없음)
> 결론: TP1 (6,18)은 **watch 후보로만 유지. production 반영 금지.**

---

## 1. 배경

v4 gate backtest(`combo_gate_v4.py`)에서 baseline `(7,15)` 대비 `(6,18)` 및 `(5,20)`이
full/OOS 모두에서 EV·excess 승률·Δ일별손익을 개선하는 듯한 신호가 나왔다.

- full: `(6,18)` EV 2.74 vs baseline 2.38 / excess 24.6 vs 22.2 / Δ일별 +1.98
- OOS(5/1 분할): `(6,18)` EV −3.1 vs −3.77 / excess 29.4 vs 25.0 / Δ일별 +3.55

다만 표본이 작고 `(6,18)`과 `(5,20)` 결과가 동일하여, 실제 영향 구간을 확인하기 위해
TP1 구간별·added 종목별 분해를 수행했다. (데이터: recommend 67일 × price_snapshot 58일, 매칭 53일 27,450건)

---

## 2. added 2건

`(7,15)` baseline에서 제외되고 `(6,18)`에서 새로 포함된 종목은 **2건**이다. (dropped 0건 — 완화는 상위집합)

| 날짜 | 종목 | 코드 | ret | TP1 | RR | ENTRY_RISK | EST_WR | DISPLAY |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 2026-04-20 | 아이엘 | 307180 | +20.9% | 16.3 | 1.00 | 미기록 | 0.574 | 77.4 |
| 2026-05-18 | DB하이텍 | 000990 | +7.6% | 16.8 | 1.03 | ORANGE | 0.599 | 80.1 |

판단:

- 두 종목 모두 수익은 플러스.
- 하지만 표본이 단 2건이다.
- 두 종목 모두 RR이 1.0 턱걸이다.
- 1건(DB하이텍)은 ENTRY_RISK=ORANGE다.
- 따라서 production 승격 근거로는 부족하다.

---

## 3. (6,18)과 (5,20)이 동일했던 이유

TP1 `18~20` 구간에 STABLE 적격 종목이 **0건**이었다.

따라서 `(5,20)`이 `(6,18)`보다 넓지만 실제 추가 종목은 없었다.
이번 동일 결과는 구조적 우위가 아니라 **해당 window의 표본 우연**으로 해석한다.

---

## 4. TP1 구간별 성과

구조 적격 후보 12건 기준 (ELITE≥70 & BALANCE≥70 & MATURE & RR≥1.0 & 거래대금≥50억 & gap≤5 & EST_WR≥0.55, TP1 밴드 무관).

| TP1 구간 | n | 승률 | 평균 ret | 중앙 ret | 해석 |
|---|---:|---:|---:|---:|---|
| 7~15 | 2 | 50.0% | −1.90% | −1.90% | baseline |
| 15~18 | 2 | 100.0% | +14.28% | +14.28% | added 2건 |
| 20~25 | 4 | 0.0% | −9.33% | −10.21% | 명백 악화 |
| 25~30 | 4 | 50.0% | −3.70% | −1.41% | 악화 |

핵심:

- `15~18`의 성과는 added 2건에 **전적으로 의존**한다 (= 그 2종목 그 자체).
- 통계적 엣지라고 보기에는 표본이 너무 작다.
- 반면 `20~25`, `25~30`의 악화는 표본 8건으로 **방향성이 뚜렷**하다.
- 따라서 `(5,25)`, `(3,30)`은 production 후보에서 제외한다.

---

## 5. 레짐 / ENTRY_RISK 한계

### MACRO_REGIME_MODE

`(6,18)` 선택 65건이 **전부 레짐 미기록**이다.
백테스트 구간이 regime logger 보강(2/67 CSV) 전 데이터라 **레짐별 안정성 검증은 불가능**하다.

### ENTRY_RISK

65건 중 58건이 ENTRY_RISK 미기록이다.

- GREEN 6건은 평균 +8.05%로 양호.
- 하지만 대부분 carry 집합이다.
- added 2건 중 1건은 미기록, 1건은 ORANGE다.

따라서 ENTRY_RISK 기준으로도 production 승격은 보류한다.

---

## 6. 최종 판정

### TP1 (6,18) — **watch 후보 유지 / production 반영 금지**

1. added 효과가 단 2종목에 의존한다.
2. added 중 1건이 ORANGE다.
3. RR이 둘 다 1.0 턱걸이다.
4. 레짐 검증이 불가능하다.
5. OOS 표본도 작다(16건).

### TP1 (5,25), (3,30) — **production 반영 금지**

- 20 초과 구간에서 손익비와 MDD가 악화된다.
- 특히 20~25 구간은 0% 승률, 평균 −9.33%로 부정적이다.

### v4 gate — **본선 승격 보류**

- 현재 v4 table은 sparse하다 (라이브 정렬 후 1,015건, prior 0.47, prior fallback 92.6%).
- 상대 gate가 STABLE 후보를 대부분 거른다 (n_added=0).
- logger 보강 후 dense table이 쌓인 뒤 재검증한다.

---

## 7. 유지할 안전선 (변경 금지)

- TP1 `(6,18)` production 즉시 반영 금지
- `(5,25)`, `(3,30)` 반영 금지
- `BUY_NOW_ELIGIBLE` 완화 금지
- `scoring_engine` STABLE gate 교체 금지
- v4 gate 본선 승격 금지

---

## 8. 다음 액션

1. **per-trade logger 보강 데이터 축적** (이미 배포: feat/v4.0-phase1-calibration)
   - `MACRO_REGIME_MODE` / `ACTION_TIER` / `ROUTE` / `TOP_PICK_TYPE`

2. **며칠~수주 후 v4 table 재빌드**
   ```bash
   python scripts/build_calibration_v4_table.py
   ```

3. **TP1 (6,18) 재분해** — 확인 항목:
   - 15~18 구간 표본 증가 여부
   - added 종목의 ENTRY_RISK 분포 (ORANGE 비중)
   - 레짐별 성과 (방어 레짐 포함 안전성)
   - OOS 지속성
   - RR 1.0 턱걸이 후보의 실제 손익

---

## 9. 결론

이번 C 분해는 `(6,18)`을 production에 넣으라는 근거가 **아니라**,
"**18 초과 밴드는 위험하고, 15~18은 watch할 가치가 있다**"는 정도의 결론이다.

현재 운영 결론:

- v4 본선 승격 보류.
- TP1 `(6,18)`은 watch 후보로 유지.
- production 추천식은 변경하지 않는다.
- logger 보강 데이터 축적 후 재백테스트한다.

---

### 재현 (참고)

```python
# combo_gate_v4 의 simulate 로직으로 동일 분해 재현 가능
from combo_gate_v4 import _simulate_official_pick, _build_v4_table_asof
# added = sim(False,(6,18)) & ~sim(False,(7,15))
# 18~20 적격 0건 확인 = sim(False,(5,20)) & ~sim(False,(6,18))  → 빈 집합
```
