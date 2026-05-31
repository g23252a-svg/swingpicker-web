# DESIGN — CARRY DISPLAY_SCORE 과차감 근본 수정

> 날짜: 2026-05-31
> 성격: **설계 문서 (코드 변경 없음)** — 엔진 점수 로직 수정 전 영향 분석
> 위험도: **높음** (CARRY 전 종목 DISPLAY 이동 + news/STALE penalty 상호작용)
> 선행 완료: ② 표시 안전망 (`_nb_score_cell`, CARRY 과차감 → '보유' 표시) — 이미 적용
> 이 문서의 목적: ① 엔진 근본 수정을 **백테스트/회귀 없이 건드리면 안 되는 이유**와 올바른 수정 경로를 박아둔다.

---

## 1. 증상

종목탭 테이블에서 CARRY 보유종목이 `S 98 · AI 88`인데 **점수(DISPLAY_SCORE) 0**.

| 종목 | FINAL | DISPLAY | STALE_PENALTY | ROW_BUILD_MODE | AGE |
|---|---:|---:|---:|---|---:|
| 인성정보 | 80.1 | **0** | 20 | CARRY_LEGACY | 30 |
| 하이스틸 | 78.3 | **0** | 20 | CARRY_LEGACY | 30 |
| 오픈베이스 | 62.0 | **0** | 20 | CARRY_LEGACY | 23 |
| TPC로보틱스 | 76.6 | **1.6** | **0** | CARRY_LEGACY | 3 |

FINAL=80인데 STALE=20이면 DISPLAY는 60이어야 한다. 0은 설명 불가.
TPC는 STALE_PENALTY=0(age 3일, STALE 아님)인데도 1.6 — STALE로는 전혀 설명 안 됨.

범위: CARRY 96건 중 70건 DISPLAY=0. **비-CARRY 483건엔 이 현상 0건** (CARRY 한정).

---

## 2. 근본 원인 (코드+데이터로 확정)

### 2-1. ROW_BUILD_MODE 두 갈래 (`pipeline_calibrate._refresh_carry_rows`)

CARRY 보유종목은 매일 재분석을 시도한다:
- **CARRY_REFRESHED (57건):** 재분석 성공 → FINAL 새로 계산. 실제차감(11.1) ≈ STALE_PENALTY(16.3). **정상.** DISPLAY=0은 원래 FINAL이 낮아서(평균 25)지 버그 아님.
- **CARRY_LEGACY (39건):** 재분석 **실패**(`analyze_returned_none`) → 과거 CSV 행을 통째 복사. 실제차감(34.3) ≫ STALE_PENALTY(19.0), **초과분 +15.4.** ← **버그.**

### 2-2. legacy 행은 과거 DISPLAY(이미 깎인 값)를 복사한다 (line 176~178)

```python
prev_map = prev_df.set_index("종목코드")
for code in legacy_codes:
    legacy_rows.append(prev_map.loc[code].to_dict())   # ← 과거 행 통째 복사
```

`prev_map`은 **어제 CSV**다. legacy 행의 `DISPLAY_SCORE`는 **어제 이미 penalty가 적용된 DISPLAY**.
재분석 실패가 며칠 이어지면 어제의 깎인 DISPLAY 위에 또 깎고… **penalty가 매일 누적**된다.

### 2-3. 그 위에 penalty가 3중으로 또 적용된다

| 위치 | 차감 | 대상 |
|---|---|---|
| `pipeline_calibrate.py:186` | DISPLAY −15 | CARRY_LEGACY 전체 |
| `pipeline_calibrate.py:330` | DISPLAY −STALE_PENALTY (최대 35) | STALE/DEAD 단계 |
| `pipeline_calibrate.py:514` | DISPLAY −extra (refresh<50%) | CARRY_LEGACY |

세 차감 모두 **현재 DISPLAY를 기준으로 차감**(`out["DISPLAY_SCORE"] = DISPLAY − penalty`).
legacy의 현재 DISPLAY가 이미 어제 깎인 값이므로 → **과거 누적 + 오늘 3중 = 0 바닥**.

### 2-4. 핵심: FINAL은 살아있는데 DISPLAY만 누적 차감

`scoring_engine.py:450`은 `DISPLAY_SCORE = FINAL_SCORE`로 시작한다(매일 리셋되는 게 정상).
그런데 legacy는 FINAL도 과거값(80.1 보존)이면서 DISPLAY는 누적 차감된 과거값을 들고 온다.
**결과: FINAL 80.1 ↔ DISPLAY 0 괴리.** 같은 종목이 화면 위치마다 다른 점수로 보임
(테이블은 DISPLAY=0, 다른 카드는 ELITE_SCORE=69.7).

---

## 3. 왜 함부로 못 고치나 (충돌·부작용)

### 3-1. 기존 STALE 테스트 계약과 충돌
`tests/test_carry_stale_guard_v3928.py::test_display_score_is_reduced_by_penalty`:
```python
r = _one(CARRY_AGE_DAYS=10, ..., DISPLAY_SCORE=80.0)
assert r["DISPLAY_SCORE"] == 80.0 - 27.0   # 53
```
→ 현재 계약은 **"입력 DISPLAY 기준 차감"**. 단순히 `FINAL−penalty`로 바꾸면 이 테스트가 깨진다.
(단 이 테스트는 입력 DISPLAY=80을 직접 주입 = legacy 오염 없는 깨끗한 입력이므로, FINAL=80과
같다면 결과 동일. 충돌은 **테스트 입력 설계** 문제지 로직 모순은 아님 — 테스트도 같이 손봐야 함.)

### 3-2. news penalty와의 상호작용 (1차 수정안조차 불완전)
수정안 `DISPLAY = clip(FINAL − STALE_PENALTY, 0, 100)` 시뮬레이션 결과:
- DISPLAY 바뀌는 종목 **24건 = CARRY 22 + 비-CARRY 2**
- 비-CARRY 2건: NC 82.9→86, 디바이스 100→92.2 ← **`pipeline_news`의 EV 가감점이 사라짐**
  (`pipeline_news.py:52/70`: `DISPLAY = clip(FINAL + ev, 0, 100)`)
- 즉 `FINAL−STALE` 단순 대체는 **뉴스 반영을 날린다.** 올바른 식은
  `DISPLAY = clip(FINAL + news_ev − STALE_PENALTY, 0, 100)` 형태여야 함.

### 3-3. CARRY 22건 전부 점수 이동
인성정보 0→60, 하이스틸 0→58, TPC 2→77 등. 보유관리 화면·정렬·라벨이 전부 영향받는다.
잘못 고치면 **STALE 경고(보유 너무 오래 들고 있다)** 신호까지 약화될 수 있다.

---

## 4. 올바른 수정 경로 (제안 — 미실행)

핵심 원칙: **DISPLAY_SCORE는 매일 FINAL에서 새로 산출하고, penalty는 누적이 아니라 당일 1회만.**

### 옵션 A — legacy도 FINAL 기준 재산출 (권장)
legacy 행을 복사할 때 과거 DISPLAY를 버리고, 당일 FINAL에서 재계산:
```python
# legacy_df 생성 직후 (line 186 부근)
legacy_df["DISPLAY_SCORE"] = pd.to_numeric(legacy_df["FINAL_SCORE"], errors="coerce").fillna(0)
# 이후 STALE penalty(330) + legacy penalty(186 -15)는 'FINAL 기준 DISPLAY'에 1회 적용
```
- 효과: 과거 누적 오염 제거. penalty는 당일분만.
- 단 FINAL도 과거값(재분석 실패라서) — 가격은 오늘인데 점수는 어제 = 별도 이슈(허용 가능, STALE이 그걸 표현).

### 옵션 B — 차감 출처 단일화
186/330/514 세 곳의 DISPLAY 직접 차감을 없애고, 마지막에 한 번만:
```python
DISPLAY = clip(FINAL + news_ev − STALE_PENALTY − legacy_penalty, 0, 100)
```
- 효과: 누적·순서 의존 제거, news 보존.
- 단 리팩터 범위 큼(3곳 제거 + 1곳 신설), pipeline_news 순서와 조율 필요.

### 공통 필수
- `test_display_score_is_reduced_by_penalty`를 새 계약(FINAL 기준)으로 갱신
- 비-CARRY DISPLAY 불변 회귀 테스트 추가(news 보존 확인: NC/디바이스)
- CARRY 22건 before/after 스냅샷 회귀 테스트
- **combo_optimizer/gate 백테스트로 EV·정렬 영향 확인 후 production** (v4 라운드 원칙과 동일)

---

## 5. 권고

- **②(표시 안전망)는 이미 적용** — CARRY 과차감을 '보유' 표시로 가려 구독자 혼란은 즉시 해소됨.
- **①(엔진 수정)은 옵션 A를 우선 검토**하되, news_ev 상호작용·STALE 테스트 갱신·CARRY 22건 회귀·
  백테스트를 모두 통과한 뒤에만 production. 단독 hotfix 금지.
- 이 수정은 **신규진입 산식(TOP_PICK / BUY_NOW_ELIGIBLE / scoring_engine FINAL)을 바꾸지 않는다.**
  DISPLAY_SCORE(표시 점수) 산출 경로만 정리하는 것.

---

## 6. 유지할 금지선

- legacy 행의 과거 DISPLAY를 그대로 신뢰해 추가 차감 금지 (누적 오염의 원인)
- DISPLAY penalty를 여러 곳에서 순차 차감 금지 — 출처 단일화 또는 FINAL 기준 재산출
- 엔진 수정 시 news EV 반영을 날리지 말 것 (`FINAL−STALE` 단순 대체 금지)
- TOP_PICK / BUY_NOW_ELIGIBLE / FINAL_SCORE 산식 무변경
- 백테스트(EV·정렬·STALE 경고) 통과 전 production 반영 금지

---

## 7. 관련 파일·근거

- `pipeline_calibrate.py` : 176~178(legacy 복사), 186(−15), 330(−STALE), 514(−extra)
- `scoring_engine.py:450` : `DISPLAY_SCORE = FINAL_SCORE` (당일 시작점)
- `pipeline_news.py:52/70` : `DISPLAY = clip(FINAL + ev, 0, 100)` (뉴스 반영)
- `tests/test_carry_stale_guard_v3928.py` : 갱신 필요한 기존 계약
- 표시 안전망: `components/tab_stocks.py::_nb_score_cell` + `tests/test_score_cell_carry.py`
- 데이터 근거: recommend_20260529.csv — CARRY_LEGACY 39건 초과차감 +15.4, 비-CARRY 0건
