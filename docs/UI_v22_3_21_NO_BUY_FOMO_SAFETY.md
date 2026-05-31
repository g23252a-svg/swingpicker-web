# UI v22.3.21 — No-Buy 카드 FOMO-safety

> 날짜: 2026-05-31
> 브랜치: v22.3.21-no-buy-card-fomo-safety (main 머지 완료)
> 성격: 프론트 표시 안전성 (추천식/엔진 로직 무변경)
> 한 줄: **백엔드가 '사지 마라'고 막은 날, 프론트가 그 신호를 약화시키지 않게 한다.**

---

## 1. 문제 (왜 고쳤나)

매수금지(시장 위험 CRITICAL) 화면에서 프론트가 백엔드 안전신호와 모순됐다.

```
🔴 오늘 매수 금지 (시장 위험)        ← 백엔드: 사지 마라
매수 289,500 → 목표 352,500 (+21.8%)  ← 프론트: 그래도 수익 커 보이는데?
✅ 지금 매수 OK (조건 미달이지만 가격은 OK)  ← 프론트: 사도 되는구나?
```

구독자(유료 Toss/데이터셰프)는 백엔드가 아무리 막아도 화면에서 큰 목표가·초록 CTA를 보면
"사도 되는구나"로 해석한다. 즉 **표시 자체가 FOMO를 유발**해 안전장치를 무력화한다.

---

## 2. 고정 원칙

```
공식 신규매수 가능 = TOP_PICK == 1 AND BUY_NOW_ELIGIBLE == 1
```

이 조건이 **아니면** 화면에서:
- 목표가/매수가를 매수 CTA처럼 크게 띄우지 않는다 → 회색 '참고용' 각주로 강등
- 초록색 매수 유도 라벨('지금 매수 OK' 등)을 띄우지 않는다 → 회색/중립 라벨
- '매수 금지'보다 '오늘은 신규 매수 쉬어갑니다' 보호 톤을 쓴다

금지 문구(공식 조건 미통과 시 전부 FOMO): `지금 매수 OK`, `매수 가능`, `진입 가능`,
`강한 매수`, 큰 `목표 +xx%`.

이 원칙은 단일 함수로 고정: `components/buy_now_badge.is_official_new_buy(row)`.

---

## 3. 적용 (3종, 전부 components/tab_market.py)

### ① 가격 강등 (시안 → 회색 각주)
- 가장 가까운 종목 카드 + 관찰 후보 '오늘 매매 제외' 리스트(3종목)
- 전: `매수 289,500 → 목표 352,500 (+21.8%)` (시안색, 매수 유도)
- 후: `참고용 — 조건 충족 시 목표 +21.8% (289,500 → 352,500) · 오늘은 매수 대상이 아닙니다` (회색)

### ② 초록 CTA 게이트
- 가장 가까운 종목 카드 + 추천 카드 양쪽의 '✅ 지금 매수 OK'
- `is_official_new_buy(row)` True → `✅ 오늘 신규 매수 가능` (초록 유지)
- False → `⏳ 가격 도달 — 공식 매수 대상 아님` (회색 중립)
- 주의: 추천 카드 루프는 `top_picks = df[TOP_PICK]`로 ELIGIBLE 필터가 없어서,
  TOP_PICK이지만 ELIGIBLE=0인 종목에 초록이 새던 구멍이 있었음 → 게이트로 차단.

### ③ 엔진 상태 카드 톤
- 가격 FOMO 같은 직접 모순은 없었으나(위험을 빨강으로 보여주는 건 정확), 2가지 개선:
  - 매크로 리스크 칸: 매수금지 구간이면 `→ 오늘은 신규매수 보류 구간` 행동지침 추가
  - '최대허용': 매수금지 구간이면 매수성 ROUTE 단어('적극 매수' 등) 대신 `신규매수 보류` 표시
    (시장이 막은 날 '최대허용: 매수검토'가 매수 가능으로 오해되는 것 방지)
- 게이트: `_is_market_no_buy_mode` (매크로 WARNING/CRITICAL **또는** ROUTE 차단)
- **정상장 표시는 기존과 100% 동일** — 매수금지 구간일 때만 톤 변경.

---

## 4. 테스트 (UI라서 회귀 방지 필수)

- `tests/test_no_buy_card.py` (13 passed) — 모델 규칙:
  official_buy=False면 promote_target=False·price_treatment≠hero·cta=None·rest 톤,
  핵심: '+NN%' 수익 문구가 메인(헤드라인/서브텍스트)에 나오면 **실패**(각주에만 허용),
  양성 대조: official_buy=True면 hero+CTA 허용.
- `tests/test_official_buy_gate.py` (3 passed) — 게이트:
  TOP_PICK=0 또는 ELIGIBLE=0이면 False, 둘 다 1이어야 True, 문자열/실수/결측 방어.
- `tests/test_engine_status_tone.py` (운영 전용, nicegui 필요) — 매수금지 판정:
  CRITICAL/WARNING→보류, NORMAL→정상, CAUTION 단독→보류 아님, ROUTE 차단→보류.

---

## 5. 컴포넌트 (선택적 깊은 개편용, 미연결)

`build_no_buy_card_model()` + `render_no_buy_card()`는 막대그래프(축 통과선)·순위 차단사유·
관리자 raw expander가 포함된 풀 카드. 현재 tab_market의 복잡 블록에 통째로 끼우면 배너 중복
위험이 있어, 위 ①②③ 인라인 패치로 핵심 FOMO만 먼저 잡았다. 카드 자체 리디자인이 필요할 때
이 컴포넌트를 호출 연결하면 된다.

---

## 6. 유지할 금지선 (되돌리지 말 것)

- official_buy=False(=TOP_PICK&ELIGIBLE 아님)인 카드에 큰 목표가/매수가 hero 노출 금지
- official_buy=False 카드에 초록 매수 CTA('지금 매수 OK' 등) 금지
- 매수금지 구간(WARNING/CRITICAL or ROUTE 차단)에서 '최대허용'에 매수성 단어 노출 금지
- 위 게이트는 `is_official_new_buy` / `_is_market_no_buy_mode` 단일 함수로 유지(중복 판정 금지)
- 추천식/엔진/TOP_PICK/BUY_NOW_ELIGIBLE 계약은 이 UI 작업에서 변경하지 않는다(표시 게이트만)

---

## 7. 배포 기록

- 가격강등 + 초록CTA 게이트: main 머지 (커밋 ~425bc4e 계열)
- 엔진 상태 톤: main 머지 (acae3b9..d795fb2)
- Railway 자동 재배포 → ldyprotrader.com 반영
- 검증: 로컬(STORAGE_SECRET=local-dev python main.py, 매크로 비어 NORMAL로 떨어져 초록 게이트 발현 확인) + 운영(CRITICAL 화면에서 가격 각주·엔진 보류 톤 확인)
