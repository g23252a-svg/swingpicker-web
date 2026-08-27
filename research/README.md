# research/ — 예측력 탐색 하네스

## 왜 있는가

`docs/SELECTION_SEARCH_20260827.md`에서 확인된 것: 엔진의 선택 알파가 IS/OOS 양쪽
유의하게 음수이고, **편입 결정 자체가 역예측적**이다. 이걸 고치려면 수많은 가설을
같은 방법론으로 재야 한다.

이 세션에서만 방법론 오류를 세 번 냈다.

1. 일평균 수익을 복리로 곱해 -65%/-80% 같은 값을 만들었다.
2. 종목행 t검정(p=0.0205)을 유의하다고 봤는데 일평균 페어드는 p=0.86이었다.
   같은 날 픽은 상관되어 있으므로 **행 단위 검정은 자유도를 뻥튀기한다.**
3. 손절가가 진입가보다 위에 놓인 행에서 '즉시 이익 손절'이 잡혀 가짜 +20.9%가
   생겼는데 p=0.0002라 통과처럼 보였다.

**측정과 검정은 `harness.py` 한 곳에만 둔다.** 탐색자는 신호(Series)만 만든다.

## 판정 기준 (사전 등록)

`evaluate()` 는 아래를 **전부** 통과해야 PASS 를 준다.

- 일평균 수익 > 0
- 일평균 페어드 t검정 p < 0.05 (양측)
- 부호뒤집기 순열 p < 0.05
- 블록 부트스트랩(block=3) CI95 가 0 을 포함하지 않음
- drop-top-2 후에도 양수 — 상위 2일이 전부가 아님
- 10% 절사평균 양수
- IS/OOS 부호 일치
- 분기 4분할 중 3개 이상 양수

여러 신호를 한 번에 낼 때는 `bh_fdr()` 로 다중검정을 따로 건다.

## 사용

```python
import sys; sys.path.insert(0, '/home/user/swingpicker-web')
from research.harness import load_panel, evaluate, eval_signal, fmt

P = load_panel()                        # 70,907행 · 583종목 · 123일
IS = P[P.seg == 'IS'].reset_index(drop=True)
print(fmt(evaluate(eval_signal('-ret_5d', IS), IS, name='5일 반전', top_n=10)))
```

**탐색은 IS에서만 한다.** OOS는 최종 확인 때 한 번만 쓴다.
신호는 `eval_signal` 로 재현 가능한 **문자열 식**으로 남긴다 — 숫자가 아니라 식을 믿는다.

## 알려진 한계

유니버스 583종목은 엔진의 `top_n=600`(거래대금 상위)에서 나왔고 **중도 소멸 종목이
0건**이다. 저유동·소형 구간 결론에는 생존편향이 걸려 있다.
`services/universe_history.py` 가 시점별 유니버스(2,841종목, 소멸 46)를 복원해 뒀으나
시세 백필은 아직이다(`scripts/backfill_universe_ohlcv.py`).

## 캐시

`_panel_cache.parquet` / `_pool_cache.parquet` 는 재생성 가능한 산출물이라
커밋하지 않는다(`.gitignore`).
