# OHLCV 버전 저장소 — v85-2a (2026-10-16)

> 문제: `data/ohlcv_cache_YYYYMMDD.parquet`·`quiet_lane_ohlcv_YYYYMMDD.parquet` 가 매일 밤 ~1년치
> **전체 스냅샷**을 새 파일로 쓴다. 164+30개 · 3,400만 행 중 97.5%가 앞 파일과 같은 값이고,
> 밤마다 깃이 ~7.4MB(합집합 재작성 포함 ~16.9MB) 커진다 — 저장소 용량의 약 2/3.
> 이번 단계(2a): **코드 + 그림자 이중 기록만.** 소비자 전환·날짜 파일 커밋 중단·이력 재작성
> (force push / 기존 파일 untrack) 없음.

## 1. 레이아웃

```
data/ohlcv_store/main/v/YYYYMMDD.parquet   그날 파일에서 처음 보였거나 값이 바뀐 (종목, 날짜) 행
data/ohlcv_store/main/m/YYYYMMDD.parquet   그날 파일의 구성 — 커밋 마커
data/ohlcv_store/lane/{v,m}/YYYYMMDD.parquet   레인 캐시 (인덱스 '날짜', 등락률 없음)
data/ohlcv_store/verify_log.jsonl          야간 그림자 검증 한 줄씩 (밤 × 종류)
```

- **v**: `code, date, 시가…(원본 값 컬럼 이름 그대로), rev` · zstd. `rev=True` 는 이미 있던 키의 값이
  바뀐 것 — 액면분할/병합 후 수정주가 재작성(65종목, 23,059쌍), 당일 봉 자리표시(시가 0·거래량 과소)의
  다음 날 확정, 당일 거래량 보정. 값은 비트 단위 비교(NaN 끼리 같음).
- **m**: `pos, code, dmin, dmax, n` — 파일의 종목 블록 순서와 각 블록의 날짜 구간. 파일 메타데이터에
  원본 arrow 스키마(pandas 메타데이터 포함, zlib+base64), 행수, n_new, n_rev, 개정 종목, sha256.
  중간에 날짜가 빠진 종목은 구간이 둘 이상이 될 뿐 그대로 표현된다.
- 복원: 그 밤까지의 v 를 밤 순서로 접은 최신 값에서 m 구간을 잘라 pos·날짜 순으로 잇고 원본
  스키마로 캐스팅 → **원본과 같은 arrow 테이블**(→ `pd.read_parquet` 프레임 동일).
- **m 을 마지막에 쓴다**(임시 파일 + `os.replace`). m 없는 밤의 v 는 무시된다.
- 이름 규칙: 하위 폴더 + 숫자 이름 → `ohlcv_cache_*`·`quiet_lane_ohlcv_*`·`recommend_*` 글롭(최신 파일을
  `sorted(glob)[-1]` 로 고르는 소비자 7곳 이상)에 절대 안 걸린다. 재귀 글롭으로 data/ 를 훑는 코드는 없다.
- 같은 밤 재수집(수동 재실행) → 그 밤 조각만 **교체**. 더 오래된 밤이 바뀐 경우는 `ingest --ymd all`
  만 그 밤부터 뒤를 다시 계산한다(뒤 밤의 날짜 파일이 모두 있을 때만).

API(`services/ohlcv_store.py`): `ingest` · `snapshot` · `verify` · `hydrate` · `stats` · `members` ·
`latest_union` · `shadow`. CLI: `python -m services.ohlcv_store {ingest,verify,hydrate,stats,shadow} --data-dir data`.

## 2. 검증 (실데이터, data/ 에는 쓰지 않음 — 심링크 미러 → scratch 저장소)

| 항목 | 결과 |
|---|---|
| 부트스트랩 ingest (main 164밤 + lane 30밤, 밤마다 원본 대조 포함) | 57s (main 52 · lane 5.5) — 실패 0 |
| verify all, deep(`pd.read_parquet` 프레임까지) | **main 164/164 · lane 30/30 정확** · 62s |
| hydrate → 원본 비교 | 194/194 프레임 동일. 같은 pyarrow(26.0.0)로 쓴 원본 4개는 **바이트까지 동일**(나머지는 작성 pyarrow 23~25 의 created_by 차이) |
| latest-wins 합집합 ↔ `ohlcv_union_hl.parquet` | 768,591행 · 키 차이 0 · O/H/L/C 불일치 0 |
| latest-wins 합집합 ↔ `ohlcv_union.parquet` (float32) | 768,591행 · 키 차이 0 · 시가/종가 불일치 0 |
| 재실행 멱등 | 증분 0.0s(아무것도 안 씀) · `--ymd all` 전수 대조 10s, 재계산 0 |
| 크기 | 저장소 29.0MB (main v 18.6 + m 1.5 · lane v 8.7 + m 0.3) vs 날짜 파일 756.2MB (3.8%) |
| 밤당 증가 (최근 10밤) | main v 33~55KB + m ~8KB · lane v 60~175KB + m ~8KB · 휴장 복제 밤 v 0.8KB |

합집합의 분할 경계 점프(latest-wins 의 성질)는 저장소가 만들지도 없애지도 않는다 — 합집합과 같은 값이다.

## 3. 야간 배치 (auto_collect.yml)

`OHLCV Store Shadow Ingest (v85)` 스텝 — Show outputs 뒤, Commit & Push 앞.
- 조건: 다른 후속 스텝과 같은 `skip != 'true' && holiday != 'true'`, `continue-on-error: true`, 15분 제한.
- `python -m services.ohlcv_store shadow --data-dir data --log data/ohlcv_store/verify_log.jsonl`
  - 증분 ingest(첫 밤은 부트스트랩 ~1~2분) → 대상 밤(종류별 최신 날짜 파일)은 deep 검증, 함께 적힌 밤은
    arrow 수준 검증 → `verify_log.jsonl` 에 `ymd, kind, ok, rows, n_new, n_rev, mismatch, seconds…` 한 줄.
  - 잡 로그에 요약 3줄, 불일치면 `::warning::`.
- 커밋: `if [ -d data/ohlcv_store ]; then git add -f data/ohlcv_store; fi` 추가.
  **날짜 파일은 지금처럼 그대로 커밋**(그림자) — 읽는 쪽은 아무것도 바뀌지 않는다.
- `.dockerignore` 에 `data/ohlcv_store/` (Railway 이미지 제외 — 기존 `data/*.parquet` 패턴은 하위 폴더에 안 걸림).
- `tests/conftest.py` 의 data/ 오염 감시가 `data/ohlcv_store/` 도 본다.

## 4. 전환 (별도 PR — 깨끗한 검증 ≥ 5밤 뒤)

게이트: `verify_log.jsonl` 의 연속 ≥ 5밤 × 두 종류가 `ok: true`, 그리고 `verify --ymd all` 전수 정확.

1. **hydrate 를 먼저 연결**한다 — auto_collect(Run Collector 앞), prefetch_flow(`investor_flow_kis.universe_codes`
   가 최신 main/lane 파일 코드를 읽는다), backfill_flow, ci.yml(실데이터 테스트 전). hydrate 가 실패하면
   잡을 멈춘다(빈 이력 위에서 swing_features·backtest_validation 이 조용히 계산되는 것보다 낫다).
2. 커밋 스텝의 `git add -f data/*.parquet` 를 "m 마커가 있는 밤의 날짜 파일은 건너뛰기" 루프로 바꾼다.
   ingest 가 실패한 밤(m 없음)은 지금처럼 날짜 파일을 통째로 커밋한다(fail-safe).
3. 합집합 두 개(`ohlcv_union*.parquet`, 밤마다 9.4MB 재작성)는 그 다음 단계 — `git rm --cached` 후
   hydrate 가 기존 빌더로 다시 만든다. pick_history 가 candidate_depth 보다 먼저 union_hl 을 읽으므로
   (pipeline_finalize) hydrate 는 collector 보다 **앞에서** 합집합을 만들어야 한다.
4. 이미 커밋된 날짜 파일(756MB)은 그대로 둔다 — 이력 크기는 줄지 않는다(이력 재작성 금지).

예상: 2단계 후 밤마다 ~9.6MB(-43%), 3단계 후 ~0.15~0.25MB(-98.8%).

## 5. 되돌리기

- 2a(이번 PR): 워크플로 스텝을 지우면 끝 — 날짜 파일은 계속 커밋되고 있었으므로 잃는 것이 없다.
  `data/ohlcv_store/` 는 남겨도 되고 `git rm -r --cached` 해도 된다.
- 전환 후: `python -m services.ohlcv_store hydrate --data-dir data` 로 전환 이후 밤의 날짜 파일을 만들어
  한 번 `git add -f` 커밋한 뒤(밤당 ~4.4MB 일회성) 워크플로를 되돌린다. 그냥 되돌리면 옛 워크플로가 이력 구멍을 본다.

## 6. 같은 PR 의 버그 두 개

- **레인 캐시 '날짜' 인덱스** — `winner_profile._load_universe_ohlcv` 가 레인 행 전부를 버렸다.
  `_with_date_col` 로 맞추고, 최신 파일에 없는 픽 종목은 그 종목을 담은 가장 최근 날짜 파일에서 잰다
  (`_load_code_history`). quiet_lane_track 측정 2일/2픽 → 20일/46픽(5일 창이 닫힌 105픽 중 — 나머지는
  t+6 이전에 밴드 밖으로 밀려나 어느 파일에도 시세가 없다). ext_lowtv 유니버스 ≈600 → ≈1,190,
  10/06~10/16 7일 모두 다른 종목(전부 레인 종목)이 뽑힌다 — 기존 기록은 고치지 않는다.
- **휴장일 팬텀 캐시** — v84 가드보다 먼저 저장돼 10/09 에도 전일 복제 캐시가 커밋됐다. collector 저장
  지점에서 같은 판정(`ohlcv_store.cache_write_allowed`)을 하고 'holiday' 면 저장하지 않는다(판정 보류·실패는
  저장). 같은 실행의 CARRY 재수집도 기억된 판정으로 막는다. 기존 팬텀 파일(main 11: 0302·0501·0505·0525·
  0603·0717·0817·0924·0925·1005·1009 / lane 3: 0924·0925·1005)은 이번에 옮기지 않는다.
