# -*- coding: utf-8 -*-
"""
version_info.py (v20.6.3 SSOT Complete)
- CHANGELOG: 전체 업데이트 이력 관리
- UI: 프리미엄 사이드바 뱃지, 업데이트 타임라인
- 100/100: dashboard.py에서 요청하는 모든 물자(변수/함수) 완비
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger("version_info")

# ----------------- 1. 진실의 원천 (CHANGELOG) -----------------
CHANGELOG: List[Dict[str, Any]] = [
    {
        "version": "22.3.6",
        "date": "2026-05-16",
        "type": "patch",
        "title": "v22.3.6 — PRE_ENTRY_RISK 측정 시작: 5/15 폭락 손실의 진짜 원인 추적",
        "items": [
            "\U0001F6A8 **5/15 코스피 폭락 손실 4건 분석:** 5월 15일 코스피가 -6.12% 폭락하면서 그 직전 추천 종목 4건이 손절 처리됐습니다(삼성증권 -10.28%, 삼성E&A -13.02%, 삼성E&A -8.42%, HD현대일렉트릭 -8.27%). 손실 종목을 추적해보니 4건 모두 진입 시점에 '구조점수(STRUCT) 70~85 + VWAP에서 8% 이상 떠있음'이라는 같은 위험 신호를 갖고 있었습니다. 즉 폭락이 우연히 맞은 게 아니라, 폭락 전부터 위험 패턴이었던 종목들이 폭락에 터진 것입니다.",
            "\U0001F9EA **PRE_ENTRY_RISK shadow 신설 (4개 룰 비교):** 위험 신호 종목을 Top3 후보에서 제외했을 때 성과가 어떻게 달라지는지 매일 자동 측정하는 실험실을 추가했습니다. 4개 룰을 동시에 비교합니다: A) STRUCT 70~85만 / B) STRUCT 70~85 + VWAP 8%↑ ('RED') / C) STRUCT<90 + VWAP 15%↑ ('ORANGE') / D) B와 C 합집합. 어느 룰이 '추천 구성을 덜 흔들면서 손실을 가장 잘 줄이는지' 정량 비교합니다.",
            "\U0001F947 **B_red 룰이 가장 유망한 후보로 확인:** 단일 백테스트와 rolling walk-forward(3개 기간 교차검증) 모두에서 B_red(STRUCT 70~85 + VWAP 8%↑)가 5개 기준을 전부 통과한 유일한 룰입니다. 평균 ΔEV +1.76, 평균 Top3 구성 변경률 27.8%로, A룰(48.9%)보다 '덜 흔들면서 비슷한 손실 회피 효과'를 보였습니다.",
            "\U0001F4CA **상승장에서도 효과 유지 신호:** 이전 STRUCT risk shadow는 상승장 효과가 거의 0이었는데, B_red는 상승장 fold에서도 ΔEV +2.68로 강한 양수가 나왔습니다. '약세장 한정'이 아니라 장세 무관하게 작동할 가능성을 시사합니다. 다만 상승장 표본이 아직 1개 구간뿐이라 단정은 이릅니다.",
            "\U0001F512 **production 적용 아직 금지 — 측정 단계 유지:** 단일 백테스트와 RWF 모두 통과했지만, production_candidate는 여전히 False로 유지합니다. RWF는 'CSV에 위험 표시를 추가해도 되는지' 판단용이고, 실제 추천 로직 제외는 그보다 더 보수적으로 갑니다. 다음 단계로 추천 종목 카드에 위험 표시 컬럼을 추가하는 작업을 검토합니다.",
            "\U0001F4CD **추천 화면 변화 없음 (이번에도):** 이번 패치도 추천 종목 선정\u00b7점수\u00b7추천가는 변경 없습니다. 추가된 것은 백테스트 JSON의 pre_entry_risk_shadow 섹션과 RWF 분석 도구뿐이며, 회원이 보는 화면에는 아직 위험 표시가 없습니다. 측정 → 검증 → 표시 → 적용 순서를 지킵니다.",
            "\U0001F50D **5/15 폭락이 가르쳐준 것:** 이번 패치의 핵심 교훈은 '시장 폭락'이 손실의 원인이 아니라 '폭락 직전에 위험 패턴 종목을 걸러내지 못한 것'이 원인이라는 점입니다. 단순한 폭락일 진입 차단(CRASH gate)보다, 폭락 전부터 위험을 알아채는 PRE_ENTRY 신호가 더 근본적인 방어 장치라는 관점에서 작업을 진행합니다.",
        ],
        "schema_min": 5
    },
    {
        "version": "22.3.5",
        "date": "2026-05-14",
        "type": "patch",
        "title": "v22.3.5 — Shadow 실험실: 추천 로직을 바꾸기 전 '바꿨다면?'을 매일 자동 측정",
        "items": [
            "\U0001F9EA **Shadow 실험실 신설 (성과 탭 하단):** 추천 로직\u00b7점수\u00b7매수가를 전혀 바꾸지 않으면서, '특정 규칙을 적용했다면 성과가 어땠을지'를 매일 백테스트에서 병렬 측정하는 실험실 구조를 도입했습니다. 운영 추천에는 일절 반영되지 않는 measurement-only 단계입니다.",
            "\U0001F3AF **ENTRY_MODE shadow \u2014 강한 종목 미체결 회수 측정:** 추천매수가에 체결되지 못한 종목 중 구조 점수가 강하고(STRUCT\u226590) 과열되지 않은(VWAP_GAP\u22648) 케이스에, 다음날 시초가가 +3% 이내면 따라잡았을 경우(chase-entry)를 시뮬레이션합니다. 체결가 기준으로 손절\u00b7손익비를 다시 계산해 정직하게 측정합니다.",
            "\U0001F6E1\uFE0F **STRUCT risk shadow \u2014 애매한 구조 구간 제외 효과 측정:** 구조 점수가 어중간한(STRUCT 70~85) 종목을 Top3 후보에서 제외했을 때 성과가 어떻게 달라지는지 측정합니다. 검증 결과, 이 구간은 손절을 늘리는 게 아니라 '애매하게 버티다 수익을 못 내는' 비승리 구간에서 기대수익을 갉아먹는 패턴으로 확인됐습니다.",
            "\U0001F50D **rolling walk-forward 검증 통과:** STRUCT 70~85 제외 효과를 3개 기간으로 나눠 교차검증한 결과 3개 기간 모두 기대수익(EV)이 개선됐고, 하락장에서 특히 강한 방어 효과(평균 +2.9)를 보였습니다. 상승장에서도 손해가 아닌 중립이었습니다.",
            "\U0001F512 **production 적용은 아직 금지 \u2014 측정 단계:** 두 shadow 모두 단일 백테스트 결과가 좋아도 'production 적용 가능'으로 표시되지 않도록 설계했습니다. Top3 구성을 절반 가까이 바꾸는 변화는 신중해야 하므로, 2~3일 데이터를 더 누적하고 검토한 뒤에야 실제 적용 여부를 판단합니다.",
            "\U0001F4CA **추천 화면 변화 없음:** 이번 패치도 추천 종목 선정\u00b7점수\u00b7추천가 산정은 모두 그대로입니다. 추가된 것은 성과 탭의 Shadow 실험실 카드 하나뿐이며, 기존 백테스트 지표(tp1_rate / stop_rate / ev 등)의 의미도 완전히 동일하게 유지됩니다.",
            "\U0001F52D **앞으로 어떻게 활용되나요:** Shadow 실험실 수치가 며칠간 안정적으로 유지되는지 관찰한 뒤, 효과가 일관되면 추천 CSV에 진입 모드\u00b7위험 구간 표시 컬럼을 추가하는 방향을 검토합니다. 실제 추천 로직 반영은 그 다음 단계입니다.",
        ],
        "schema_min": 5
    },
    {
        "version": "22.3.4",
        "date": "2026-05-13",
        "type": "patch",
        "title": "v22.3.4 — TP1/TP2/TP3 도달 측정 시작 + 손절 전 도달률 추가",
        "items": [
            "📊 **TP1/TP2/TP3 도달률 측정 시작:** 그동안 백테스트가 1차 목표가(TP1) 또는 손절 첫 도달까지만 추적했는데, 이번 패치부터 **TP1 → TP2 → TP3 전체 경로**를 매일 추적합니다. TP3 목표가가 실제로 얼마나 도달하는지 데이터로 검증할 기반이 마련됐습니다.",
            "🎯 **'손절 전 도달률' 신규 지표:** TP1/TP2/TP3 각 단계가 손절 도달 *이전에* 닿았는지 별도 기록합니다. 실전 트레이딩에서는 손절 먼저 맞으면 그 뒤 TP에 가도 못 먹기 때문에, 이 지표가 실제 수익과 더 직결됩니다. 같은 봉에 TP/손절 동시 도달 시 보수적으로 손절 우선 처리합니다.",
            "📈 **추천 후 최대 상승률/최대 하락률 기록:** 종목별로 추천일+10영업일 동안 어디까지 올라갔는지(max_high_pct), 어디까지 떨어졌는지(min_low_pct)도 같이 기록합니다. TP3에 못 닿은 종목도 평균 +14%까지 올라갔다가 꺾이는 패턴이 잡힐 수 있습니다.",
            "🔬 **첫 측정 결과 (4월 한 달 백테스트 / 🏆최강 Top3 기준):** TP1 도달률 38.6% · TP2 도달률 13.6% · **TP3 도달률 3.4%** · 손절률 22.7%. TP3에 도달하는 종목은 30건 중 1건 수준입니다. 현재 🏆최강 라벨은 손절 회피에는 강하지만 TP3까지 끌고 가는 필터로는 부족하다는 게 데이터로 확인됐습니다.",
            "🧪 **추천 화면 변화 없음 — 백그라운드 측정만:** 이번 패치는 데이터 수집 인프라만 추가됐습니다. 추천 종목 선정 로직, 점수 계산, 추천매수가/매도가/손절가 산정 모두 변경 없습니다. 사용자 화면도 그대로입니다.",
            "🔭 **앞으로 어떻게 활용되나요:** 1~2주 데이터가 누적되면 (1) TP3 도달 확률이 실측치로 보정되고, (2) '빠르게 TP1만 먹고 나올 종목' vs 'TP3까지 끌고 갈 종목'을 라벨로 구분하며, (3) 추천 매도 전략을 종목 성격별로 차별화할 수 있게 됩니다.",
            "🛡️ **하위 호환 보장:** 기존 tp1_rate / stop_rate / ev 같은 지표 의미는 v3.7.x와 완전히 동일하게 유지됩니다. 검증 페이지의 기존 숫자가 갑자기 바뀌는 일은 없습니다.",
        ],
        "schema_min": 5
    },
    {
        "version": "22.3.3",
        "date": "2026-05-12",
        "type": "major",
        "title": "v22.3.3 — 💎 Prime 전용 풀 대시보드 출시 (종목 상세화면 완전 리뉴얼)",
        "items": [
            "💎 **Prime 회원 전용 풀 대시보드 화면 출시:** 종목분석 탭에서 종목을 클릭하면 기존보다 훨씬 풍부한 새 종목 상세 화면이 자동으로 나옵니다. 헤더부터 최종 판정 띠까지 한 화면에서 전부 확인 가능합니다.",
            "📊 **메인 캔들차트 + 가격 레벨 6개:** 한국식 캔들(상승 빨강/하락 파랑) + HMA20/VWAP/SUPERTREND 보조선 + TP1/TP2/TP3/매수가/손절가/현재가 6개 점선을 정확한 가격축에 표시합니다. 이전엔 TP1까지만 보였는데 TP2/TP3까지 한눈에 보입니다.",
            "📈 **보조지표 5개 동시 표시:** RSI(14)·MFI(14)·MACD 기울기·V_POWER·거래강도 게이지를 차트 아래 한 줄로 배치 — 각 지표마다 미니 스파크라인 + 색상 분기로 추세 강도까지 즉시 확인 가능합니다.",
            "🎯 **시나리오 A·B·C 자동 분석:** 종목별로 📈 기본 시나리오(진입 후 분할익절) / 🔍 보수 시나리오(HMA20 회복 대기) / ❌ 리스크 시나리오(손절·TIME_STOP)를 자동 생성합니다. 매수해야 할지 관망해야 할지 한눈에 판단 가능합니다.",
            "🎨 **3축 밸런스 레이더 차트:** STRUCT/TIMING/AI/BALANCE/TRIGGER 5축을 펜타곤으로 시각화 — 종목의 균형이 한쪽에 치우쳤는지 즉시 파악할 수 있습니다.",
            "📋 **수익률 현황 + 핵심 레벨 + 리스크 뉴스 3분할 패널:** 1일/5일/10일/20일/60일/120일 수익률 + 시장 대비 상대수익률(rel_20/60/120) + 9개 가격 레벨 + 뉴스 심리/매크로 리스크/BB 확장/스윙 지지 신호를 한 영역에서 동시 확인 가능합니다.",
            "👑 **최종 판정 띠:** 페이지 하단에 종목 등급별 색상(골드/초록/빨강/회색)으로 종합 판정 + 강점 + 리스크를 자동 요약합니다. 예: '601개 중 1위, 실성능 검증된 콤보 등급. 구조 최강 + 진입가 적정 + RR 1.34. 다만 MFI 과열은 체크.'",
            "💡 **마우스 hover 툴팁:** RR_NOW_TP1·KELLY_B·SUPERTREND·V_POWER 등 모든 지표 라벨에 마우스를 올리면 해당 지표가 무엇을 의미하는지 설명이 표시됩니다. 더 이상 지표 약자를 검색할 필요 없습니다.",
            "👤 **투자자 가이드 카드:** 종목별로 보유자 가이드(손절선 관리) / 신규 진입 가이드(분할 접근 가능 여부) / 1차 목표가 / 시간 기준을 한 카드에 정리합니다. OVERHEAT 종목은 자동으로 '신규매수 부적합' 경고가 표시됩니다.",
            "🔄 **분할 익절 플랜 + DipSniper 체크포인트:** TP1 80%·TP2 45%·TP3 25% 분할 익절 비중을 시각화 + 진입가/손절가 사수 / 자동 익절 트리거 / TIME_STOP 발동 조건까지 체크리스트로 확인 가능합니다.",
            "🆚 **종목 간 자동 비교:** 같은 라벨(핵심매수/관심관찰 등)의 다른 종목을 자동으로 비교 종목으로 매핑하여 ELITE 점수 차이를 표시합니다.",
            "🛡️ **안전장치 4중 보장:** 새 화면 렌더링 중 어떤 오류가 발생하더라도 자동으로 기존 v1 화면으로 fallback되어 서비스가 멈추지 않습니다. ?v2=0 쿼리스트링으로 강제 v1 사용도 가능합니다.",
            "🆓 **무료 회원 영향 없음:** 무료/게스트 사용자는 기존 화면 그대로 사용하시면 됩니다. Prime 회원과 관리자에게만 자동으로 새 화면이 적용됩니다.",
        ],
        "schema_min": 5
    },
    {
        "version": "22.3.2",
        "date": "2026-05-10",
        "type": "patch",
        "title": "v22.3.2 — 운영 안정성 4건 + 새 추천 분류 시스템 준비 시작",
        "items": [
            "📮 **문의 게시판 안정화:** 같은 문의가 'None / None'으로 160건 중복 표시되던 버그를 해결했습니다. 앞으로 문의가 정상적으로 한 번만 등록되고, 데이터 백업도 안전하게 저장됩니다.",
            "🛡️ **추천 신뢰도 검증 시스템 정상화:** 매일 자동으로 도는 '추천 정확도 자가 점검' 시스템이 잘못된 비교 방식 때문에 항상 '경고' 상태였던 문제를 해결했습니다. 이제 진짜로 추천이 잘못 가는 날에만 경고가 뜨도록 정확하게 작동합니다.",
            "📊 **추천 데이터 최신화:** 5월 1일에 멈춰 있던 추천 시스템 검증 데이터를 5월 8일까지 일제히 새로고침했습니다. 손익비 1.0 미만 종목이 TOP_PICK에 들어가던 일시적 오류 자연 해소.",
            "🧠 **추천 분류 시스템 4단계 재설계 시작:** 지금까지는 '점수 높은 종목 Top3' 한 가지로만 표시했지만, 앞으로는 다음 4가지로 나눠 보여드릴 예정입니다 — 🟢 **지금 사도 되는 종목** / 🟡 **매집해두면 1~2주 안에 갈 종목** / 🟠 **좋지만 지금은 비싸서 눌림 기다릴 종목** / 🔴 **점수 높아도 추격하면 안 되는 종목**. 이번 패치에서는 이를 위한 신호 데이터 수집을 시작했고, 충분히 검증된 후 단계적으로 화면에 노출할 예정입니다 (현재 사용자 화면 변화 없음).",
            "🔭 **변동성 · 매집 · 추격 위험 신호 수집 시작:** 종목별 변동성 강도, 매집 흐름, 윗꼬리(추격 위험) 지표를 매일 추가로 기록하기 시작했습니다. 1~2주 데이터가 쌓이면 위 4단계 분류의 정확도를 검증할 수 있습니다. **현재 추천 결과에는 영향 없음, 단순 데이터 수집 단계입니다.**",
        ],
        "schema_min": 5
    },
    {
        "version": "22.2",
        "date": "2026-04-29",
        "type": "patch",
        "title": "v22.2 — 약관 동의 오류 긴급 수정 + 추천 데이터 정합성 강화",
        "items": [
            "🚨 **약관 동의 오류 긴급 수정:** \"처리 중 오류 발생. 다시 시도해주세요\" 메시지 반복 표시되어 진입 불가능했던 문제 해결 — DB 저장 메서드(record_terms_agreement) 누락이 원인 + Gist 백업 자동 복구 + UTC 타임스탬프 + 이메일 정규화(lowercase + strip)",
            "🛡️ **추천 데이터 자동 검증 (Recommend Contract):** 추천매수가 / 손절가 / 추천매도가 1·2·3 가격 관계 항상 올바른지 자동 검증 — 손절가 ≥ 매수가 차단, 매도가 단조 증가 강제, PLAN_REASON / EXEC_RULE_ID 빈 값 차단, REGIME 화이트리스트(normal/high_vol/low_vol), 손익비 > 0, 손절폭 ≥ 0",
            "🔒 **한글 컬럼 SSOT 강제화:** 추천 종목 dict 생성을 TradePlanResult.to_recommend_row() 한 곳으로 통일 — 누가 한글 키 매핑 실수로 변경해도 단일 출처에서만 수정되므로 회귀 위험 차단",
            "🔇 **TP 산출 fallback 로깅:** compute_realistic_targets 실패 시 silent → logger.warning 변경 — 추천 목표가가 RR_LEGACY로 fallback될 때 사유 추적 가능 (silent baseline 158 → 157건 1건 감소)",
            "🔬 **회귀 차단 테스트 인프라 확장:** 추천 데이터 계약 검증 테스트 68건 신규 추가 (이전 147 → 215 passed) — 향후 시스템 변경 시 추천 출력 정합성 CI에서 자동 보증 + STRICT_RECOMMEND_CONTRACT 환경변수로 운영/개발 모드 분리",
            "📐 **점수 변화:** Phase 3+4 한글 SSOT 88점 → 97점 (4단계 패치 v1~v4)",
        ],
        "schema_min": 5
    },
    {
        "version": "22.1",
        "date": "2026-04-26",
        "type": "major",
        "title": "v22.1 — 결제·문의·운영 안정성 대규모 개선 (Step S~AA)",
        "items": [
            "💎 **카드 결제 인프라 완비:** 토스페이먼츠 가맹점 가입 + 도메인 연결 + LIVE/TEST 자동 감지 — 라이브 키 발급 후 즉시 활성화 가능",
            "🚀 **사회적 증거 시스템:** 누적 가입자/Prime 멤버/최근 7일 신규/활성 회원 + 분석 종목 498+ 자동 계산 (recommend_latest.csv 기반)",
            "🎉 **베타 50% 할인:** Prime 첫 100명 한정 19,900원 → 9,950원 (2026-05-31까지)",
            "🛡️ **Prime 만료일 보존 연장:** 조기 갱신 시 기존 만료일 + 30일 (잔여 기간 손실 X) — 기존 만료일이 미래면 잔여 보존, 과거면 오늘+30일",
            "⚡ **결제 즉시 권한 갱신:** 결제 완료 후 재로그인 불필요 — app.storage.user 즉시 업데이트로 Prime 메뉴 바로 사용 가능",
            "💳 **결제 기록 DB 영구 저장:** payments 테이블 신규 (order_id PRIMARY KEY) + 성공/실패/위변조/중복 모두 기록 + Telegram 이중 알림",
            "🔒 **Gist 동기화 분리:** payments_db.json 별도 파일 (이전 inquiries 오염 위험 제거) — TABLE_TO_GIST_FILE 매핑으로 정확한 분기",
            "🔐 **이메일 해시 강화:** SHA-256 8자리 → 12자리 (충돌 확률 0.001% → 0.00000003%) + 하위 호환 유지",
            "📮 **문의 게시판 전면 개선:** 같은 글 9번 반복 표시 버그 수정 (inquiry_id PRIMARY KEY + 더블클릭 방어 + 5초 윈도우 안정 ID + INSERT OR REPLACE UPSERT)",
            "📂 **문의 카테고리 5종:** 💳결제/💰환불(1순위) / 🐛버그(2순위) / 💡기능제안 / 📝일반 — 1순위는 운영자 🚨 우선 알림",
            "💬 **답변 시스템 도입:** 관리자 답변 등록 + 본인 글 즉시 확인 + 상태 배지(🟡대기/🔵처리중/✅답변완료/⚫종료)",
            "🔐 **개인정보 보호 강화:** 결제/환불/버그 문의는 본인+관리자만 열람, 일반/기능제안만 공개 게시판 — 일반 유저 토글 분리",
            "📞 **카카오톡 채널 운영:** @swingpicker 자동응답 4종(도움/결제/환불/기능) + 사이트 푸터 직접 연결 + 긴급 문의 우회 가능",
            "🛠️ **사이트 공지 배너 시스템:** 4가지 레벨(info/warning/error/maintenance) + Railway 환경변수 즉시 제어(재배포 X) — 패치/점검/긴급 공지 즉시 노출",
            "⚠️ **관리자 안전장치:** 전체 초기화 RESET 2중 확인 + payments 테이블 보존(감사 로그) + Gist 파일명 버그 수정(users.json → users_db.json)",
            "📋 **입금확인 처리 개선:** 삭제 → closed 상태 변경 + 자동 답변 등록 — 분쟁 대비 영구 보존, 화면에서만 자동 숨김",
            "🏢 **사업자 정보 완비:** 상호/대표/사업자등록번호/주소 모든 페이지 푸터 표시 + 이용약관/개인정보처리방침/환불정책 페이지 추가",
            "🛡️ **결제 보안 검증 6종:** paymentKey 토스 서버 승인 + amount PLAN_PRICES 검증 + orderId 중복 방지(DB UNIQUE) + DB 저장 + role 'prime' + 30일 연장",
            "🔧 **Telegram HTML escape:** 사용자 입력 5개 필드(닉네임/이메일/제목/내용/카테고리) escape — HTML 인젝션/피싱 공격 방지",
            "🎨 **카카오톡 카드 가독성:** 노란 배경+노란 글씨(가독성 0) → 카카오 옐로우 테두리+흰 글씨+검정 버튼 (브랜드 컬러 유지)",
            "📏 **입력 안전성:** 제목 100자 / 내용 5~2000자 + 실시간 글자 수 표시 + 이메일 필수+형식 검증 + 로그인 사용자 readonly(변조 방지)",
            "📐 **점수 변화:** 결제/멤버십 91 → 98점, 문의 게시판 62 → 97점 (총 9단계 패치 Step S~AA)",
        ],
        "schema_min": 5
    },
    {
        "version": "22.0",
        "date": "2026-04-25",
        "type": "major",
        "title": "v22 — 추천 신뢰도 일치 + 자동 검증 시스템 (4개 축 통일)",
        "items": [
            "🎯 **TOP_PICK 누출 차단:** ROUTE=WAIT인데 TOP_PICK=1 모순 케이스 완전 제거 — ARMED/ATTACK만 통과 (positive gate)",
            "🏆 **TOP_PICK 이원화:** AGGRESSIVE(TP1 15%+ 손익비 우선) / STABLE(TP1 7~15% 승률·밸런스 우선) 분리 — 장 상황별 맞춤 추천",
            "📊 **선언 승률 = 실제 승률:** 랭킹·승률·Kelly·백테스트 4개 축을 ELITE_SCORE 하나로 통일 (선언-실현 갭 20%p → 5%p 이내 목표)",
            "💰 **Kelly 과대배팅 방지:** 계획 손익비와 실측 손익비 중 작은 값 사용 — 베팅 비중 최대 40% 감축으로 안전마진 확보",
            "📉 **승률 추정 보수화:** 점수 90점 → 승률 70%(과대) → 64%(보수) — 사용자 기대치와 실제 결과 갭 축소",
            "🧮 **점수 구간별 실측 학습:** 누적되는 거래 결과로 ELITE 점수 구간별 실제 손익비/승률 자동 학습 — 시간 지날수록 정확도 향상",
            "🌍 **벤치마크 초과수익 측정:** KOSPI/KOSDAQ 대비 초과 알파 자동 계산 — 단순 절대수익 → '시장 대비 진짜 잘했는가' 측정",
            "🚦 **자동 CI 검증:** GitHub Actions로 매일 16:30 누출/갭/단조성 점검 (HARD 3개 + SOFT 4개 게이트)",
            "📋 **일일 단조성 리포트:** 점수 구간별 Wilson 신뢰하한 + 시장 매핑 커버리지 + 7일 STABLE 평균 자동 기록",
            "🎯 **적응형 진입 판정:** 시총·변동성(ATR) 기반 IS_NOW_ENTRY 임계값 동적 조정 — 대형주 0.3% / 중형 0.5% / 소형 0.8%",
            "📈 **5-method 학습 확장:** 한 거래당 ELITE/RANK/AI/ROUTE_ARMED/ROUTE_ATTACK 5개 축으로 자동 기록 — 어느 축이 가장 잘 맞는지 비교 학습",
            "🛡️ **종목코드 정규화:** float/공백/None/NaN 모두 안전 처리 — 시장 매핑 정확도 향상으로 벤치마크 신뢰도 ↑",
            "📦 **8축 정렬 SSOT:** TOP_PICK × 진입타이밍 × ROUTE × ELITE × RR × 밸런스 × 진입갭 × 종합점수 일관 정렬",
            "🔧 **0건 날도 검증 갱신:** TOP_PICK 0건이어도 latest.json 갱신 — 어제 데이터 잔존 차단",
            "🧪 **테스트 14건 추가:** Route 계약 6 + 단조성 리포트 smoke 8 (기존 72건 회귀 모두 PASS)",
            "🚀 **production 안착:** 11 files changed, +2,211 lines — 첫 CI 실행에서 즉시 main의 누출 1건(한전기술) 자동 감지",
        ],
        "schema_min": 5
    },
    {
        "version": "21.6",
        "date": "2026-04-17",
        "type": "major",
        "title": "TOP_PICK v21.6 — 36일 백테스트 기반 최적 조합 재설계",
        "items": [
            "🏆 **TOP_PICK 재설계:** 36일 12,599건 백테스트 기반 최적 조합 (기존 승률 57% → 신규 88%)",
            "🏆 **ROUTE 제한 제거:** ARMED/ATTACK 전용 → 전체 ROUTE 허용 (WAIT 승률 93%로 ARMED 76% 압도)",
            "🏆 **3축 하드게이트 추가:** S≥80 + T≥70 조건 신설 (백테스트 TP1 도달률 35%+)",
            "🏆 **TP1 기대수익 필터:** TP1%≥15% 조건 추가 — '크게 먹을 수 있는' 종목만 선별",
            "🏆 **ELITE 상향:** 60→70 (ELITE 단독 역효과 방지, 3축과 결합 시 시너지 극대화)",
            "📊 **TP1_PCT 컬럼 추가:** 현재가 대비 TP1 기대수익률 자동 계산",
            "📊 **RR 단독 필터 제거:** RR≥0.5 독립 조건 삭제 (ELITE≥70이 RR 게이트 내포)",
            "🛡️ **CARRY 차단:** TOP_PICK에서 CARRY 제외 — 신규 매수 시그널만 표시 (기보유자용 혼동 방지)",
            "🔬 **백테스트 검증:** E≥70+S≥80+T≥70+TP1≥15% → UQ=17종목, TP1 35.3%, 승률 88.2%, EV +8.9%",
        ],
        "schema_min": 5
    },
    {
        "version": "21.3",
        "date": "2026-04-12",
        "type": "major",
        "title": "ELITE Ranking + Combo Optimizer + Auth Hardening + Live Price Engine",
        "items": [
            "🏆 **ELITE_SCORE SSOT:** 3축 밸런스(40%) + 현재가 RR(25%) + 밸런스(20%) + 진입갭(10%) + ROUTE(5%) 공식 랭킹 지표 승격",
            "🏆 **TOP_PICK 하드게이트:** ELITE≥60 + ARMED/ATTACK + RR≥0.5 + BAL≥50 + 거래대금≥50억 + 진입갭≤5% + 과열 제외 → 476→8종목 필터",
            "🎯 **조합 최적화 엔진:** combo_optimizer.py — 과거 데이터 기반 최고 승률 지표 조합 자동 탐색 (S≥90 T≥80 AI≥60 → 승률 78.8%)",
            "💎 **EV 이중 랭킹:** 승률 최적 + 기대값(EV) 최적 동시 탐색 — EV = 승률×평균수익 − (1−승률)×평균손실",
            "📊 **시장 대시보드 개선:** 엔진 상태 패널(매크로 리스크/Breadth/신뢰도) + ELITE Top 3 + 최적 조합 매칭 종목 리스트",
            "📊 **종목 테이블 재설계:** AI점수 1칸 → ELITE/종합/S/T/AI/균형/RR 7컬럼, 🏆 ELITE순 기본 정렬, 숫자 정렬 수정",
            "📊 **칸반 카드 강화:** TOP_PICK 🏆배지 + ELITE_REASON 직접 노출 + S/T/AI 미니 스코어",
            "💰 **Kelly 자동 연동:** 종목 선택 시 EST_WIN_RATE + RR_NOW_TP1 자동 매핑, 금액만 조절하면 자동 재계산",
            "💼 **포트폴리오 입력 UX:** 종목 검색 드롭다운(KRX 2,882종목) + 평단가/수량 입력 → 보유 카드 표시 + 개별 삭제",
            "📡 **LIVE_PRICE 엔진:** 시간외 가격 → RR + ELITE + BALANCE + TOP_PICK 전체 실시간 재계산",
            "🔐 **Auth SSOT 완성:** compute_access_status() 단일 함수 + require_premium() 서버측 강제 검증 + @premium_guard 데코레이터",
            "⏰ **만료 자동 강등:** downgrade_expired_users() — 만료 PRIME→FREE 자동 전환 + 관리자 강등 버튼",
            "👑 **관리자 화면 개선:** 실제 접근 상태(✅활성/❌만료/⚪무료/🔑관리자) 컬럼 + 상단 요약 배지",
            "🔧 **Feature Schema 통일:** _compute_feature_hash() → FEATURE_CONTRACT.schema_hash SSOT, ML 캐시 정상 로드",
            "📈 **성과 탭 복구:** .dockerignore data/ 제외 → 무거운 파일만 제외, 22일치 성과 차트 + 다크 테마 호버 수정",
            "🏆 **TOP_PICK 검증 리포트:** top_pick_validation_YYYYMMDD.json 자동 생성 (종목수/평균ELITE/RR/승률)",
            "📊 **CARRY 재분석:** 9%→96% refresh rate, DI패턴 top_df 보강, 진단 로그 강화",
            "🛡️ **NaN 크래시 방어:** _si() Safe Int + _nz() NaN-aware 폴백 — LIVE_PRICE NaN 시 종가 자동 전환",
            "🔓 **프리미엄 전체 표시:** prime/admin 50종목 제한 해제 → 전체 476종목 열람 가능",
            "🚀 **Railway 자동 배포:** GitHub 연결 복구, push 시 자동 deploy 활성화",
            "📱 **Google Play 등록:** 개발자 계정 생성 완료, 본인 확인 심사 중",
        ],
        "schema_min": 5
    },
    {
        "version": "20.6.5",
        "date": "2026-03-14",
        "type": "minor",
        "title": "Module Extraction + Pytest Migration — collector 구조 분할 시작",
        "items": [
            "🏗️ **trigger_engine.py 신규:** calculate_trigger_score + calc_volume_profile_v2 (198줄) collector.py에서 추출, pipeline_score.py 직접 import 전환",
            "🏗️ **investor_flow.py 신규:** fetch_investor_net_buying (55줄) collector.py에서 추출, pipeline_score/pipeline_data 직접 import 전환",
            "📉 **collector.py 경량화:** 2393줄 → 2148줄 (-245줄), 재수출로 하위 호환 유지",
            "🔌 **순환 의존 해소:** pipeline_score.py/pipeline_data.py가 trigger_engine/investor_flow 직접 import → collector 경유 불필요",
            "✅ **pytest 마이그레이션:** test_toxic_filter.py + test_stop_logic.py → pytest class 기반 (pytest -v / 스크립트 양방향 실행)",
        ],
        "schema_min": 5
    },
    {
        "version": "20.6.4",
        "date": "2026-03-14",
        "type": "patch",
        "title": "SSOT Hardening + Bare Except Purge + Stop Logic Tests",
        "items": [
            "🎯 **SSOT 강화:** scoring_engine.py 레거시+벡터 ROUTE 임계치 15건 → collector_config.IndicatorConfig로 통일 (행 단위/벡터 경로 모두 포함)",
            "🎯 **레거시 EBS SSOT:** calculate_ebs_independent() RSI/Vol_Quality 하드코딩 → config 참조",
            "🛡️ **bare except 정리:** 핵심 실행 경로 (collector/pipeline_calibrate/prefetch_flow/version_info/time_utils/strategy_lab) 16건 → 명시적 예외 타입",
            "🧠 **ML 가중치 오염 완화:** _calc_ml_weight partial failure 시 0점 종목을 coverage 계산에서 분리",
            "✅ **stop_logic 전용 테스트:** test_stop_logic.py 신규 — 6중 안전장치 + check_entry_filter API 정합성 검증 (25+ 케이스)",
            "🐛 **회귀 수정:** determine_state_dynamic() _cfg_ind 초기화 순서 버그 → EXIT_WARNING 검출 복구",
            "🧠 **ML center 완전 수정:** _calc_ml_weight() ml_center도 ml_active 기준 산출 (partial failure 보정 완성)",
        ],
        "schema_min": 5
    },
    {
        "version": "20.6.3",
        "date": "2026-03-10",
        "type": "minor",
        "title": "SSOT Complete + Deterministic Engine — Direct Macro SSOT + Adaptive Reason",
        "items": [
            "🎯 **SSOT 완성:** dashboard.py 독자 build_global_score() 제거 → scoring_engine.build_global_score() 단일 경로, CSV/Dashboard/Backtest 점수 100% 일치",
            "⚡ **Trigger 병렬 배치:** ThreadPoolExecutor 4-worker 병렬 처리 (iterrows 제거), sequential 대비 ~3x 속도 향상",
            "⚡ **ROUTE 벡터 판정:** apply(determine_state_dynamic) 제거 → _vec_determine_state_dynamic() 벡터 연산, 100% 단건 함수 일치 검증 완료",
            "📝 **점수 설명 컬럼:** SCORE_REASON_TOP1/TOP2 실제 점수순 정렬 (numpy argsort 완전 벡터화), 장세 연동 임계치 (BEAR=50, CAUTION=60, NORMAL=70)",
            "🔧 **예외 처리 등급화:** 투자 판단 축(ML/TRIGGER/SECTOR/FLOW) 실패 시 [AXIS:*] 태그 warning → run_health 후속 평가",
            "💾 **ML 캐시 개선:** feature_cache pickle → joblib + schema sidecar (feature_cols_hash), 구버전 자동 마이그레이션",
            "📡 **Macro SSOT 직접 전달:** MACRO_RISK/MARKET_BREADTH를 CSV + run_health JSON + run_meta JSON sidecar에 직접 저장",
            "📋 **run_meta sidecar:** 레거시 CSV용 메타 복원 경로 — macro_risk/가중치/scoring_axes 등 파이프라인 컨텍스트 보존",
            "📝 **주석/문서 동기화:** scoring_engine/pipeline_score/pipeline_finalize/ml_engine/run_health/version_info docstring을 v20.6.3 실제 구현과 일치시킴",
            "✅ **테스트 동봉:** test_v206_features.py — ROUTE 벡터 정확도/설명 컬럼 점수순/장세 연동/Macro SSOT 회귀 검증",
        ],
        "schema_min": 5
    },
    {
        "version": "20.5c",
        "date": "2026-03-08",
        "type": "minor",
        "title": "KIS API 수급 연동 완전체 — Run Status: OK 100/100 달성",
        "items": [
            "📡 **KIS API 수급 선수집 (prefetch_flow.py v2.0):** pykrx 전용 구조 → KIS API v2.0 완전 교체, `/uapi/domestic-stock/v1/quotations/foreign-institution-total` (tr_id: FHPTJ04400000) 단일 호출로 외인·기관 동시 수집",
            "🗂️ **flow 캐시 구조 확정:** `frg` / `inst` / `ant` 키 기반 저장, 주말 stale 오판 수정 (직전 평일 기준 비교)",
            "🔌 **collector 캐시 우선 로드:** `fetch_investor_net_buying()` 상단에 KIS 캐시 우선 로드 블록 삽입, pykrx 폴백은 캐시 미존재 시에만 실행",
            "✅ **end-to-end 검증 완료:** 수급 캐시 로드: 외인 30건 기관 30건 → 🟢 Run Status: OK (신뢰도: 100/100, 최대허용: ATTACK)",
            "🔧 **GitHub Secrets 연동:** KIS_APP_KEY / KIS_APP_SECRET 등록, prefetch_flow.yml 환경변수 연결 완료",
            "🐛 **load_cache 주말 버그 수정:** 일요일 실행 시 trade_ymd(금) 불일치 → stale=True 오판 해소",
        ],
        "schema_min": 5
    },
    {
        "version": "19.1.0",
        "date": "2026-03-01",
        "type": "minor",
        "title": "CI/CD 파이프라인 + 보안·안정성 긴급 패치 (3-Fix Hotfix)",
        "items": [
            "🔧 **GitHub Actions CI/CD:** Push/PR마다 순환참조 검사 + Silent Exception 검출 + 배포 전 안전 체크 자동화 (ci.yml, pre-deploy.yml)",
            "🔒 **관리자 암호화 강화 (Fix#3):** SHA256 단순 해싱 → bcrypt 12-round 전환, 레인보우 테이블 공격 완전 차단, 타이밍 공격 방어 내장 (bcrypt.checkpw)",
            "🧵 **Thread-Safety 수정 (Fix#1):** DataStore.scored 프로퍼티가 참조(Reference)만 반환하던 가짜 스레드 안전성 → .copy() 스냅샷 반환으로 Race Condition 근본 차단",
            "📢 **Silent Exception 로거 (Fix#4):** _authenticate_user 로그인 타임스탬프 + get_auth_status 구독 만료일 파싱 실패 시 logger.error/warning 기록 (기존 except:pass 제거)",
            "📊 **Silent Exception 자동 추적:** 전체 120건 except:pass 잔존 현황을 매 Push마다 CI에서 리포트, 점진적 제로화 추적 시작",
            "🛡️ **의존성 방향 검사:** AST 기반 import 그래프 분석으로 views→services→core 단방향 강제, NiceGUI 격리 규칙 자동 검증",
            "⚙️ **환경변수 분리:** BCRYPT_COST(운영12/테스트4), LOG_LEVEL, LOG_TO_FILE Railway 환경변수 추가",
        ],
        "schema_min": 5
    },
    {
        "version": "19.0.0",
        "date": "2026-02-22",
        "type": "major",
        "title": "Production-Grade Refactor: P2~P5 완전체 (276/276 Tests)",
        "items": [
            "🏗️ **P2 모듈화:** collector.py 3306줄→2730줄, 7개 모듈 분리 (collector_config, data_source, macro_filter, news_engine, telegram_sender, validation, scoring_engine)",
            "🗄️ **Parquet 캐시:** pickle RCE 취약점 제거 → Parquet 기반 안전 캐시 (allow_legacy_pickle=False 기본)",
            "⚙️ **Config 중앙화:** 40+개 상수 dataclass화 (collector_config.py), 매직넘버 자동검출 grep 패턴",
            "🔒 **섹터 이중 보상 잠금:** STRUCT 불변, TIMING만 +8/+4, FINAL=TIMING×w_t 수학적 검증",
            "📊 **P3 #13 자동 백테스트:** 폐루프 완성 (추천→실현수익률→승률테이블→Kelly), 7개 안전장치 (누수차단/진입청산고정/binning/min_n+스무딩/Kelly제한/비용반영/기업행위필터)",
            "📍 **P3 #14 포지션 트래킹:** 실시간 SL/TP/드로다운 감지, event_key 중복방지, positions.json SSOT, 원자적 저장(tmp→rename+락)",
            "🧠 **P3 #15 Multi-Timeframe:** 주봉/월봉 구조 필터 (STRUCT±10/15), 미완성봉 누수방지, Config화, TIMING 무관 수학적 검증",
            "🔄 **#16 Gemini 429 방어:** exponential backoff + jitter + Retry-After 우선 + cap + total_timeout(60s), news_engine SSOT 공유",
            "🗃️ **#17 DB 싱글톤:** thread-safe Double-Checked Locking, 쿼리 직렬화 _conn_lock, TTL(10분) gist 재로드, force_refresh",
            "🛡️ **P4 #18 평문 제거:** MASTER_ADMIN_PW 전역변수 del, _ADMIN_PW_SET(bool)만 유지, compare_digest timing-safe",
            "📝 **P4 #19 tomllib 전환:** SECTION_SUBKEY 네임스페이스 등록 (AUTH_MASTER_ADMIN_PW), 섹션 키 충돌 방지",
            "🧪 **P5 #20 단위테스트 확대:** TIMING/STRUCT/SuperTrend/EBS/safe_float 경계값+NULL방어, build_global_score 불변식",
            "📈 **P5 #21 일일 성과 리포트:** #13 realized_returns 재사용, report_key 중복방지, 점수대별 승률/Top 수익·손실",
            "✅ **276/276 ALL PASSED:** 9개 테스트 파일 (scoring_weights 54, ssot_import 14, toxic_filter 12, p2_refactor 53, auto_backtest 31, llm_retry 30, db_singleton 15, security 23, p5_final 44)",
        ],
        "schema_min": 5
    },
    {
        "version": "18.0.0",
        "date": "2026-02-17",
        "type": "major",
        "title": "Premium UI + ML v17.1 + Critical Bug Fixes",
        "items": [
            "🎨 **Premium Dark Theme:** Outfit 폰트, 글래스모피즘 카드, 그라디언트 메트릭, 탭 하이라이트 등 전면 UI 리뉴얼",
            "💎 **Hero Banner:** 기존 st.title → 그라디언트 히어로 배너 + AI 엔진 스펙 뱃지",
            "🃏 **Kanban 2.0:** 칸반 카드 글래스모피즘 + 점수별 색상 뱃지 + R:R 시각 바 + 호버 부유 효과",
            "🧠 **ML v17.1:** 피처 버전 메타데이터(trading_meta_v17.json) 저장 → 코드/모델 불일치 시 레거시 자동 폴백",
            "🔒 **Thread Safety:** load_model()에 threading.Lock 적용 → ThreadPoolExecutor 레이스 방지",
            "🐛 **매수가 중복 계산 Fix:** R:R 2.0→2.5 복원, 7%↑ 급등주 추격방지 정상화",
            "🚫 **OVERHEAT 필터:** RSI≥75 과열 종목이 정예군 Top120에서 자동 제외",
            "🤖 **ML 피처 확장:** 6→16개 (MFI, MACD_Hist, BB_Width, ATR, OBV_Slope 등)",
            "📊 **XGBoost 앙상블:** LSTM 60% + XGBoost 40% 가중평균, Focal Loss 적용",
            "🔑 **Secrets 매핑 테이블:** TOML 중첩 섹션 → 환경변수 확정 매핑 (telegram_token → TG_TOKEN)",
            "🗄️ **DB 컬럼 Fix:** users 테이블 14컬럼 vs INSERT 12값 불일치 → 컬럼명 명시로 해결",
            "🔍 **Gist 진단 패널:** 회원정보 미로드 시 10단계 진단 UI 자동 표시",
            "📈 **게이지 차트:** 투명 다크 배경, 4단계 구간 색상, Outfit 폰트 적용",
            "👑 **관리자 통계:** DAU/WAU를 3색 그라디언트 카드로 시각화",
            "🧪 **Unit Tests:** test_ml_engine.py 8개 테스트 케이스 (차원 불일치, 모델 없음, 정상 추론 등)",
        ],
        "schema_min": 5
    },
    {
        "version": "12.3.1",
        "date": "2026-02-14",
        "type": "major",
        "title": "Absolute Defense: Fix Import Issue",
        "items": [
            "🛡️ **Logic Isolation:** Import 시점의 UI 의존성 제거",
            "⚙️ **Robust Config:** 보급로(PRIME_TG_JOIN_URL) 변수 확립",
            "🚦 **Integrity Gate:** 무결성 검증 로직 함수화",
        ],
        "schema_min": 5
    }
]

# ----------------- 2. 핵심 로직 (Core) -----------------

def _get_conf(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val: return str(val)
    try:
        import streamlit as st
        if key in st.secrets: return str(st.secrets[key])
        if "core" in st.secrets and key in st.secrets["core"]: return str(st.secrets["core"][key])
    except Exception: pass  # [v20.6.4]
    return default

def _parse_version(v_str: str) -> Tuple[int, ...]:
    try: return tuple(map(int, (v_str.split('.'))))
    except (ValueError, TypeError): return (0, 0, 0)  # [v20.6.4]

# [📍 핵심 보급품] 앱 버전
APP_VERSION = CHANGELOG[0]["version"] if CHANGELOG else "18.0.0"
VERSION_TUPLE = _parse_version(APP_VERSION)

# [📍 핵심 보급품] 텔레그램 조인 URL
PRIME_TG_JOIN_URL = _get_conf("LDY_PRIME_JOIN_URL", "https://t.me/+DovDEluWnEJhOTY1")

def get_latest_log() -> Optional[Dict]:
    return CHANGELOG[0] if CHANGELOG else None

# [📍 핵심 보급품] 버전 라벨 함수
def get_version_label(include_build: bool = True) -> str:
    if include_build:
        return APP_VERSION
    return f"{VERSION_TUPLE[0]}.{VERSION_TUPLE[1]}"

def validate_integrity() -> bool:
    env_ver = _get_conf("LDY_APP_VERSION", APP_VERSION)
    if env_ver != APP_VERSION:
        logger.critical(f"🚨 VERSION CORRUPTION: Environment({env_ver}) != Core({APP_VERSION})")
        return False
    return True

# ----------------- 3. 유틸리티 -----------------

def _hex_to_rgb(hex_color: str) -> str:
    """#3B82F6 → '59,130,246' (CSS rgba용)"""
    h = hex_color.lstrip('#')
    try:
        return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
    except (ValueError, IndexError):  # [v20.6.4]
        return "100,100,100"

# ----------------- 4. UI 렌더링 (Streamlit 기반) -----------------

def show_toast_notification():
    import streamlit as st
    if "has_seen_version_toast" not in st.session_state:
        latest = get_latest_log()
        if latest: st.toast(f"🚀 v{APP_VERSION} 업데이트!", icon="💎")
        st.session_state["has_seen_version_toast"] = True


def render_sidebar_version_badge():
    """[v18.0] 사이드바 버전 뱃지 — 프리미엄 디자인"""
    import streamlit as st
    latest = get_latest_log()
    ver_type = latest['type'] if latest else "patch"
    ver_date = latest.get('date', '') if latest else ''
    item_count = len(latest.get('items', [])) if latest else 0

    colors = {"major": "#3B82F6", "minor": "#06B6D4", "patch": "#10B981"}
    bg_color = colors.get(ver_type, "#64748B")
    type_label = {"major": "MAJOR", "minor": "MINOR", "patch": "PATCH"}.get(ver_type, "UPDATE")

    st.sidebar.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(30,30,30,0.95), rgba(20,20,30,0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 4px solid {bg_color};
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 8px;
            font-family: 'Outfit', -apple-system, sans-serif;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.65rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.08em;">
                    LDY Pro Trader
                </span>
                <span style="
                    font-size:0.6rem; font-weight:600; color:{bg_color};
                    background:rgba({_hex_to_rgb(bg_color)},0.12);
                    padding:2px 8px; border-radius:10px;
                    letter-spacing:0.05em;
                ">{type_label}</span>
            </div>
            <div style="font-size:1.3rem; font-weight:700; color:#F1F5F9; letter-spacing:-0.02em;">
                v{APP_VERSION}
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:0.68rem; color:rgba(255,255,255,0.35);">
                <span>{ver_date}</span>
                <span>{item_count} changes</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_recent_updates(limit: int = 5):
    """[v18.0] 업데이트 노트 — 타임라인 스타일"""
    import streamlit as st

    st.markdown("""
        <style>
        .update-timeline {
            position: relative;
            padding-left: 28px;
        }
        .update-timeline::before {
            content: '';
            position: absolute;
            left: 8px;
            top: 4px;
            bottom: 0;
            width: 2px;
            background: linear-gradient(180deg, #3B82F6, rgba(59,130,246,0.05));
            border-radius: 1px;
        }
        .update-entry {
            position: relative;
            margin-bottom: 28px;
            padding-left: 18px;
        }
        .update-entry::before {
            content: '';
            position: absolute;
            left: -24px;
            top: 6px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 2px solid #3B82F6;
            background: #0E1117;
        }
        .update-entry.latest::before {
            background: #3B82F6;
            box-shadow: 0 0 10px rgba(59,130,246,0.5);
        }
        .update-ver {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 1.05rem;
            font-weight: 700;
            color: #F1F5F9;
        }
        .update-title {
            font-size: 0.88rem;
            color: #94A3B8;
            margin-top: 2px;
        }
        .update-date {
            font-size: 0.7rem;
            color: #475569;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 2px;
        }
        .update-tag {
            display: inline-block;
            font-size: 0.58rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 10px;
            margin-left: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            vertical-align: middle;
        }
        .tag-major { background: rgba(59,130,246,0.15); color: #3B82F6; }
        .tag-minor { background: rgba(6,182,212,0.15); color: #06B6D4; }
        .tag-patch { background: rgba(16,185,129,0.15); color: #10B981; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧩 System Intelligence Updates")

    # 타임라인 렌더링
    html_parts = ['<div class="update-timeline">']

    for i, log in enumerate(CHANGELOG[:limit]):
        ver = log['version']
        date = log.get('date', '')
        title = log.get('title', '')
        ver_type = log.get('type', 'patch')
        tag_class = f"tag-{ver_type}"
        latest_class = "latest" if i == 0 else ""
        n_items = len(log.get('items', []))

        html_parts.append(f"""
        <div class="update-entry {latest_class}">
            <div>
                <span class="update-ver">v{ver}</span>
                <span class="update-tag {tag_class}">{ver_type}</span>
            </div>
            <div class="update-date">{date} · {n_items} changes</div>
            <div class="update-title">{title}</div>
        </div>
        """)

    html_parts.append('</div>')
    st.markdown(''.join(html_parts), unsafe_allow_html=True)

    # 상세 내용은 Expander
    for i, log in enumerate(CHANGELOG[:limit]):
        with st.expander(f"v{log['version']} — {log['title']}", expanded=(i == 0)):
            for item in log['items']:
                st.markdown(item)
