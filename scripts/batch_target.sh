#!/usr/bin/env bash
# [v84.2] 배치 대상 세션 결정 — prefetch 체인과 auto_collect 가드의 단일 출처.
#
# 왜: cron은 몇 시간씩 늦게 발화한다(10/12 prefetch → 10/13 00:55 KST). 자정을 넘긴
#   발화는 '오늘(달력)'이 아니라 '방금 끝난 거래일'을 만들어야 한다. collector 의
#   resolve_trade_date 도 데이터가 있는 최신 세션을 고른다(10/6 00:41 실행 →
#   recommend_20261005 생성으로 확인). v84.1은 이를 '다음 날 장전 배치'로 오진해
#   16시 이전 체인을 막았고, 그 결과 10/12 배치가 통째로 빠졌다.
#
# 창(KST):
#   16:00~24:00  evening   대상 = 오늘 (주말이면 weekend → 대상 없음)
#   00:00~08:00  overnight 대상 = 직전 평일 (지연 발화 회수)
#   08:00~16:00  market    대상 없음 — 장중 실행은 장중 시세로 오늘 배치를 만들 위험
#
# 출력(eval 용): BATCH_WINDOW=... / BATCH_TARGET=YYYYMMDD 또는 빈 값
set -u
NOW=$(TZ=Asia/Seoul date +%s)
kst() { TZ=Asia/Seoul date -d "@$1" "+$2"; }
H=$(kst "$NOW" %H)
DOW=$(kst "$NOW" %u)
if [ "$H" -ge 16 ]; then
  if [ "$DOW" -ge 6 ]; then
    echo "BATCH_WINDOW=weekend"; echo "BATCH_TARGET="
  else
    echo "BATCH_WINDOW=evening"; echo "BATCH_TARGET=$(kst "$NOW" %Y%m%d)"
  fi
elif [ "$H" -lt 8 ]; then
  d=1
  while [ "$(kst $((NOW - d * 86400)) %u)" -ge 6 ]; do d=$((d + 1)); done
  echo "BATCH_WINDOW=overnight"; echo "BATCH_TARGET=$(kst $((NOW - d * 86400)) %Y%m%d)"
else
  echo "BATCH_WINDOW=market"; echo "BATCH_TARGET="
fi
