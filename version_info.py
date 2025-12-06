# version_info.py
# -*- coding: utf-8 -*-

APP_VERSION = "6.5.0"

# 🔹 PRIME 전용 텔레그램 채널 초대 링크
PRIME_TG_JOIN_URL = _get_conf("https://t.me/+DovDEluWnEJhOTY1", "")

CHANGELOG = [
    {
        "version": "6.5.0",
        "date": "2025-12-06",
        "title": "Collector v6.5 / 계정 시스템 안정화",
        "items": [
            "Collector v6.5: 60일 지수 수익률·상대강도(α) 반영",
            "KOSPI/KOSDAQ 지수 fallback 로직 개선 (직전 영업일 자동 탐색)",
            "회원 DB를 GitHub Gist + 로컬 캐시 구조로 안정화",
            "로그인/회원가입 시 이메일 소문자 통일 + 형식 검증 추가",
        ],
    },
    {
        "version": "6.4.0",
        "date": "2025-11-30",
        "title": "점수 로직 / 라우팅 개선",
        "items": [
            "LDY_SCORE 스코어링 안정화 및 페널티 구조 조정",
            "ROUTE 태그: BRK / Watch / MR / PULL 기준 재정의",
        ],
    },
    # 필요하면 과거 버전 계속 추가
]
