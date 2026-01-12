# ---------------------------------------------------------
# [SCHEMA] 프로젝트 공통 상수 정의 (Source of Truth)
# Collector와 Dashboard가 이 파일을 함께 바라보게 됩니다.
# ---------------------------------------------------------

class RouteState:
    """종목의 현재 상태(Strategy Route) 정의"""
    OVERHEAT = "OVERHEAT"   # 과열 (Passive) - 이익실현/주의
    WAIT = "WAIT"           # 대기 (Passive) - 추세는 좋으나 타점 대기
    ARMED = "ARMED"         # 임박 (Active) - 발사 준비 (기존 SQZ/Watch 상위)
    ATTACK = "ATTACK"       # 공략 (Active) - 진입 시그널 발생
    NEUTRAL = "NEUTRAL"     # 중립 (Passive)
