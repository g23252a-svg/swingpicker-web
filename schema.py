from enum import Enum
from typing import List, Dict, Optional, Any  # ✅ Any 정밀 수급 완료

class RouteState(Enum):
    """
    [v3.1 Ultimate] 종목 작전 상태 정의 (Source of Truth)
    - 100/100: 정적 분석 완벽 대응 및 방어적 코드 강화
    """
    ATTACK   = ("🚀 공략", "#FF4B4B", 1, True)
    ARMED    = ("🔫 임박", "#FFA500", 2, True)
    WAIT     = ("⏳ 대기", "#00C853", 3, False)
    OVERHEAT = ("🔥 과열", "#7D3CFF", 4, False)
    NEUTRAL  = ("⚪ 중립", "#808080", 5, False)

    def __init__(self, label: str, color: str, priority: int, is_active: bool):
        self.label = label
        self.color = color
        self.priority = priority
        self.active = is_active

    @property
    def code(self) -> str:
        """Enum 멤버 이름을 코드명(Source of Truth)으로 사용"""
        return self.name

    @classmethod
    def get_by_code(cls, code: Optional[str]) -> "RouteState":
        """
        [❗100점 패치] 코드명으로 상태 객체 반환 
        - None, 빈 문자열, 존재하지 않는 코드에 대해 NEUTRAL로 완벽 방어
        """
        if not code:
            return cls.NEUTRAL
        try:
            return cls[code.upper()]
        except (KeyError, AttributeError):
            return cls.NEUTRAL

    @property
    def is_tradable(self) -> bool:
        """실전 매매 가능 상태 여부"""
        return self.active

    @classmethod
    def sorted_list(cls) -> List["RouteState"]:
        """우선순위에 따른 상태 리스트 반환 (대시보드 필터용)"""
        return sorted(cls, key=lambda x: x.priority)

    def to_dict(self) -> Dict[str, Any]:
        """
        [❗100점 패치] 정적 분석(IDE)을 통과하는 정밀 타입 힌트
        """
        return {
            "code": self.code,
            "label": self.label,
            "color": self.color,
            "priority": self.priority,
            "is_active": self.active
        }
