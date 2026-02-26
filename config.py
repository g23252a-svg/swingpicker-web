# config.py — 환경변수 검증 (Fail-Fast, Pydantic 전용)
"""
앱 시작 시 필수 환경변수를 검증합니다.
누락 시 Railway 빌드 단계에서 즉시 에러 → 이전 버전 유지

requirements.txt에 반드시 추가:
    pydantic-settings>=2.0

사용법:
    from config import settings
    port = settings.PORT
"""
import sys
import logging

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional

logger = logging.getLogger("config")


class Settings(BaseSettings):
    """환경변수 스키마 — 필수 키 누락 시 앱 시작 불가"""

    # ── 필수 (없으면 앱 시작 불가) ──
    STORAGE_SECRET: str = Field(
        ...,
        min_length=8,
        description="NiceGUI 세션 암호화 키 (최소 8자)",
    )

    # ── 선택 (없으면 해당 기능만 비활성화) ──
    DART_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    LDY_GIST_ID: Optional[str] = None
    LDY_GIST_TOKEN: Optional[str] = None
    MASTER_ADMIN_PW: Optional[str] = None

    # ── 기본값 있음 ──
    PORT: int = 8080
    GEMINI_MODEL: str = "gemini-2.0-flash"
    DART_TIMEOUT: int = 10
    TZ: str = "Asia/Seoul"

    # ── 유효성 검증 ──
    @field_validator("DART_API_KEY")
    @classmethod
    def validate_dart_key(cls, v):
        if v and len(v) < 20:
            raise ValueError("DART_API_KEY가 너무 짧음 (20자 이상)")
        return v

    @field_validator("PORT")
    @classmethod
    def validate_port(cls, v):
        if not (1024 <= v <= 65535):
            raise ValueError(f"PORT 범위 초과: {v} (1024~65535)")
        return v

    model_config = {"env_file": ".env", "case_sensitive": True}

    def feature_flags(self) -> dict:
        """각 기능의 활성화 여부를 한눈에 확인"""
        return {
            "dart": bool(self.DART_API_KEY),
            "gemini": bool(self.GEMINI_API_KEY),
            "gist": bool(self.LDY_GIST_ID and self.LDY_GIST_TOKEN),
            "admin": bool(self.MASTER_ADMIN_PW),
        }


# 앱 시작 시 1번 검증 — 실패하면 여기서 즉시 종료
try:
    settings = Settings()
    logger.info(f"✅ config 검증 완료 — 기능: {settings.feature_flags()}")
except Exception as e:
    logger.critical(f"❌ 환경변수 검증 실패 — 배포 불가: {e}")
    sys.exit(1)
