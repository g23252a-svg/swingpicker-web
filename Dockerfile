# ─────────────────────────────────────────────
# LDY Pro Trader — NiceGUI Docker Image
# Railway/Render/Fly.io 배포용
# ─────────────────────────────────────────────
FROM python:3.11-slim

# 시스템 패키지 (최소)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (캐시 활용)
COPY requirements_nicegui.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# Railway는 PORT 환경변수를 자동 주입
ENV PORT=8080
EXPOSE 8080

# NiceGUI 실행
CMD ["python", "main.py"]
