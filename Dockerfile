# ─────────────────────────────────────────────
# LDY Pro Trader — NiceGUI Docker Image
# Railway 배포 최적화 (v4.0)
# ─────────────────────────────────────────────
FROM python:3.11-slim

# 1. 환경 변수 (로그 실시간 출력 + KST 시간대)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Seoul

# 2. 시스템 패키지 (tzdata 추가)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리
WORKDIR /app

# 4. 의존성 설치 (pip 최신화 → requirements)
COPY requirements_nicegui.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 포트 (Railway가 $PORT로 동적 덮어씀)
ENV PORT=8080
EXPOSE ${PORT}

# 7. 실행
CMD ["python", "main.py"]
