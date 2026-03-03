@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ═══════════════════════════════════════════════════
:: SwingPicker Auto Collector (Windows)
:: 위치: C:\Users\g2325\Downloads\swingpicker-web\run_collector.bat
:: ═══════════════════════════════════════════════════

set PROJ_DIR=C:\Users\g2325\Downloads\swingpicker-web
set LOG_DIR=%PROJ_DIR%\logs

:: 로그 디렉토리 생성
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: 날짜/시간 변수
for /f "tokens=1-3 delims=/" %%a in ("%date%") do set YMD=%%a%%b%%c
for /f "tokens=1-2 delims=:." %%a in ("%time: =0%") do set HMS=%%a%%b
set LOG_FILE=%LOG_DIR%\collect_%YMD%_%HMS%.log

:: 로그 시작
echo ═══════════════════════════════════════════════ >> "%LOG_FILE%"
echo  SwingPicker Collector 시작: %date% %time% >> "%LOG_FILE%"
echo ═══════════════════════════════════════════════ >> "%LOG_FILE%"

cd /d "%PROJ_DIR%"

:: ── .env 환경변수 로드 ──
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "first=%%a"
        if not "!first:~0,1!"=="#" (
            if not "%%b"=="" set "%%a=%%b"
        )
    )
    echo [OK] .env 로드 완료 >> "%LOG_FILE%"
) else (
    echo [ERROR] .env 파일이 없습니다! >> "%LOG_FILE%"
    goto :END
)

:: ── 최신 코드 풀 ──
echo [PULL] git pull... >> "%LOG_FILE%"
git fetch origin main >> "%LOG_FILE%" 2>&1
git reset --hard origin/main >> "%LOG_FILE%" 2>&1

:: ── 수집 실행 ──
echo [RUN] collector.py 실행... >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

python collector.py >> "%LOG_FILE%" 2>&1
set COLLECT_EXIT=%errorlevel%

if %COLLECT_EXIT%==0 (
    echo [OK] 수집 완료 >> "%LOG_FILE%"
) else (
    echo [FAIL] 수집 실패 (exit: %COLLECT_EXIT%) >> "%LOG_FILE%"
)

:: ── 결과 확인 ──
python -c "import pandas as pd; df=pd.read_csv('data/recommend_latest.csv',dtype={'종목코드':str}); print(f'총 {len(df)}건, 시장: {dict(df[\"시장\"].value_counts())}, 오염: {df[\"종목명\"].astype(str).str.match(r\"^\d+$\").sum()}건')" >> "%LOG_FILE%" 2>&1

:: ── Git 커밋 & 푸시 ──
echo [PUSH] Git 푸시... >> "%LOG_FILE%"

git add -f data\*.csv data\*.json data\*.pkl data\*.pth 2>nul
git add -f data\*.parquet data\*.duckdb 2>nul

git diff --staged --quiet
if %errorlevel%==1 (
    git commit -m "chore(local): collect %YMD%" >> "%LOG_FILE%" 2>&1

    git fetch origin main >> "%LOG_FILE%" 2>&1
    git merge origin/main --no-edit -X ours --allow-unrelated-histories >> "%LOG_FILE%" 2>&1

    git push origin main >> "%LOG_FILE%" 2>&1
    if !errorlevel!==0 (
        echo [OK] Git 푸시 성공 >> "%LOG_FILE%"
    ) else (
        echo [FAIL] Git 푸시 실패 >> "%LOG_FILE%"
    )
) else (
    echo [SKIP] 변경사항 없음 >> "%LOG_FILE%"
)

:: ── 7일 이상 된 로그 삭제 ──
forfiles /P "%LOG_DIR%" /M "collect_*.log" /D -7 /C "cmd /c del @path" 2>nul

:END
echo ═══════════════════════════════════════════════ >> "%LOG_FILE%"
echo  완료: %date% %time% >> "%LOG_FILE%"
echo ═══════════════════════════════════════════════ >> "%LOG_FILE%"

endlocal
