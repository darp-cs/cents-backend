@echo off
setlocal

set ROOT=%~dp0..\
cd /d "%ROOT%"

where docker >nul 2>nul
if errorlevel 1 (
    echo Docker is required but was not found on PATH.
    echo Install Docker Desktop and rerun this script.
    exit /b 1
)

where py >nul 2>nul
if not errorlevel 1 (
    set PYTHON_CMD=py
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python 3.12 is required but was not found on PATH.
        echo Install Python 3.12 and rerun this script.
        exit /b 1
    )
    set PYTHON_CMD=python
)

if not exist .env (
    copy .env.example .env
    echo Created .env from .env.example
)

if not exist .venv (
    %PYTHON_CMD% -m venv .venv
)

.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d
.venv\Scripts\python.exe -m alembic upgrade head

echo.
echo Setup complete.
echo To start the app, run:
echo   scripts\start.bat
endlocal
