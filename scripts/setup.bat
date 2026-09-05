@echo off
setlocal

set ROOT=%~dp0..\
cd /d "%ROOT%"

set PY_CMD=
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
    if not errorlevel 1 set PY_CMD=py -3.12
)

if not defined PY_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
        if not errorlevel 1 set PY_CMD=python
    )
)

if not defined PY_CMD (
    echo Python 3.12 is required but no usable Python 3.12 command was found.
    echo Install Python 3.12 and ensure py -3.12 or python resolves to Python 3.12.
    exit /b 1
)

if not exist .venv\Scripts\python.exe (
    %PY_CMD% -m venv .venv
)

if not exist .venv\Scripts\python.exe (
    echo Failed to create .venv. Ensure Python includes venv support.
    exit /b 1
)

.venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo The existing .venv is not using Python 3.12.
    echo Delete .venv and rerun this script to recreate it with Python 3.12.
    exit /b 1
)

.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    exit /b 1
)

.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies from requirements.txt.
    exit /b 1
)

.venv\Scripts\python.exe -c "import asyncio; from app.db.base import init_db; from app.vector_store import ensure_vector_store_ready; asyncio.run(init_db()); ensure_vector_store_ready()"
if errorlevel 1 (
    echo Failed to initialize local database or vector store.
    exit /b 1
)

echo.
echo Setup complete.
echo To start the app, run:
echo   scripts\start.bat
endlocal
