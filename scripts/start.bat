@echo off
setlocal

set ROOT=%~dp0..\
cd /d "%ROOT%"

if not exist .venv\Scripts\python.exe (
    echo Virtual environment not found. Run scripts\setup.bat first.
    exit /b 1
)

.venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000
endlocal
