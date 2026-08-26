@echo off
setlocal

set ROOT=%~dp0..\
cd /d "%ROOT%"

if not exist .venv\Scripts\activate.bat (
    echo Virtual environment not found. Run scripts\setup.bat first.
    exit /b 1
)

call .venv\Scripts\activate.bat
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
endlocal
