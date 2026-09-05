$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python virtual environment not found. Run .\scripts\setup.ps1 first."
}

& (Join-Path $root ".venv\Scripts\python.exe") -m uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000
