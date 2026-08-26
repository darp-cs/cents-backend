$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    throw "Python virtual environment not found. Run .\scripts\setup.ps1 first."
}

. (Join-Path $root ".venv\Scripts\Activate.ps1")
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
