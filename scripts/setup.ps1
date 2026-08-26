$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is required but was not found on PATH. Install Docker Desktop and rerun this script."
}

$pythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} else {
    throw "Python 3.12 is required but was not found on PATH. Install Python 3.12 and rerun this script."
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

if (-not (Test-Path ".venv")) {
    & $pythonCmd -m venv .venv
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

docker compose up -d
& $venvPython -m alembic upgrade head

Write-Host ""
Write-Host "Setup complete."
Write-Host "To start the app, run:"
Write-Host "  .\scripts\start.ps1"
