$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-CheckedCommand {
    param(
        [scriptblock]$Command,
        [string]$FailureMessage
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

function Test-Python312 {
    param(
        [string]$Command,
        [string[]]$Args
    )

    & $Command @Args -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}

$pythonCmd = $null
$pythonArgs = @()
$pythonCandidates = @(
    @{ Cmd = "py"; Args = @("-3.12") },
    @{ Cmd = "python3.12"; Args = @() },
    @{ Cmd = "python3"; Args = @() },
    @{ Cmd = "python"; Args = @() }
)

foreach ($candidate in $pythonCandidates) {
    if (-not (Get-Command $candidate.Cmd -ErrorAction SilentlyContinue)) {
        continue
    }

    if (Test-Python312 -Command $candidate.Cmd -Args $candidate.Args) {
        $pythonCmd = $candidate.Cmd
        $pythonArgs = $candidate.Args
        break
    }
}

if (-not $pythonCmd) {
    throw "Python 3.12 is required but no usable Python 3.12 command was found. Install Python 3.12 and ensure one of 'py -3.12', 'python3.12', 'python3', or 'python' resolves to Python 3.12."
}

if (-not (Test-Path ".venv")) {
    & $pythonCmd @pythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv. Ensure Python includes the venv module and rerun this script."
    }
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment Python was not found at '$venvPython'. Delete .venv and rerun setup, or create it manually with '$pythonCmd $($pythonArgs -join ' ') -m venv .venv'."
}

& $venvPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "The existing .venv is not using Python 3.12. Delete .venv and rerun this script so it can be recreated with Python 3.12."
}

Invoke-CheckedCommand -Command { & $venvPython -m pip install --upgrade pip } -FailureMessage "Failed to upgrade pip."
Invoke-CheckedCommand -Command { & $venvPython -m pip install -r requirements.txt } -FailureMessage "Failed to install dependencies from requirements.txt."
Invoke-CheckedCommand -Command { & $venvPython -c "import asyncio; from app.db.base import init_db; from app.vector_store import ensure_vector_store_ready; asyncio.run(init_db()); ensure_vector_store_ready()" } -FailureMessage "Failed to initialize local database or vector store."

Write-Host ""
Write-Host "Setup complete."
Write-Host "To start the app, run:"
Write-Host "  .\scripts\start.ps1"
