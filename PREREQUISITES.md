# Prerequisites explained

This project can run natively with Python 3.12 and does not require Docker or WSL.

This page explains Docker and WSL anyway, since they are common tools you may encounter in backend projects.

## Docker

### What it is

Docker packages software and its dependencies into containers so it runs consistently across machines.

### Do you need it for this project right now?

No. The current stack is SQLite + ChromaDB and runs locally without Docker.

### When Docker can still help

- You want isolated services for experiments.
- You want reproducible dev environments across a team.
- You want to test future service dependencies without installing them globally.

## WSL2 (Windows only)

### What it is

WSL2 (Windows Subsystem for Linux 2) runs a Linux environment directly on Windows.

### Do you need it for this project right now?

No. Windows native is supported for the current SQLite + Chroma setup.

### When WSL2 can still help

- You prefer Linux tooling and shell workflows.
- You want environment parity with Linux CI or production.
- You hit a Windows-specific native package issue and want a Linux fallback.

### Install command (optional)

Run once in PowerShell, then restart:

```powershell
wsl --install -d Ubuntu
```

After restart, open Ubuntu from the Start menu or use `wsl -d Ubuntu`.

## Required tools for this repository

- Python 3.12
- pip (bundled with Python)
- venv support (bundled with Python; on some Linux distros install python3.12-venv)
