# Setup for Windows (laptop or school PC). No administrator rights needed.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv into the user folder..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing Python and packages from uv.lock..."
uv sync

if (-not (Test-Path "config\local.toml")) {
    Copy-Item "config\local.example.toml" "config\local.toml"
    Write-Host "Created config\local.toml - edit the data paths for this machine."
}

Write-Host "Checking notebooks and tests..."
uv run marimo check notebooks
uv run pytest -q

Write-Host ""
Write-Host "Done. Open the environment check with:"
Write-Host "  uv run marimo edit notebooks\00_environment_check.py"
