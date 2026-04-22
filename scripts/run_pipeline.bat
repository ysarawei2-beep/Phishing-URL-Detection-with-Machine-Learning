@echo off
REM ---------------------------------------------------------------------------
REM One-shot pipeline runner for Windows (cmd.exe / PowerShell).
REM
REM   1. creates a .venv (if missing)
REM   2. installs requirements
REM   3. runs the full training pipeline
REM   4. runs the unit tests
REM
REM Usage:   scripts\run_pipeline.bat   (from the project root)
REM ---------------------------------------------------------------------------
setlocal enabledelayedexpansion

REM Go to project root (parent of this script)
cd /d "%~dp0.."

REM 1. Virtual environment
if not exist ".venv" (
    echo [run_pipeline] Creating virtual environment...
    py -3 -m venv .venv
)

call .venv\Scripts\activate.bat

REM 2. Dependencies
echo [run_pipeline] Installing dependencies...
python -m pip install --upgrade pip >NUL
pip install -r requirements.txt || goto :error

REM 3. Training
echo [run_pipeline] Training model...
python -m src.training.train || goto :error

REM 4. Tests
echo [run_pipeline] Running unit tests...
pytest -q || goto :error

echo [run_pipeline] Done. Results are in .\results and .\models_saved
exit /b 0

:error
echo [run_pipeline] ERROR: step failed.
exit /b 1
