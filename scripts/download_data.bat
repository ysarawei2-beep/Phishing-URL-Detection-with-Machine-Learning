@echo off
REM ---------------------------------------------------------------------------
REM Download the Kaggle dataset with the Kaggle CLI (Windows).
REM
REM Requirements:
REM   * pip install kaggle   (already in requirements.txt)
REM   * Kaggle API token saved to %USERPROFILE%\.kaggle\kaggle.json
REM     See https://github.com/Kaggle/kaggle-api#api-credentials
REM
REM Usage:  scripts\download_data.bat   (from the project root)
REM ---------------------------------------------------------------------------
setlocal

set DATASET=shashwatwork/phishing-dataset-for-machine-learning
cd /d "%~dp0.."

set DEST=data\raw
if not exist "%DEST%" mkdir "%DEST%"

echo [download_data] Fetching %DATASET% from Kaggle...
kaggle datasets download -d %DATASET% -p %DEST% --unzip
if errorlevel 1 goto :error

echo [download_data] Files now in %DEST%:
dir /b "%DEST%"
exit /b 0

:error
echo [download_data] ERROR: download failed. Make sure you have a Kaggle API token.
exit /b 1
