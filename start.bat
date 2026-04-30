@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: Performance Optimizations for OpenMP (Threading Affinity)
:: This ensures threads stay on the fastest cores and prevents skipping.
set OMP_PROC_BIND=TRUE
set OMP_PLACES=CORES
set OMP_WAIT_POLICY=PASSIVE
set KMP_BLOCKTIME=0

echo.
echo =======================================================
echo   Fish Speech S2 Pro - GUI
echo =======================================================
echo Checking environment...

set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%APPDATA%\uv\bin;%LOCALAPPDATA%\uv\bin;%LOCALAPPDATA%\Programs\uv;%PATH%"

where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] uv not found in PATH. Run install.bat first, then reopen this terminal.
    pause
    exit /b 1
)

echo Checking for Ninja...

where ninja >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Ninja not found in PATH. torch.compile may be slow or fail.
    echo           Run install.bat if you haven't yet, or restart your terminal.
) else (
    echo [OK] Ninja found.
)

echo Starting application...
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found. Please run install.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
uv run app.py
pause
