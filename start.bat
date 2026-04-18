@echo off
setlocal enabledelayedexpansion

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

:: Refreshing PATH from registry is causing "Line too long" errors due to doubling entries.
:: Normal terminal behavior already includes these paths. If you just installed something,
:: please restart your terminal instead.
echo Checking for Ninja...

where ninja >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Ninja not found in PATH. torch.compile may be slow or fail.
    echo           Run install.bat if you haven't yet, or restart your terminal.
) else (
    echo [OK] Ninja found.
)

echo Starting application...
call .venv\Scripts\activate.bat
uv run app.py
pause
