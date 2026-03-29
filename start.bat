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

:: Refresh PATH from Registry in case Ninja/CUDA were just installed via winget
for /f "tokens=2*" %%A in ('reg query "HKLM\System\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"
set "PATH=%SYS_PATH%;%USER_PATH%;%PATH%"

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
