@echo off
setlocal
cd /d "%~dp0"
title Fish Audio S2 Easy GUI - Self-Contained Installer
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] Installation failed.
  pause
  exit /b 1
)
pause
