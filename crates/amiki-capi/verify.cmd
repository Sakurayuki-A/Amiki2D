@echo off
REM ============================================================================
REM Amiki2D C ABI -- full verification (five checks)
REM
REM ASCII-only on purpose: cmd.exe parses .cmd using the system code page.
REM What each check proves is documented in verify.ps1 and in README.md.
REM ============================================================================
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify.ps1"
exit /b %errorlevel%
