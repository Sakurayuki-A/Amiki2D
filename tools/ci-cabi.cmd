@echo off
REM ============================================================================
REM Amiki2D CI -- job "c-abi"
REM
REM   CI:      .github/workflows/ci.yml calls this file (windows-latest)
REM   locally: tools\ci-cabi.cmd
REM
REM Runs crates\amiki-capi\verify.cmd, which is the strongest cross-language check
REM in the project: header compiles as C and C++ at /W4 /WX, every declared symbol
REM is exported, the header version matches Cargo.toml, and the C, C# and Node
REM examples all produce byte-identical output.
REM
REM Prerequisites (the workflow installs them; locally they must already exist):
REM   - MSVC (vcvars64.bat) -- verify.ps1 looks for it
REM   - .NET SDK 10 (the C# example targets net10.0; currently preview)
REM   - Node.js (the Node example uses koffi against the native DLL)
REM
REM A GPU is NOT required: verify.cmd's examples run on whichever backend is
REM available and the byte-comparison is backend-independent.
REM
REM ASCII-only on purpose: cmd.exe parses .cmd using the system code page.
REM ============================================================================
setlocal
cd /d "%~dp0.."

call "crates\amiki-capi\verify.cmd"
if errorlevel 1 goto fail

echo.
echo CI c-abi job: ALL PASSED
exit /b 0

:fail
echo.
echo CI c-abi job: FAILED
exit /b 1
