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
REM One number in check 5 depends on the backend, so do not treat it as fixed:
REM on the author's RTX 3050 (GPU path) the three examples hash to 90A8B3B8...,
REM while a GPU-less machine (CPU path, which is what a runner is) was measured at
REM FC9482BA... with the three languages still identical to each other. What this
REM check pins is "the three agree", not one particular digest; the CHANGELOG
REM records the GPU digest per release.
REM
REM Two details here are deliberately defensive, both learned the hard way:
REM
REM   1. It calls verify.cmd by its ABSOLUTE path ("%~dp0..\..."). With a quoted
REM      RELATIVE path, cmd computes %~dp0 inside the callee by joining the
REM      invocation path onto the current directory, so verify.ps1 was looked up
REM      as "...\crates\amiki-capi\crates\amiki-capi\verify.ps1" and never ran.
REM   2. It then requires the literal line "ALL CHECKS PASSED" in the captured
REM      output. The exit code alone was NOT enough: in the broken run above cmd
REM      reported success, this wrapper printed "ALL PASSED", and the C ABI went
REM      unverified. A gate that can pass without running is worse than no gate,
REM      so the success marker is checked explicitly.
REM
REM ASCII-only on purpose: cmd.exe parses .cmd using the system code page.
REM ============================================================================
setlocal
cd /d "%~dp0.."
set "LOG=%TEMP%\amiki_ci_cabi.log"

call "%~dp0..\crates\amiki-capi\verify.cmd" > "%LOG%" 2>&1
set "RC=%errorlevel%"
type "%LOG%"

if not "%RC%"=="0" goto fail
findstr /C:"ALL CHECKS PASSED" "%LOG%" >nul
if errorlevel 1 goto nosentinel

echo.
echo CI c-abi job: ALL PASSED
exit /b 0

:nosentinel
echo.
echo CI c-abi job: FAILED - verify.cmd exited 0 but never printed "ALL CHECKS PASSED"
echo (that combination means the checks did not actually run)
exit /b 1

:fail
echo.
echo CI c-abi job: FAILED (verify.cmd exit %RC%)
exit /b 1
