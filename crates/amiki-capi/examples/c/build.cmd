@echo off
REM ============================================================================
REM Amiki2D C ABI -- build & run the C example
REM
REM   build.cmd          build and run
REM   build.cmd build    build only
REM
REM NOTE: this file is deliberately ASCII-only. cmd.exe parses .cmd files using
REM the system code page (936 here), so non-ASCII comments corrupt the script.
REM The Chinese explanation lives in README.md next to this file.
REM ============================================================================
setlocal

cd /d "%~dp0"

REM ---- 1. MSVC environment ---------------------------------------------------
REM Deliberately written as top-level statements (no nested blocks): inside a
REM parenthesised block, %VCVARS% would be expanded at parse time and be empty.
if defined VCINSTALLDIR goto :msvc_ready

REM Ask vswhere.exe first. Hardcoding the VS edition folders is what broke the
REM first real CI run -- the runner's Visual Studio is under none of them, so
REM "vcvars64.bat not found" failed the C example, the header compile check and
REM the symbol check all at once. vswhere ships with every VS 2017+ installer at
REM a fixed path, so it is the layout-independent answer. The hardcoded list
REM below stays as a fallback for machines without it.
set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat"`) do if not defined VCVARS set "VCVARS=%%I"
if defined VCVARS if not exist "%VCVARS%" set "VCVARS="

if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if not defined VCVARS (
    echo [ERROR] vcvars64.bat not found.
    echo         Tried vswhere.exe, VCINSTALLDIR and the usual 2019/2022 folders.
    echo         Run this from an "x64 Native Tools Command Prompt", or set VCINSTALLDIR.
    exit /b 1
)
echo [1/4] loading MSVC environment ...
call "%VCVARS%" >nul
if errorlevel 1 (
    echo [ERROR] failed to load MSVC environment
    exit /b 1
)

:msvc_ready
if not defined VCToolsInstallDir (
    echo [ERROR] MSVC environment is incomplete - run from an "x64 Native Tools" prompt.
    exit /b 1
)

REM ---- 2. build the Rust cdylib ----------------------------------------------
echo [2/4] cargo build -p amiki-capi --release ...
pushd ..\..\..\..
cargo build -p amiki-capi --release
if errorlevel 1 (
    echo [ERROR] cargo build failed
    popd
    exit /b 1
)
set "TARGET=%CD%\target\release"
popd

REM ---- 3. compile and link ---------------------------------------------------
echo [3/4] compiling minimal.c ...
cl /nologo /utf-8 /W4 /WX /I..\..\include minimal.c /Fe:minimal.exe /Fo:minimal.obj /link /LIBPATH:"%TARGET%" amiki2d.dll.lib
if errorlevel 1 (
    echo [ERROR] C compile or link failed
    exit /b 1
)

copy /y "%TARGET%\amiki2d.dll" . >nul

if /i "%~1"=="build" (
    echo build OK: minimal.exe
    exit /b 0
)

REM ---- 4. run ----------------------------------------------------------------
echo [4/4] running ...
echo.
minimal.exe
if errorlevel 1 (
    echo [ERROR] run failed
    exit /b 1
)

if not exist amiki_c_demo.bmp (
    echo [ERROR] amiki_c_demo.bmp was not produced
    exit /b 1
)
for %%F in (amiki_c_demo.bmp) do echo output image: %%~zF bytes

endlocal
