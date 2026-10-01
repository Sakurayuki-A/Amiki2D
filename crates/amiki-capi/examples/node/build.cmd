@echo off
REM ============================================================================
REM Amiki2D C ABI -- build & run the Node.js example
REM
REM   build.cmd          build and run
REM   build.cmd build    build only (npm install + cargo build)
REM
REM ASCII-only on purpose: cmd.exe parses .cmd using the system code page.
REM ============================================================================
setlocal

cd /d "%~dp0"

echo [1/3] cargo build -p amiki-capi --release ...
pushd ..\..\..\..
cargo build -p amiki-capi --release
if errorlevel 1 (
    echo [ERROR] cargo build failed
    popd
    exit /b 1
)
set "AMIKI_LIB_DIR=%CD%\target\release"
popd

echo [2/3] npm install ...
if not exist node_modules (
    call npm install --no-audit --no-fund
    if errorlevel 1 (
        echo [ERROR] npm install failed
        exit /b 1
    )
) else (
    echo       node_modules already present, skipping
)

if /i "%~1"=="build" (
    echo build OK
    exit /b 0
)

echo [3/3] running ...
echo.
node minimal.js
if errorlevel 1 (
    echo [ERROR] run failed
    exit /b 1
)

if not exist amiki_node_demo.bmp (
    echo [ERROR] amiki_node_demo.bmp was not produced
    exit /b 1
)
for %%F in (amiki_node_demo.bmp) do echo output image: %%~zF bytes

endlocal
