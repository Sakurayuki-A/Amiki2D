@echo off
REM ============================================================================
REM Amiki2D WASM example -- build & run
REM
REM   build.cmd          build the wasm module and run the Node demo
REM   build.cmd build    build only
REM
REM ASCII-only on purpose: cmd.exe parses .cmd using the system code page.
REM ============================================================================
setlocal

cd /d "%~dp0"

set "WASM=..\..\..\..\target\wasm32-unknown-unknown\wasm-release\amiki2d.wasm"

echo [1/3] ensuring the wasm32 target is installed ...
rustup target list --installed | findstr /C:"wasm32-unknown-unknown" >nul
if errorlevel 1 (
    echo       installing wasm32-unknown-unknown ...
    rustup target add wasm32-unknown-unknown
    if errorlevel 1 (
        echo [ERROR] could not install the wasm32-unknown-unknown target
        exit /b 1
    )
)

echo [2/3] cargo build --profile wasm-release --target wasm32-unknown-unknown --no-default-features ...
pushd ..\..\..\..
cargo build -p amiki-capi --profile wasm-release --target wasm32-unknown-unknown --no-default-features
if errorlevel 1 (
    echo [ERROR] wasm build failed
    popd
    exit /b 1
)
popd

if not exist "%WASM%" (
    echo [ERROR] %WASM% was not produced
    exit /b 1
)
copy /y "%WASM%" amiki2d.wasm >nul
for %%F in (amiki2d.wasm) do echo       module size: %%~zF bytes

if /i "%~1"=="build" (
    echo build OK
    exit /b 0
)

REM Run the demo, but also produce a NATIVE CPU reference and byte-compare it.
REM That comparison is the real verification: same scene, different target
REM architecture and a different ABI lowering, must give identical pixels.
echo [3/3] running the Node demo ...
echo.
node minimal.js amiki2d.wasm
if errorlevel 1 (
    echo [ERROR] node demo failed
    exit /b 1
)

if /i not "%~1"=="nocompare" (
    echo.
    echo [verify] building a native CPU reference and byte-comparing ...
    pushd ..\c
    call build.cmd build >nul 2>&1
    if errorlevel 1 (
        echo [WARN] could not build the C example; skipping the comparison
    ) else (
        set "AMIKI_BACKEND=cpu"
        minimal.exe >nul 2>&1
        set "AMIKI_BACKEND="
        if exist amiki_c_demo.bmp (
            fc /b amiki_c_demo.bmp ..\wasm\amiki_wasm_demo.bmp >nul 2>&1
            if errorlevel 1 (
                echo [FAIL] native CPU and WASM outputs DIFFER
                popd
                exit /b 1
            ) else (
                echo [OK]   native CPU and WASM outputs are BYTE-IDENTICAL
            )
        )
    )
    popd
)

endlocal
