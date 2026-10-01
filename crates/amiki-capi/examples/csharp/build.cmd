@echo off
REM ============================================================================
REM Amiki2D C ABI -- build & run the C# example
REM
REM   build.cmd          build and run
REM   build.cmd build    build only
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
set "NATIVE=%CD%\target\release"
popd

echo [2/3] dotnet build ...
dotnet build -c Release -p:AmikiNativeDir="%NATIVE%" --nologo -v:quiet
if errorlevel 1 (
    echo [ERROR] dotnet build failed
    exit /b 1
)

if /i "%~1"=="build" (
    echo build OK
    exit /b 0
)

echo [3/3] running ...
echo.
dotnet run -c Release --no-build -p:AmikiNativeDir="%NATIVE%"
if errorlevel 1 (
    echo [ERROR] run failed
    exit /b 1
)

endlocal
