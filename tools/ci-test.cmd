@echo off
REM ============================================================================
REM Amiki2D CI -- job "test"
REM
REM   CI:      .github/workflows/ci.yml calls this file (windows-latest)
REM   locally: tools\ci-test.cmd
REM
REM One source of truth on purpose: before this file, the workflow YAML was the
REM only place the gate sequence existed, so "runs in CI" and "runs on my machine"
REM were two different lists that could drift apart. Now they are the same list.
REM
REM ASCII-only on purpose: cmd.exe parses .cmd using the system code page.
REM
REM Assumes: rustup toolchain with clippy and the wasm32-unknown-unknown target
REM (the workflow installs both; locally: rustup target add wasm32-unknown-unknown).
REM
REM GPU note: GitHub runners have no GPU adapter. The GPU tests skip themselves
REM when GpuRenderer::new() fails, so the CPU path is still fully covered. To
REM reproduce that locally, set AMIKI_BACKENDS=bogus before running this script.
REM ============================================================================
setlocal
cd /d "%~dp0.."

echo [1/6] cargo test --workspace
cargo test --workspace
if errorlevel 1 goto fail

echo [2/6] cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets -- -D warnings
if errorlevel 1 goto fail

echo [3/6] cargo doc --workspace --no-deps (RUSTDOCFLAGS=-D warnings)
set "RUSTDOCFLAGS=-D warnings"
cargo doc --workspace --no-deps
if errorlevel 1 goto fail
set "RUSTDOCFLAGS="

echo [4/6] feature matrix: clippy -D warnings for each capi combination
cargo clippy -p amiki-capi --no-default-features --all-targets -- -D warnings
if errorlevel 1 goto fail
cargo clippy -p amiki-capi --no-default-features --features codec --all-targets -- -D warnings
if errorlevel 1 goto fail
cargo clippy -p amiki-capi --no-default-features --features gpu --all-targets -- -D warnings
if errorlevel 1 goto fail

echo [5/6] feature matrix: the no-default-features test targets must build and pass
cargo test -p amiki-capi --no-default-features
if errorlevel 1 goto fail
cargo test -p amiki-capi --no-default-features --features codec
if errorlevel 1 goto fail

echo [6/6] wasm (CPU only): cargo build --profile wasm-release --target wasm32-unknown-unknown
cargo build -p amiki-capi --profile wasm-release --target wasm32-unknown-unknown --no-default-features
if errorlevel 1 goto fail

echo.
echo CI test job: ALL PASSED
exit /b 0

:fail
echo.
echo CI test job: FAILED
exit /b 1
