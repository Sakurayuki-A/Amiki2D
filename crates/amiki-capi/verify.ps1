# ============================================================================
# Amiki2D C ABI -- full verification
#
# Run it with:   verify.cmd
#
# ASCII-only on purpose: Windows PowerShell 5.1 reads a BOM-less UTF-8 file as
# ANSI, so non-ASCII comments would be mangled. See README.md for the Chinese
# explanation of what each step proves.
#
# Five checks, each one is something a broken C ABI would fail:
#   1. the crate builds
#   2. amiki.h compiles as C AND as C++ at /W4 /WX   (catches malformed decls,
#      missing types, and broken extern "C" guards)
#   3. every function declared in amiki.h is actually exported by the library
#      (catches forgotten #[no_mangle] / name mismatches)
#   4. the three binding examples (C / C# / Node) build and run
#   5. the three examples produce BYTE-IDENTICAL images
#      (catches struct layout, calling convention, string encoding and
#       ownership mistakes -- the whole point of the C ABI)
# ============================================================================

# Deliberately NOT 'Stop': PowerShell 5.1 turns a native command's stderr into a
# terminating error record, and cargo writes its progress to stderr -- with
# 'Stop' the very first cargo call would abort the script. Native steps are
# checked through $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'

# $PSScriptRoot is <repo>/crates/amiki-capi -> the repo root is two levels up.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$capi = Join-Path $root 'crates\amiki-capi'
$inc = Join-Path $capi 'include'
$ex = Join-Path $capi 'examples'
$target = Join-Path $root 'target\release'

$fail = 0
function Step($n, $msg) { Write-Host ""; Write-Host "[$n] $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    OK  $msg" -ForegroundColor Green }
function Bad($msg) { Write-Host "    FAIL  $msg" -ForegroundColor Red; $script:fail++ }

# --- locate MSVC -------------------------------------------------------------
$vcvars = $null
foreach ($p in @(
    'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat',
    'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat',
    'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat'
)) { if (Test-Path $p) { $vcvars = $p; break } }

# --- 1. build -----------------------------------------------------------------
Step 1 'cargo build -p amiki-capi --release'
Push-Location $root
$null = cmd /c "cargo build -p amiki-capi --release 2>&1"
if ($LASTEXITCODE -ne 0) { Bad 'cargo build failed' } else { Ok 'built' }
Pop-Location

$dll = Join-Path $target 'amiki2d.dll'
if (-not (Test-Path $dll)) { Bad "missing $dll" } else { Ok "amiki2d.dll ($([math]::Round((Get-Item $dll).Length/1MB,1)) MB)" }

# --- 2. header compiles as C and C++ -----------------------------------------
Step 2 'amiki.h compiles as C and C++ (/W4 /WX)'
if (-not $vcvars) {
    Bad 'vcvars64.bat not found -- skipping the compile check'
} else {
    $tmp = Join-Path $env:TEMP 'amiki_hdr_check'
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    foreach ($mode in @('C', 'CPP')) {
        $tp = if ($mode -eq 'C') { '' } else { '/TP' }
        $cmd = "`"$vcvars`" >nul 2>&1 && cl /nologo /utf-8 /W4 /WX $tp /I`"$inc`" /c `"$inc\test_header.c`" /Fo:`"$tmp\h_$mode.obj`""
        $out = cmd /c $cmd 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { Bad "$mode mode: $out" } else { Ok "$mode mode" }
    }
}

# --- 3. every declared symbol is exported ------------------------------------
Step 3 'every function declared in amiki.h is exported'
if (-not $vcvars) {
    Bad 'vcvars64.bat not found -- cannot run dumpbin'
} else {
    $dump = cmd /c "`"$vcvars`" >nul 2>&1 && dumpbin /exports `"$dll`" 2>&1" | Out-String
    $exported = [regex]::Matches($dump, '\b(amiki_[a-z0-9_]+)\b') |
                ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
    $declared = (Get-Content (Join-Path $inc 'amiki.h')) |
                ForEach-Object { [regex]::Match($_, '^AMIKI_API\s+[\w\s\*]+?\b(amiki_[a-z0-9_]+)\s*\(') } |
                Where-Object { $_.Success } |
                ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
    $missing = $declared | Where-Object { $exported -notcontains $_ }
    if ($missing) {
        Bad ("declared but not exported: " + ($missing -join ', '))
    } elseif ($declared.Count -eq 0) {
        Bad 'parsed 0 declarations from amiki.h -- the regex is wrong, not the ABI'
    } else {
        Ok "$($declared.Count) declared / $($exported.Count) exported, none missing"
    }
}

# --- 4. the three examples build and run -------------------------------------
Step 4 'three binding examples (C / C# / Node) build and run'
$bmp = @{
    'C'    = Join-Path $ex 'c\amiki_c_demo.bmp'
    'C#'   = Join-Path $ex 'csharp\amiki_csharp_demo.bmp'
    'Node' = Join-Path $ex 'node\amiki_node_demo.bmp'
}
foreach ($k in @('C', 'C#', 'Node')) {
    $dir = switch ($k) { 'C' { 'c' } 'C#' { 'csharp' } 'Node' { 'node' } }
    $wd = Join-Path $ex $dir
    if (Test-Path $bmp[$k]) { Remove-Item $bmp[$k] -Force }
    $out = cmd /c "cd /d `"$wd`" && build.cmd 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Bad "$k example failed`n$out"
    } elseif (-not (Test-Path $bmp[$k])) {
        Bad "$k example produced no image"
    } else {
        Ok "$k example"
    }
}

# --- 5. byte-identical output -------------------------------------------------
Step 5 'the three examples produce byte-identical images'
$hashes = @{}
foreach ($k in @('C', 'C#', 'Node')) {
    if (Test-Path $bmp[$k]) { $hashes[$k] = (Get-FileHash $bmp[$k] -Algorithm SHA256).Hash }
}
if ($hashes.Count -lt 3) {
    Bad "only $($hashes.Count) of 3 images exist"
} elseif (($hashes.Values | Sort-Object -Unique).Count -eq 1) {
    Ok "all three identical -- SHA256 $($hashes['C'].Substring(0,32))..."
} else {
    foreach ($k in $hashes.Keys) { Write-Host "      $k  $($hashes[$k].Substring(0,32))..." }
    Bad 'images differ -- the ABI marshalling is wrong somewhere'
}

# --- summary ------------------------------------------------------------------
Write-Host ""
if ($fail -eq 0) {
    Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host "$fail CHECK(S) FAILED" -ForegroundColor Red
    exit 1
}
