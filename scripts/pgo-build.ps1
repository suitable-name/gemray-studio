<#
.SYNOPSIS
  Profile-guided-optimization (PGO) release builds of diagram-gui and gemray-worker
  on Windows, over two axes:

    CPU:  avx2   -> -C target-cpu=x86-64-v3 (AVX2+FMA+BMI baseline; Haswell/Zen or newer)
          scalar -> -C target-cpu=x86-64    (portable baseline; SIMD kernels still runtime-dispatch)
    GPU:  gpu    -> diagram-gui/gpu + gemray-worker/gpu (wgpu megakernel, CPU fallback per frame)
          cpu    -> no gpu feature at all (pure CPU tracer binaries)

.DESCRIPTION
  Per combination, in its own target directory (target\pgo-<cpu>-<gpu>):
    1. build the training example `pgo_train` instrumented with -C profile-generate,
       with the same gemray features the final binaries will pull in
    2. run it (CPU tracer, denoiser, tone-map, meet-point solver; no GPU, no DB)
    3. merge the .profraw files with llvm-profdata (rustup component llvm-tools-preview)
    4. rebuild diagram-gui + gemray-worker with -C profile-use=<merged profile>
    5. copy the executables to <OutDir> as diagram-gui-<cpu>-<gpu>.exe / gemray-worker-<cpu>-<gpu>.exe

  The scalar variants train with GEMRAY_SIMD=scalar so the scalar kernels get a
  profile; at run time every binary still auto-detects AVX2/AVX-512 unless you set
  GEMRAY_SIMD yourself.

  --target is passed explicitly so RUSTFLAGS reach only target code (build scripts and
  proc-macros stay uninstrumented). The release profile (lto = true, codegen-units = 1)
  plus PGO makes each final GUI build slow -- expect several minutes per combination.

.PARAMETER Cpu
  avx2, scalar, or both (default both).
.PARAMETER Gpu
  gpu, cpu, or both (default both).
.PARAMETER OutDir
  Where the renamed executables are copied (default <repo>\target\pgo-out).
.PARAMETER SkipTrain
  Reuse an existing merged.profdata in each combination's target directory.

.EXAMPLE
  .\scripts\pgo-build.ps1                          # all four combinations
  .\scripts\pgo-build.ps1 -Cpu avx2 -Gpu gpu       # one build
  .\scripts\pgo-build.ps1 -Cpu scalar -Gpu cpu     # portable, CPU-only
#>
[CmdletBinding()]
param(
    [ValidateSet('avx2', 'scalar', 'both')]
    [string]$Cpu = 'both',
    [ValidateSet('gpu', 'cpu', 'both')]
    [string]$Gpu = 'both',
    [string]$OutDir = '',
    [switch]$SkipTrain
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$target = 'x86_64-pc-windows-msvc'
if (-not $OutDir) { $OutDir = Join-Path $root 'target\pgo-out' }

# --- llvm-profdata from the active toolchain --------------------------------------
$sysroot = (& rustc --print sysroot).Trim()
$profdata = Join-Path $sysroot "lib\rustlib\$target\bin\llvm-profdata.exe"
if (-not (Test-Path $profdata)) {
    Write-Host "llvm-tools-preview not installed; adding it via rustup..."
    & rustup component add llvm-tools-preview
    if (-not (Test-Path $profdata)) { throw "llvm-profdata.exe not found at $profdata" }
}
$installedTargets = & rustup target list --installed
if (-not ($installedTargets -contains $target)) { & rustup target add $target }

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Command)
    Write-Host "==> $Description" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Description failed (exit $LASTEXITCODE)" }
}

function Build-PgoCombination {
    param(
        [string]$CpuName,   # avx2 | scalar
        [string]$TargetCpu, # x86-64-v3 | x86-64
        [string]$SimdCap,   # '' | scalar
        [string]$GpuName    # gpu | cpu
    )

    $name = "$CpuName-$GpuName"
    $tdir = Join-Path $root "target\pgo-$name"
    $pdir = Join-Path $tdir 'profiles'
    $merged = Join-Path $tdir 'merged.profdata'
    $cpuFlag = "-C target-cpu=$TargetCpu"

    # gemray features the apps pull in: diagram-gui -> hdr (+gpu), gemray-worker -> serde (+gpu).
    # The training build uses the same set so the profiled function bodies match the final build.
    if ($GpuName -eq 'gpu') {
        $trainFeatures = 'hdr,serde,gpu'
        $appFeatures = @('--features', 'diagram-gui/gpu,gemray-worker/gpu,gemray-worker/worker')
    } else {
        $trainFeatures = 'hdr,serde'
        $appFeatures = @('--features', 'gemray-worker/worker')
    }

    $savedRustflags = $env:RUSTFLAGS
    $savedSimd = $env:GEMRAY_SIMD
    $savedProfileFile = $env:LLVM_PROFILE_FILE

    try {
        Push-Location $root
        if (-not $SkipTrain -or -not (Test-Path $merged)) {
            if (Test-Path $pdir) { Remove-Item -Recurse -Force $pdir }
            New-Item -ItemType Directory -Force $pdir | Out-Null

            # 1. Instrumented training build.
            $env:RUSTFLAGS = "$cpuFlag -C profile-generate=$pdir"
            Invoke-Checked "[$name] instrumented build of pgo_train (gemray features: $trainFeatures)" {
                cargo build --release --target $target --target-dir $tdir `
                    -p gemray --features $trainFeatures --example pgo_train
            }

            # 2. Train. LLVM_PROFILE_FILE names the raw profiles; %p = pid, %m = module hash.
            $env:LLVM_PROFILE_FILE = Join-Path $pdir 'train-%p-%m.profraw'
            if ($SimdCap) { $env:GEMRAY_SIMD = $SimdCap } else { Remove-Item Env:GEMRAY_SIMD -ErrorAction SilentlyContinue }
            $exe = Join-Path $tdir "$target\release\examples\pgo_train.exe"
            Invoke-Checked "[$name] training run (GEMRAY_SIMD='$SimdCap')" { & $exe }

            # 3. Merge.
            $raw = @(Get-ChildItem -Path $pdir -Filter '*.profraw')
            if ($raw.Count -eq 0) { throw "no .profraw files were written to $pdir" }
            Invoke-Checked "[$name] merging $($raw.Count) profile(s)" {
                & $profdata merge -o $merged $pdir
            }
        } else {
            Write-Host "[$name] reusing $merged"
        }

        # 4. Optimized build of the real binaries with the merged profile.
        # (No `-C llvm-args=-pgo-warn-missing-function`: it reports every function the
        # training run never executed -- tens of thousands of generated UI functions --
        # which is expected here and only drowns real warnings.)
        $env:RUSTFLAGS = "$cpuFlag -C profile-use=$merged"
        Remove-Item Env:LLVM_PROFILE_FILE -ErrorAction SilentlyContinue
        Remove-Item Env:GEMRAY_SIMD -ErrorAction SilentlyContinue
        Invoke-Checked "[$name] PGO release build of diagram-gui + gemray-worker ($($appFeatures[1]))" {
            cargo build --release --target $target --target-dir $tdir `
                -p diagram-gui -p gemray-worker @appFeatures
        }

        # 5. Copy out with distinguishing names.
        New-Item -ItemType Directory -Force $OutDir | Out-Null
        $releaseDir = Join-Path $tdir "$target\release"
        foreach ($bin in @('diagram-gui', 'gemray-worker')) {
            $src = Join-Path $releaseDir "$bin.exe"
            if (-not (Test-Path $src)) { throw "expected binary missing: $src" }
            $dst = Join-Path $OutDir "$bin-$name.exe"
            Copy-Item -Force $src $dst
            Write-Host ("   {0}  ({1:N1} MB)" -f $dst, ((Get-Item $dst).Length / 1MB)) -ForegroundColor Green
        }
        Write-Host ""
    }
    finally {
        Pop-Location
        if ($null -ne $savedRustflags) { $env:RUSTFLAGS = $savedRustflags } else { Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue }
        if ($null -ne $savedSimd) { $env:GEMRAY_SIMD = $savedSimd } else { Remove-Item Env:GEMRAY_SIMD -ErrorAction SilentlyContinue }
        if ($null -ne $savedProfileFile) { $env:LLVM_PROFILE_FILE = $savedProfileFile } else { Remove-Item Env:LLVM_PROFILE_FILE -ErrorAction SilentlyContinue }
    }
}

$cpuList = if ($Cpu -eq 'both') { @('avx2', 'scalar') } else { @($Cpu) }
$gpuList = if ($Gpu -eq 'both') { @('gpu', 'cpu') } else { @($Gpu) }

foreach ($c in $cpuList) {
    $targetCpu = if ($c -eq 'avx2') { 'x86-64-v3' } else { 'x86-64' }
    $simdCap = if ($c -eq 'scalar') { 'scalar' } else { '' }
    foreach ($g in $gpuList) {
        Build-PgoCombination -CpuName $c -TargetCpu $targetCpu -SimdCap $simdCap -GpuName $g
    }
}

Write-Host "All requested PGO builds are in $OutDir" -ForegroundColor Green
