#!/usr/bin/env bash
#
# Profile-guided-optimization (PGO), optionally followed by BOLT, for diagram-gui and
# gemray-worker. Supports sequential execution or parallel dispatch across all tiers.
#
# PGO training runs natively on the host target to generate target-agnostic LLVM profile
# data, allowing clean cross-compilation (e.g. Linux -> Windows) without Wine or binfmt_misc.
#
# Usage:
#   scripts/pgo-bolt-build.sh                                # sequential, host OS, all tiers
#   scripts/pgo-bolt-build.sh --os windows                   # sequential cross-compilation for Windows
#   scripts/pgo-bolt-build.sh --os windows --parallel        # parallel cross-compilation for Windows
#   scripts/pgo-bolt-build.sh --isa avx2 --gpu gpu           # single combination
#   scripts/pgo-bolt-build.sh --bolt-scene bench/scene.json  # PGO + BOLT (Linux only)
#
set -euo pipefail

# Disable sccache/cache wrappers. They provide 0% cache hit rate for PGO builds
# (since the profile data changes on every run) and sccache frequently crashes
# when trying to rapidly concurrently hash large .profdata files.
export RUSTC_WRAPPER=""
export RUSTC_WORKSPACE_WRAPPER=""

# ---------------------------------------------------------------------------------
# Argument defaults & parsing
# ---------------------------------------------------------------------------------
ISA_ARG=all
GPU_ARG=both
OS_ARG=auto
OUT_DIR=""
SKIP_TRAIN=0
BOLT_SCENE=""
BOLT_SAMPLES=64
NO_BOLT=0
PARALLEL=0

die() { printf '%s\n' "error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --isa)          ISA_ARG="${2:?--isa needs a value}"; shift 2 ;;
        --gpu)          GPU_ARG="${2:?--gpu needs a value}"; shift 2 ;;
        --os)           OS_ARG="${2:?--os needs a value}"; shift 2 ;;
        --out-dir)      OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --bolt-scene)   BOLT_SCENE="${2:?--bolt-scene needs a value}"; shift 2 ;;
        --bolt-samples) BOLT_SAMPLES="${2:?--bolt-samples needs a value}"; shift 2 ;;
        --skip-train)   SKIP_TRAIN=1; shift ;;
        --no-bolt)      NO_BOLT=1; shift ;;
        --parallel)     PARALLEL=1; shift ;;
        -h|--help)      sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              die "unknown argument: $1 (try --help)" ;;
    esac
done

case "$ISA_ARG" in
    all)                  ISA_LIST=(avx512 avx2 scalar) ;;
    avx512|avx2|scalar)   ISA_LIST=("$ISA_ARG") ;;
    *)                    die "--isa must be avx512, avx2, scalar, or all" ;;
esac

case "$GPU_ARG" in
    both)                 GPU_LIST=(gpu cpu) ;;
    gpu|cpu)              GPU_LIST=("$GPU_ARG") ;;
    *)                    die "--gpu must be gpu, cpu, or both" ;;
esac

# ---------------------------------------------------------------------------------
# Host and target resolution
# ---------------------------------------------------------------------------------
HOST_OS="$(uname -s)"
HOST_TARGET="$(rustc -vV | sed -n 's/^host: //p')"
HOST_EXE=""
[[ "$HOST_TARGET" == *windows* ]] && HOST_EXE=".exe"

if [ "$OS_ARG" = auto ]; then
    case "$HOST_OS" in
        Linux)                OS_ARG=linux ;;
        MINGW*|MSYS*|CYGWIN*) OS_ARG=windows ;;
        *)                    die "unsupported host $HOST_OS; pass --os windows|linux explicitly" ;;
    esac
fi

case "$OS_ARG" in
    linux)   RUST_TARGET=x86_64-unknown-linux-gnu; EXE="" ;;
    windows) RUST_TARGET=x86_64-pc-windows-gnu;   EXE=".exe" ;;
    *)       die "--os must be windows or linux" ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "$OUT_DIR" ] || OUT_DIR="$ROOT/bin"

# llvm-profdata always runs on the HOST system
SYSROOT="$(rustc --print sysroot)"
PROFDATA="$SYSROOT/lib/rustlib/$HOST_TARGET/bin/llvm-profdata$HOST_EXE"
if [ ! -x "$PROFDATA" ]; then
    if command -v llvm-profdata >/dev/null 2>&1; then
        PROFDATA="$(command -v llvm-profdata)"
    else
        die "llvm-profdata not found in host rustlib ($PROFDATA) or system PATH"
    fi
fi

# BOLT is only supported on Linux ELF binaries
BOLT_ENABLED=0
if [ "$NO_BOLT" -eq 0 ] && [ "$OS_ARG" = linux ]; then
    if [ -n "$BOLT_SCENE" ] && command -v llvm-bolt >/dev/null 2>&1 && command -v merge-fdata >/dev/null 2>&1; then
        [ -f "$BOLT_SCENE" ] || die "--bolt-scene file not found: $BOLT_SCENE"
        BOLT_SCENE="$(cd "$(dirname "$BOLT_SCENE")" && pwd)/$(basename "$BOLT_SCENE")"
        BOLT_ENABLED=1
    fi
fi

step() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------------------------
# Combination worker
# ---------------------------------------------------------------------------------
build_combination() {
    local isa="$1" gpu="$2"
    local name="$isa-$gpu"
    
    # Isolate cargo builds from profiling data to prevent cargo/build.rs from deleting the profile
    local bdir="$ROOT/target/pgo-builds/$OS_ARG-$name"
    local pdir="$ROOT/target/pgo-profiles/$OS_ARG-$name"
    local rawdir="$pdir/raw"
    local merged="$pdir/merged.profdata"
    
    local target_cpu simd_cap train_features
    local -a app_features
    
    case "$isa" in
        avx512) target_cpu=x86-64-v4; simd_cap=avx512 ;;
        avx2)   target_cpu=x86-64-v3; simd_cap=avx2 ;;
        scalar) target_cpu=x86-64;    simd_cap=scalar ;;
    esac
    
    if [ "$gpu" = gpu ]; then
        train_features='hdr,serde,gpu'
        app_features=(--features 'diagram-gui/gpu,gemray-worker/gpu,gemray-worker/worker')
    else
        train_features='hdr,serde'
        app_features=(--features 'gemray-worker/worker')
    fi
    
    local cpu_flag="-C target-cpu=$target_cpu"
    local bolt_link_flags=""
    if [ "$BOLT_ENABLED" -eq 1 ]; then
        bolt_link_flags=" -C link-arg=-Wl,--emit-relocs -C force-frame-pointers=yes"
    fi
    
    # 1. Native host profiling stage
    if [ "$SKIP_TRAIN" -eq 0 ] || [ ! -f "$merged" ]; then
        rm -rf "$pdir"
        mkdir -p "$rawdir"
        
        step "[$name][train] Building instrumented runner on host ($HOST_TARGET)"
        RUSTFLAGS="$cpu_flag -C profile-generate=$rawdir" \
        cargo build --release --target "$HOST_TARGET" --target-dir "$bdir" \
        -p gemray --features "$train_features" --example pgo_train
        
        step "[$name][train] Running workload (GEMRAY_SIMD=$simd_cap)"
        LLVM_PROFILE_FILE="$rawdir/train-%p-%m.profraw" GEMRAY_SIMD="$simd_cap" \
        "$bdir/$HOST_TARGET/release/examples/pgo_train$HOST_EXE"
        
        # Expand all profraw files safely via nullglob
        shopt -s nullglob
        local -a prof_files=("$rawdir"/*.profraw)
        shopt -u nullglob
        
        [ "${#prof_files[@]}" -gt 0 ] || die "[$name][train] No .profraw files generated in $rawdir"
        
        step "[$name][merge] Merging ${#prof_files[@]} profile(s) with $PROFDATA"
        "$PROFDATA" merge -o "$merged" "${prof_files[@]}"
        
        # Assert the file actually exists and is not empty before letting Cargo loose
        [ -s "$merged" ] || die "[$name][merge] llvm-profdata failed or output is empty: $merged"
    else
        step "[$name][train] Reusing existing profile: $merged"
    fi
    
    # 2. Optimized release compilation for the target
    step "[$name][build] Building PGO-optimized binaries for $RUST_TARGET"
    RUSTFLAGS="$cpu_flag -C profile-use=$merged$bolt_link_flags" \
    cargo build --release --target "$RUST_TARGET" --target-dir "$bdir" \
    -p diagram-gui -p gemray-worker "${app_features[@]}"
    
    local release_dir="$bdir/$RUST_TARGET/release"
    local bin src dst
    for bin in diagram-gui gemray-worker; do
        src="$release_dir/$bin$EXE"
        [ -f "$src" ] || die "[$name][build] Expected binary missing: $src"
        dst="$OUT_DIR/$bin-$OS_ARG-$name$EXE"
        cp -f "$src" "$dst"
        printf '    %s (%s)\n' "$dst" "$(du -h "$dst" | cut -f1)"
    done
    
    # 3. Optional BOLT post-processing (Linux only)
    if [ "$BOLT_ENABLED" -eq 1 ]; then
        bolt_worker "$name" "$release_dir" "$bdir"
    fi
}

bolt_worker() {
    local name="$1" release_dir="$2" bdir="$3"
    local src="$release_dir/gemray-worker"
    local work="$bdir/bolt"
    mkdir -p "$work"
    
    step "[$name][bolt] Instrumenting gemray-worker"
    llvm-bolt "$src" -instrument --instrumentation-file="$work/prof.fdata" \
    --instrumentation-file-append-pid -o "$work/gemray-worker.inst"
    
    step "[$name][bolt] Collecting profile ($BOLT_SAMPLES spp)"
    "$work/gemray-worker.inst" render \
    --scene "$BOLT_SCENE" --out "$work/train.png" \
    --width 640 --height 360 --samples "$BOLT_SAMPLES" --no-gpu
    
    merge-fdata "$work"/prof.fdata* > "$work/merged.fdata"
    
    step "[$name][bolt] Rewriting binary"
    llvm-bolt "$src" -data="$work/merged.fdata" -o "$work/gemray-worker.bolt" \
    -reorder-blocks=ext-tsp -reorder-functions=hfsort+ \
    -split-functions -split-all-cold -icf=1 -dyno-stats
    
    cp -f "$work/gemray-worker.bolt" "$OUT_DIR/gemray-worker-$OS_ARG-$name"
    printf '    %s (BOLT)\n' "$OUT_DIR/gemray-worker-$OS_ARG-$name"
}

# ---------------------------------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------------------------------
printf 'Host System:  %s (%s)\n' "$HOST_OS" "$HOST_TARGET"
printf 'Target Build: %s (%s)\n' "$OS_ARG" "$RUST_TARGET"
printf 'ISA Tiers:    %s\n' "${ISA_LIST[*]}"
printf 'GPU Tiers:    %s\n' "${GPU_LIST[*]}"
printf 'Execution:    %s\n' "$([ "$PARALLEL" -eq 1 ] && echo 'parallel' || echo 'sequential')"
printf 'BOLT:         %s\n' "$([ "$BOLT_ENABLED" -eq 1 ] && echo 'enabled (gemray-worker)' || echo 'disabled')"
printf 'Output Dir:   %s\n' "$OUT_DIR"

mkdir -p "$OUT_DIR"

if [ "$PARALLEL" -eq 1 ]; then
    printf '\n==> Prefetching crate dependencies for target and host...\n'
    cargo fetch --target "$RUST_TARGET"
    [ "$HOST_TARGET" != "$RUST_TARGET" ] && cargo fetch --target "$HOST_TARGET"
    
    declare -A JOB_PIDS
    declare -A JOB_LOGS
    
    for isa in "${ISA_LIST[@]}"; do
        for gpu in "${GPU_LIST[@]}"; do
            name="$isa-$gpu"
            
            # The log file lives safely in the build dir root
            bdir="$ROOT/target/pgo-builds/$OS_ARG-$name"
            mkdir -p "$bdir"
            logfile="$bdir/build.log"
            
            printf '==> [%s] Dispatched build in background (log: %s)\n' "$name" "$logfile"
            
            (
                build_combination "$isa" "$gpu"
            ) > "$logfile" 2>&1 &
            
            pid=$!
            JOB_PIDS["$pid"]="$name"
            JOB_LOGS["$pid"]="$logfile"
        done
    done
    
    printf '\n==> Waiting for %d background builds to complete...\n' "${#JOB_PIDS[@]}"
    
    FAILED=0
    for pid in "${!JOB_PIDS[@]}"; do
        name="${JOB_PIDS[$pid]}"
        logfile="${JOB_LOGS[$pid]}"
        
        if wait "$pid"; then
            printf '  [SUCCESS] %s\n' "$name"
        else
            printf '  [FAILED]  %s (see tail of %s)\n' "$name" "$logfile" >&2
            tail -n 35 "$logfile" >&2 || true
            FAILED=1
        fi
    done
    
    [ "$FAILED" -eq 0 ] || die "One or more parallel builds failed."
else
    for isa in "${ISA_LIST[@]}"; do
        for gpu in "${GPU_LIST[@]}"; do
            build_combination "$isa" "$gpu"
        done
    done
fi

printf '\nAll requested artifacts generated in %s:\n' "$OUT_DIR"
ls -lh "$OUT_DIR"/*"$OS_ARG"*