#!/usr/bin/env bash

set -euo pipefail

OUTDIR="add"
LOGFILE="${OUTDIR}/add.txt"

# Make sure output directory exists
mkdir -p "$OUTDIR"

# Clear previous log
: > "$LOGFILE"

run() {
  echo "========================================" | tee -a "$LOGFILE"
  echo "Running: $*" | tee -a "$LOGFILE"
  echo "Started at: $(date)" | tee -a "$LOGFILE"
  echo "----------------------------------------" | tee -a "$LOGFILE"

  "$@" >> "$LOGFILE" 2>&1

  echo "Finished at: $(date)" | tee -a "$LOGFILE"
  echo >> "$LOGFILE"
}
run nsys profile -o add/add1  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 1024 --hidden-dim 3072
run nsys profile -o add/add2  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 2048 --hidden-dim 3072

run nsys profile -o add/add3  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 3072
run nsys profile -o add/add4  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 8192 --hidden-dim 3072
run nsys profile -o add/add5  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 2048
run nsys profile -o add/add6  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 4096
run nsys profile -o add/add7  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 1  --tokens 4096 --hidden-dim 8192

run nsys profile -o add/add8  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 2048 --hidden-dim 3072
run nsys profile -o add/add9  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 4096 --hidden-dim 3072
run nsys profile -o add/add10  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 4  --tokens 8192 --hidden-dim 3072

run nsys profile -o add/add11  python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 4096 --hidden-dim 3072
run nsys profile -o add/add12 python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 8192 --hidden-dim 3072
run nsys profile -o add/add13 python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 64 --tokens 4096 --hidden-dim 3072
run nsys profile -o add/add14 python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 64 --tokens 8192 --hidden-dim 3072
run nsys profile -o add/add15 python scripts/profiling/profile_fused_add_rmsnorm_bandwidth.py --warmup 100 --iters 100 --batch-size 16 --tokens 4096 --hidden-dim 8192

echo "All runs completed successfully." | tee -a "$LOGFILE"
