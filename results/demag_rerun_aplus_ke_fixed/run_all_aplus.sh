#!/usr/bin/env bash
# Batch driver for A+ (Method A+) demag rerun, ke_fixed semantics.
# Policy: aplus (Unitree-Go2-Flat-Coupled-Electric, dynprm[4]=1, β = exp(-h/τ)).
# Engine state: GPU and CPU both have (dynprm[3]-dynprm[1])*ω/L Ke-correction
#               active → ke_fixed semantics by default (no flag needed).
# vx=0.5 default (matching ke_fixed for 1:1 comparison).
# Runs 14 conditions in chunks of 4 to avoid GPU OOM.

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(realpath "$0")")")"
PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
SCRIPT=results/demag_rerun_aplus_ke_fixed/run_demag_experiment.py
OUT_DIR=results/demag_rerun_aplus_ke_fixed
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

run() {
  local policy=$1 leg=$2 factor=$3 tag
  if [ "$policy" = "pd" ]; then tag="pd_nominal"
  elif [ "$leg" = "none" ]; then tag="${policy}_healthy"
  else tag="${policy}_${leg}_${factor}"
  fi
  echo "[START] $tag"
  "$PY" "$SCRIPT" \
    --policy "$policy" --leg "$leg" --demag-factor "$factor" --no-video \
    --output-dir "$OUT_DIR" \
    > "$LOG_DIR/${tag}.log" 2>&1 &
}

# Batch 1 (4)
run pd    none 1.0
run aplus none 1.0
run aplus FL   0.8
run aplus FL   0.6
wait
echo "=== batch 1 done ==="

# Batch 2 (4)
run aplus FL 0.4
run aplus FR 0.8
run aplus FR 0.6
run aplus FR 0.4
wait
echo "=== batch 2 done ==="

# Batch 3 (4)
run aplus RL 0.8
run aplus RL 0.6
run aplus RL 0.4
run aplus RR 0.8
wait
echo "=== batch 3 done ==="

# Batch 4 (2)
run aplus RR 0.6
run aplus RR 0.4
wait
echo "=== batch 4 done ==="

echo "All 14 runs finished."
