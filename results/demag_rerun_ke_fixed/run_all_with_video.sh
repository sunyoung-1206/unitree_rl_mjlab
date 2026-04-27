#!/usr/bin/env bash
# Phase 4 Ke-fixed batch WITH video.
# Overwrites data npz + generates mp4 from same rollout.
# vx=0.5 default, 14 cases.

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(realpath "$0")")")"
PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
SCRIPT=results/demag_rerun_ke_ignored/run_demag_experiment.py
OUT_DIR=results/demag_rerun_ke_fixed
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

run() {
  local policy=$1 leg=$2 factor=$3 tag
  if [ "$policy" = "pd" ]; then tag="pd_nominal"
  elif [ "$leg" = "none" ]; then tag="methoda_healthy"
  else tag="methoda_${leg}_${factor}"
  fi
  echo "[START] $tag"
  "$PY" "$SCRIPT" \
    --policy "$policy" --leg "$leg" --demag-factor "$factor" \
    --output-dir "$OUT_DIR" \
    > "$LOG_DIR/${tag}.log" 2>&1 &
}

# Batch 1 (4)
run pd      none 1.0
run methoda none 1.0
run methoda FL   0.8
run methoda FL   0.6
wait
echo "=== batch 1 done ==="

# Batch 2 (4)
run methoda FL 0.4
run methoda FR 0.8
run methoda FR 0.6
run methoda FR 0.4
wait
echo "=== batch 2 done ==="

# Batch 3 (4)
run methoda RL 0.8
run methoda RL 0.6
run methoda RL 0.4
run methoda RR 0.8
wait
echo "=== batch 3 done ==="

# Batch 4 (2)
run methoda RR 0.6
run methoda RR 0.4
wait
echo "=== batch 4 done ==="

echo "All 14 runs finished (data npz + mp4)."
