#!/usr/bin/env bash
# Run aplus_tloop policy on RR_calf demag matrix: healthy + 0.8/0.6/0.4.
# Reuses aplus checkpoint with use_torque_loop=True actuator.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(realpath "$0")")")"
PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
SCRIPT=results/demag_rerun_aplus_ke_fixed/run_demag_experiment.py
OUT_DIR=results/demag_rerun_aplus_ke_fixed
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

run() {
  local leg=$1 factor=$2 tag
  if [ "$leg" = "none" ]; then tag="aplus_tloop_healthy"
  else tag="aplus_tloop_${leg}_${factor}"
  fi
  echo "[START] $tag"
  "$PY" "$SCRIPT" \
    --policy aplus_tloop --leg "$leg" --demag-factor "$factor" \
    --output-dir "$OUT_DIR" \
    > "$LOG_DIR/${tag}.log" 2>&1 &
}

# Run all 4 RR cases in parallel (1 env each, fits on a single GPU).
run none 1.0
run RR   0.8
run RR   0.6
run RR   0.4
wait
echo "=== aplus_tloop RR matrix done ==="
