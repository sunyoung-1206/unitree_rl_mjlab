#!/usr/bin/env bash
# Ki=0 sweep: aplus_tloop policy + RR_calf demag matrix.
# Integral gain disabled (--ki-override 0) → torque-loop is P-only.
# Cases (7): healthy + RR_calf factor ∈ {0.95, 0.90, 0.85, 0.80, 0.75, 0.70}
#            i.e. demag levels 0% / 5% / 10% / 15% / 20% / 25% / 30%.
# 20 s rollout (1000 step @ 20ms), with video.
# Reuses run_demag_experiment.py from demag_rerun_aplus_ke_fixed/.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(realpath "$0")")")"
PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
SCRIPT=results/heading_lock_compare/run_demag_experiment.py
OUT_DIR=results/heading_lock_compare/aplus_tloop_hl_off
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

NUM_STEPS=1000   # 20 s

run() {
  local leg=$1 factor=$2 tag
  if [ "$leg" = "none" ]; then tag="aplus_tloop_ki000_healthy"
  else tag="aplus_tloop_ki000_${leg}_${factor}"
  fi
  echo "[START] $tag"
  "$PY" "$SCRIPT" \
    --policy aplus_tloop --leg "$leg" --demag-factor "$factor" \
    --ki-override 0 --num-steps "$NUM_STEPS" \
    --output-dir "$OUT_DIR" \
    > "$LOG_DIR/${tag}.log" 2>&1 &
}

# Batch 1: 4 cases in parallel (1 env each).
run none 1.0
run RR   0.95
run RR   0.90
run RR   0.85
wait
echo "[BATCH 1 DONE]"

# Batch 2: 3 cases in parallel.
run RR   0.80
run RR   0.75
run RR   0.70
wait
echo "[BATCH 2 DONE]"

echo "=== Ki=0 RR sweep done ==="
