#!/usr/bin/env bash
# Controller-AWARE demag experiment (RR calf only).
#
# Contrast with demag_rerun_ke_fixed (controller unaware):
#   ke_fixed :  I_des = tau_des / Kt_nom            (controller unaware)
#   THIS     :  I_des = tau_des / (Kt_nom * factor) (controller compensates)
#
# Only the RR calf's controller _Ktgr is scaled; other joints keep Kt_nom.
# _Kegr (back-EMF FF) is left at nominal per the user's literal formula.
#
# 3 cases: healthy, RR×0.8, RR×0.6.  vx=0.5 default.

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(realpath "$0")")")"
PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
SCRIPT=results/demag_rerun_ke_ignored/run_demag_experiment.py
OUT_DIR=results/demag_rerun_ctrl_aware
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

run() {
  local leg=$1 factor=$2 tag
  if [ "$leg" = "none" ]; then
    tag="methoda_healthy"
    "$PY" "$SCRIPT" \
      --policy methoda --leg "$leg" --demag-factor "$factor" \
      --ctrl-aware \
      --output-dir "$OUT_DIR" \
      > "$LOG_DIR/${tag}.log" 2>&1 &
  else
    tag="methoda_${leg}_${factor}_ctrl_aware"
    "$PY" "$SCRIPT" \
      --policy methoda --leg "$leg" --demag-factor "$factor" \
      --ctrl-aware \
      --output-dir "$OUT_DIR" \
      > "$LOG_DIR/${tag}.log" 2>&1 &
  fi
  echo "[START] $tag"
}

# Batch (3 cases in parallel)
run none 1.0
run RR   0.8
run RR   0.6
wait
echo "=== All 3 runs finished (data npz + mp4) ==="
