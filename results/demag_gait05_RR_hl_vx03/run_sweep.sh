#!/usr/bin/env bash
# gait05 (2026-07-16 학습) 정책, RR calf demag 20/40/60%, heading-lock on, vx=0.3.
# 4 cases: healthy + RR factor in {0.8, 0.6, 0.4} (= 20/40/60% demag severity).
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(realpath "$0")")")"
PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
SCRIPT=results/demag_gait05_RR_hl_vx03/run_demag_experiment.py
OUT_DIR=results/demag_gait05_RR_hl_vx03
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

NUM_STEPS=1000   # 20 s @ 20ms policy dt

run() {
  local leg=$1 factor=$2 tag
  if [ "$leg" = "none" ]; then tag="gait05_healthy"
  else tag="gait05_${leg}_${factor}"
  fi
  echo "[START] $tag"
  "$PY" "$SCRIPT" \
    --policy gait05 --leg "$leg" --demag-factor "$factor" \
    --vx 0.3 --num-steps "$NUM_STEPS" \
    --output-dir "$OUT_DIR" \
    > "$LOG_DIR/${tag}.log" 2>&1
  echo "[DONE]  $tag (exit $?)"
}

SPECS=("none:1.0" "RR:0.8" "RR:0.6" "RR:0.4")
i=0
for spec in "${SPECS[@]}"; do
  i=$((i+1))
  leg="${spec%:*}"
  factor="${spec#*:}"
  echo "[$i/${#SPECS[@]}] start: $(date +%H:%M:%S)"
  run "$leg" "$factor"
  echo "[$i/${#SPECS[@]}] done:  $(date +%H:%M:%S)"
done
echo "=== gait05 RR demag sweep (heading-lock, vx=0.3) done ==="
