#!/usr/bin/env bash
# Ki=0 sweep with heading-lock (Phase 2 재실행).
# 기존 sweep 결과와 비교하기 위해 별도 OUT_DIR 에 저장.
# 7 cases (healthy + RR factor ∈ {0.95,0.90,0.85,0.80,0.75,0.70}), 1000 step.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(realpath "$0")")")"
PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
SCRIPT=results/heading_lock_compare/run_demag_experiment.py
OUT_DIR=results/heading_lock_compare/aplus_tloop_hl_on
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

# Sequential (병렬 시 mjwarp init race 로 멈춤 — 다중 GPU 점유 회피).
SPECS=("none:1.0" "RR:0.95" "RR:0.90" "RR:0.85" "RR:0.80" "RR:0.75" "RR:0.70")
i=0
for spec in "${SPECS[@]}"; do
  i=$((i+1))
  leg="${spec%:*}"
  factor="${spec#*:}"
  echo "[$i/${#SPECS[@]}] start: $(date +%H:%M:%S)"
  run "$leg" "$factor"
  wait
  echo "[$i/${#SPECS[@]}] done:  $(date +%H:%M:%S)"
done
echo "=== Ki=0 RR sweep (force-init + heading-lock) done ==="
