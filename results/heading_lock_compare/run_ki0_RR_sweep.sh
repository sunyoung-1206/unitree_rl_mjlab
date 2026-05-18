#!/usr/bin/env bash
# Ki=0 sweep: aplus_tloop policy + RR_calf demag matrix — HL off (fair).
# Integral gain disabled (--ki-override 0) → torque-loop is P-only.
# Initial state: apply_zero_initial_state() in run_demag_experiment.py is
#   unconditional → pos=(0,0,h), quat=(1,0,0,0), vel=0 (HL on 과 동일).
# Heading-control: --no-heading-control → wz=0 manual (yaw 자유).
# 위 두 조건으로 aplus_tloop_hl_on/ 과 zero-init 만 공유하고
# heading-lock 플래그만 다른 fair-comparison 데이터 생성.
# Cases (7): healthy + RR_calf factor ∈ {0.95, 0.90, 0.85, 0.80, 0.75, 0.70}
#            i.e. demag levels 0% / 5% / 10% / 15% / 20% / 25% / 30%.
# 20 s rollout (1000 step @ 20ms), with video.
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
    --no-heading-control \
    --output-dir "$OUT_DIR" \
    > "$LOG_DIR/${tag}.log" 2>&1 &
}

# Sequential (병렬 시 mjwarp init race 로 멈춤 — HL on sweep 과 동일 패턴).
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
echo "=== Ki=0 RR sweep (HL off, zero-init) done ==="
