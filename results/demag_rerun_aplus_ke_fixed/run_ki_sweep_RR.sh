#!/usr/bin/env bash
# Ki sweep: aplus_tloop 정책 + RR_calf demag 매트릭스, Ki 4 단계 비교.
# 각 Ki 값별 별도 디렉터리 (aplus_tloop_ki025/ ki050/ ki100/ ki200/).
# 측정 항목 (notebook 분석):
#   - settling time (1차계 fit τ)
#   - overshoot (peak / tail mean)
#   - cap usage % (peak |I_int| / Ki·integral_max)
#   - posture tracking (|q_des - q| RMS)
# 10 s rollout (500 step @ 20ms): transient + 5s 정상상태 둘 다 확보, no-video.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(realpath "$0")")")"
PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
SCRIPT=results/demag_rerun_aplus_ke_fixed/run_demag_experiment.py
OUT_DIR=results/demag_rerun_aplus_ke_fixed
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

NUM_STEPS=500   # 10 s
KI_VALUES=(25 50 100 200)
# (leg, factor) 페어 — α=1.0 healthy + α=0.4 moderate + α=0.2 deep
SPECS=("none:1.0" "RR:0.4" "RR:0.2")

run() {
  local ki=$1 leg=$2 factor=$3 tag
  local ki_str
  ki_str=$(printf "%03d" "$ki")
  if [ "$leg" = "none" ]; then tag="aplus_tloop_ki${ki_str}_healthy"
  else tag="aplus_tloop_ki${ki_str}_${leg}_${factor}"
  fi
  echo "[START] $tag"
  "$PY" "$SCRIPT" \
    --policy aplus_tloop --leg "$leg" --demag-factor "$factor" \
    --ki-override "$ki" --num-steps "$NUM_STEPS" --no-video \
    --output-dir "$OUT_DIR" \
    > "$LOG_DIR/${tag}.log" 2>&1 &
}

# Batch by Ki — 3 jobs per batch (one per α), wait between batches.
for ki in "${KI_VALUES[@]}"; do
  for spec in "${SPECS[@]}"; do
    leg="${spec%:*}"
    factor="${spec#*:}"
    run "$ki" "$leg" "$factor"
  done
  wait
  echo "[BATCH DONE] Ki=$ki"
done
echo "=== Ki sweep done ==="
