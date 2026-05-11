#!/usr/bin/env bash
# q_freeze 격리 실험: aplus_tloop 정책 + RR_calf demag, action 을 t=0 값으로 freeze.
# outer loop forcing 시간 변화를 차단해 inner 적분기 단독 시상수 측정.
# 다리가 1~2 s 후 무너질 수 있어 5 s (250 step @ 20ms) 만 기록.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(realpath "$0")")")"
PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
SCRIPT=results/demag_rerun_aplus_ke_fixed/run_demag_experiment.py
OUT_DIR=results/demag_rerun_aplus_ke_fixed
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

NUM_STEPS=250  # 5 s — transient 만 깨끗하면 충분

run() {
  local leg=$1 factor=$2 tag
  if [ "$leg" = "none" ]; then tag="aplus_tloop_qfreeze_healthy"
  else tag="aplus_tloop_qfreeze_${leg}_${factor}"
  fi
  echo "[START] $tag"
  "$PY" "$SCRIPT" \
    --policy aplus_tloop --leg "$leg" --demag-factor "$factor" \
    --qfreeze --num-steps "$NUM_STEPS" \
    --output-dir "$OUT_DIR" \
    > "$LOG_DIR/${tag}.log" 2>&1 &
}

run none 1.0
run RR   0.8
run RR   0.6
run RR   0.4
wait
echo "=== aplus_tloop qfreeze RR matrix done ==="
