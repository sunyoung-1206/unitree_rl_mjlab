#!/usr/bin/env bash
# Re-run the 14-condition demag matrix WITH video (mp4) using the already-trained
# Schur policy.  Uses the same checkpoint resolved by the original pipeline.
#
# Output: results/demag_rerun_schur_trained/{data,videos,logs}/

set -u
set -o pipefail

cd "$(git rev-parse --show-toplevel)"

PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
EXP_NAME="go2_methoda_electric"
PD_CKPT="logs/rsl_rl/pd_policy20ms_physics5ms/2026-04-17_00-13-37_seed42/model_1999.pt"
OUT_ROOT="results/demag_rerun_schur_trained"
DEMAG_SCRIPT="${OUT_ROOT}/run_demag_experiment.py"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "$LOG_DIR" "${OUT_ROOT}/videos"

PIPELINE_LOG="${LOG_DIR}/_video_pipeline.log"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$PIPELINE_LOG"; }

# Resolve the latest (post-killed) MethodA training checkpoint
LOG_ROOT="logs/rsl_rl/${EXP_NAME}"
LATEST_RUN="$(ls -1d "$LOG_ROOT"/*/ 2>/dev/null \
    | grep -v 0000-killed \
    | sort | tail -1)"
LATEST_RUN="${LATEST_RUN%/}"
CKPT="$(ls -1 "$LATEST_RUN"/model_*.pt 2>/dev/null \
        | awk -F'model_' '{print $2"\t"$0}' \
        | sort -n | tail -1 | cut -f2)"
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    log "[FATAL] no checkpoint found"
    exit 1
fi

log "=== Video pipeline start ==="
log "checkpoint: $CKPT"
log "out: $OUT_ROOT  (videos enabled)"

run_demag() {
    local policy=$1 leg=$2 factor=$3 tag=$4
    local extra_ckpt_arg=()
    if [ "$policy" = "pd" ]; then
        extra_ckpt_arg=(--pd-checkpoint "$PD_CKPT")
    else
        extra_ckpt_arg=(--methoda-checkpoint "$CKPT")
    fi
    log "    [START] $tag"
    "$PY" "$DEMAG_SCRIPT" \
        --policy "$policy" --leg "$leg" --demag-factor "$factor" \
        --output-dir "$OUT_ROOT" \
        "${extra_ckpt_arg[@]}" \
        > "${LOG_DIR}/${tag}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "    [FAIL] $tag (rc=$rc)"
        return $rc
    fi
    log "    [OK]    $tag"
}

run_demag pd      none 1.0 "pd_nominal"     || true
run_demag methoda none 1.0 "methoda_healthy" || true
for leg in FL FR RL RR; do
    for f in 0.8 0.6 0.4; do
        run_demag methoda "$leg" "$f" "methoda_${leg}_${f}" || true
    done
done

log "All cases done. Re-encoding videos to H.264 baseline (player compat)..."
"$PY" "${OUT_ROOT}/reencode_videos.py" --backup \
    >> "${LOG_DIR}/_reencode.log" 2>&1 \
    && log "    [OK]    reencode" \
    || log "    [FAIL]  reencode"

log "=== Video pipeline done ==="
log "Videos: $OUT_ROOT/videos/   Data: $OUT_ROOT/data/"
