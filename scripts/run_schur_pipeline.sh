#!/usr/bin/env bash
# End-to-end pipeline:
#   1) train MethodA (Unitree-Go2-Flat-MethodA-Electric) for 2000 iters
#   2) locate the new checkpoint
#   3) run the 14-condition demag matrix into results/demag_rerun_schur_trained/
#
# Output: a single rolling log file the user can `tail -f` to watch progress.

set -u
set -o pipefail

cd "$(git rev-parse --show-toplevel)"

PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
TASK_ID="Unitree-Go2-Flat-MethodA-Electric"
EXP_NAME="go2_methoda_electric"
RUN_TAG="act-pos_pdt20ms_phyDt0p1ms_tauDec4"
MAX_ITERS=2000
PD_CKPT="logs/rsl_rl/pd_policy20ms_physics5ms/2026-04-17_00-13-37_seed42/model_1999.pt"
OUT_ROOT="results/demag_rerun_schur_trained"
DEMAG_SCRIPT="${OUT_ROOT}/run_demag_experiment.py"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "$LOG_DIR"

PIPELINE_LOG="${LOG_DIR}/_pipeline.log"
TRAIN_LOG="${LOG_DIR}/_training.log"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$PIPELINE_LOG"; }

log "=== Pipeline start ==="
log "Task: $TASK_ID  |  iters: $MAX_ITERS  |  out: $OUT_ROOT"

# ── 1) Train ─────────────────────────────────────────────────────────
log "[1/3] Launching training..."
LOG_ROOT="logs/rsl_rl/${EXP_NAME}"
mkdir -p "$LOG_ROOT"
PRE_RUNS="$(ls -1 "$LOG_ROOT" 2>/dev/null | wc -l)"
log "    existing runs in $LOG_ROOT: $PRE_RUNS"

"$PY" scripts/train.py "$TASK_ID" \
    --agent.max-iterations "$MAX_ITERS" \
    --agent.run-name "$RUN_TAG" \
    --env.scene.num-envs 4096 \
    > "$TRAIN_LOG" 2>&1
TRAIN_RC=$?
if [ $TRAIN_RC -ne 0 ]; then
    log "[FATAL] training exited with code $TRAIN_RC. See $TRAIN_LOG"
    exit 1
fi
log "[1/3] Training finished."

# ── 2) Resolve checkpoint ────────────────────────────────────────────
# The newest run dir under $LOG_ROOT containing model_*.pt that wasn't there
# pre-training is ours.  We pick the lexicographically latest run dir.
LATEST_RUN="$(ls -1d "$LOG_ROOT"/*/ 2>/dev/null | sort | tail -1)"
LATEST_RUN="${LATEST_RUN%/}"
if [ -z "$LATEST_RUN" ]; then
    log "[FATAL] no run dir found under $LOG_ROOT"
    exit 1
fi
log "    latest run dir: $LATEST_RUN"

# Pick the highest-numbered model_*.pt
CKPT="$(ls -1 "$LATEST_RUN"/model_*.pt 2>/dev/null \
        | awk -F'model_' '{print $2"\t"$0}' \
        | sort -n | tail -1 | cut -f2)"
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    log "[FATAL] no model_*.pt under $LATEST_RUN"
    exit 1
fi
log "[2/3] Selected checkpoint: $CKPT"

# ── 3) Demag matrix ──────────────────────────────────────────────────
# 14 conditions:
#   pd      × leg=none factor=1.0                            (uses PD_CKPT)
#   methoda × leg=none factor=1.0                            (healthy)
#   methoda × leg∈{FL,FR,RL,RR} factor∈{0.8, 0.6, 0.4}       (12 demag cases)
log "[3/3] Running 14-condition demag matrix..."

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
        --no-video \
        > "${LOG_DIR}/${tag}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "    [FAIL] $tag (rc=$rc)"
        return $rc
    fi
    log "    [OK]    $tag"
}

# Sequential — single GPU, avoid OOM
run_demag pd      none 1.0 "pd_nominal"     || true
run_demag methoda none 1.0 "methoda_healthy" || true
for leg in FL FR RL RR; do
    for f in 0.8 0.6 0.4; do
        run_demag methoda "$leg" "$f" "methoda_${leg}_${f}" || true
    done
done

log "=== Pipeline done ==="
log "Data: $OUT_ROOT/data/   Logs: $LOG_DIR/"
