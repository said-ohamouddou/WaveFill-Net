#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train 5-seed PointTransformer-family classifier ensembles for each backbone
# in the BACKBONES list. These checkpoints are the prerequisite for main.py
# (which loads them frozen, both for the L_cls term during WaveFill training
# and for the paired-eval classifier at test time).
#
# Output convention (matches what main.py expects to find on disk):
#   runs/all_16d_backbones_1024pts_5seed/<BACKBONE>/classifier/
#       history_seed{0..4}.json
#       model_seed{0..4}.pt
#
# Per-backbone failures (e.g. missing CUDA extension for a given backbone)
# do NOT stop the loop -- they are logged and the script continues.
#
# Usage
# -----
#   bash train_classifiers.sh                              # full sweep
#   DRY_RUN=1 bash train_classifiers.sh                    # print commands only
#   BACKBONES_OVERRIDE="DGCNN PointNet" bash train_classifiers.sh   # subset
#   DATASET=HeliALS_voxelagg_8192_16D NUM_POINTS=8192 bash train_classifiers.sh
#   FORCE=1 bash train_classifiers.sh                      # retrain even if present
# ---------------------------------------------------------------------------

set -u   # error on undefined; do NOT set -e: we want to continue on per-backbone failures.

# ----- Conda env (optional) ---------------------------------------------------
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "torch311" ]; then
    if [ -f ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        # conda's activation hooks reference unset vars; relax `set -u`
        # around them, then restore it.
        set +u
        source ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh
        conda activate torch311
        set -u
    fi
fi

# ----- Config (override via env) ----------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="${DATASET:-HeliALS_voxelagg_1024_16D}"
NUM_POINTS="${NUM_POINTS:-1024}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
NUM_MODELS="${NUM_MODELS:-5}"               # ensemble size (paper convention)
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/all_16d_backbones_1024pts_5seed}"

BACKBONES=(
    "PointTransformerV2"
    "DGCNN"
    "DeepGCN"
    "PCT"
    "PointNet"
)

# Honour BACKBONES_OVERRIDE env var (whitespace-separated) if set.
if [ -n "${BACKBONES_OVERRIDE:-}" ]; then
    # shellcheck disable=SC2206
    BACKBONES=( ${BACKBONES_OVERRIDE} )
fi

DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${OUTPUT_ROOT}/_logs"
mkdir -p "$LOG_DIR"

STATUS_CSV="${LOG_DIR}/classifier_status.csv"
echo "backbone,status,wallclock_s,log" > "$STATUS_CSV"

# Backbones whose CUDA extension (pointnet2_ops) refuses fp16 — skip --use-amp.
NO_AMP_BACKBONES="PointNet2_MSG PCT PointMLP GDAN"

run_one() {
    local backbone="$1"
    local out="${OUTPUT_ROOT}/${backbone}/classifier"
    local logfile="${LOG_DIR}/${backbone}_classifier.log"

    # Completeness = EVERY seed present, not just the last one. Testing only
    # model_seed$((NUM_MODELS-1)).pt let a partial ensemble (e.g. seeds 0,1
    # missing but 4 written by an earlier larger run) count as done. And FORCE
    # must override the skip, which it previously could not.
    local missing=0 sd
    for (( sd=0; sd<NUM_MODELS; sd++ )); do
        [ -f "${out}/model_seed${sd}.pt" ] || missing=$((missing + 1))
    done
    if [ "${FORCE:-0}" = "1" ]; then
        echo ">> [force] retraining ${backbone} (FORCE=1)"
    elif [ "$missing" -eq 0 ]; then
        echo ">> [skip] complete ${NUM_MODELS}-seed ensemble already at ${out}"
        echo "${backbone},skip,0,${logfile}" >> "$STATUS_CSV"
        return 0
    elif [ "$missing" -lt "$NUM_MODELS" ]; then
        echo ">> [incomplete] ${backbone}: ${missing}/${NUM_MODELS} seed(s) "\
             "missing -- retraining the ensemble"
    fi

    local amp_flag="--use-amp"
    if [[ " ${NO_AMP_BACKBONES} " == *" ${backbone} "* ]]; then
        amp_flag=""
    fi

    echo
    echo "================================================================"
    echo ">> Train ${NUM_MODELS}-seed classifier ensemble for ${backbone}"
    echo "   output -> ${out}"
    echo "   log    -> ${logfile}"
    echo "   amp    -> ${amp_flag:-off}"
    echo "================================================================"

    if [ "$DRY_RUN" = "1" ]; then
        echo "   [dry-run] would run:"
        echo "   python train_classifier_ensemble.py \\"
        echo "       --dataset \"${DATASET}\" --num-points \"${NUM_POINTS}\" \\"
        echo "       --backbone \"${backbone}\" \\"
        echo "       --output-dir \"${out}\" \\"
        echo "       --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} \\"
        echo "       --num-models ${NUM_MODELS} ${amp_flag}"
        return 0
    fi

    local t0; t0=$(date +%s)
    python -u train_classifier_ensemble.py \
        --dataset "${DATASET}" --num-points "${NUM_POINTS}" \
        --backbone "${backbone}" \
        --output-dir "${out}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LR}" \
        --num-models "${NUM_MODELS}" ${amp_flag} \
        2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    local t1; t1=$(date +%s)
    local dt=$((t1 - t0))

    if [ $rc -ne 0 ]; then
        echo "   FAILED (rc=$rc) after ${dt}s -- see ${logfile}"
        echo "${backbone},FAILED(rc=${rc}),${dt},${logfile}" >> "$STATUS_CSV"
        return 1
    fi
    echo "   OK in ${dt}s"
    echo "${backbone},ok,${dt},${logfile}" >> "$STATUS_CSV"
    return 0
}

# ----- Banner -----------------------------------------------------------------
echo "################################################################"
echo "# Classifier-ensemble pipeline"
echo "################################################################"
echo "  Dataset       : ${DATASET}"
echo "  Num points    : ${NUM_POINTS}"
echo "  Ensemble size : ${NUM_MODELS}"
echo "  Epochs        : ${EPOCHS}"
echo "  Batch size    : ${BATCH_SIZE}"
echo "  LR            : ${LR}"
echo "  Output root   : ${OUTPUT_ROOT}"
echo "  Backbones     : ${BACKBONES[*]}"
echo "  Dry-run       : ${DRY_RUN}"
echo "################################################################"

# ----- Loop -------------------------------------------------------------------
# Per-backbone failures must not stop the sweep, but they MUST reach the exit
# code -- otherwise a caller (run_main_experiment.sh) sees success while some
# backbone has no classifier at all.
FAILED=()
for BB in "${BACKBONES[@]}"; do
    run_one "$BB" || FAILED+=("$BB")
done

echo
echo "################################################################"
echo "# Done."
echo "# Status:  ${STATUS_CSV}"
echo "# Next:    python main.py --backbone <BB>  for each backbone you trained."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "# FAILED:  ${FAILED[*]}"
    echo "################################################################"
    exit 1
fi
echo "################################################################"
