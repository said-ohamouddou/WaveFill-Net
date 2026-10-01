#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Main imputation experiment, end to end.
#
#   Stage 1  train_classifiers.sh            frozen classifier ensembles
#                                            (5 seeds x 200 epochs per backbone)
#   Stage 2  run_imputation_experiments.py   every baseline + BOTH WaveFill-Net
#                                            variants, 5 seeds x 100 epochs
#   Stage 3  ablation_components_kan.py      Experiment 2: component ablation
#                                            (FastKAN, cross-classifier, 5 seeds)
#   Stage 4  ablation_hparam_kan.py          Experiment 3: hyperparameter sweep
#                                            (FastKAN, 4 params x 7 values, 1 seed)
#   Stage 5  build_master_table.py           consolidated LaTeX + console tables
#
# Stage 1 is the prerequisite: main.py loads those classifiers frozen, both as
# the teacher for L_cls and as the seed-matched paired evaluator at test time.
# Stage 2 trains the MLP and the ReLU-KAN interleaved per seed, so every table
# it prints carries both variants.
#
# The two WaveFill variants are PARAMETER-MATCHED by construction: MLP h=256
# (139,828 params) vs ReLU-KAN h=88 (150,700 params), a 1.08x ratio. Do not
# override --hidden unless you mean to break that matching.
#
# Resumable: anything already on disk is skipped, so re-running after an
# interruption continues where it stopped. Pass FORCE=1 to retrain everything.
#
# Usage
# -----
#   bash run_main_experiment.sh                       # full run
#   DRY_RUN=1 bash run_main_experiment.sh             # print commands only
#   SKIP_CLASSIFIERS=1 bash run_main_experiment.sh    # classifiers already done
#   BACKBONES="PointNet DGCNN" bash run_main_experiment.sh
#   SEEDS=0 EPOCHS_IMP=25 bash run_main_experiment.sh # quick smoke pass
#   RUN_ABLATIONS=0 bash run_main_experiment.sh       # main experiment only
#   SKIP_CLASSIFIERS=1 RUN_ABLATIONS=1 bash run_main_experiment.sh  # ablations only
#   FORCE=1 bash run_main_experiment.sh               # retrain from scratch
# ---------------------------------------------------------------------------

set -u   # do NOT set -e: a failing backbone is logged, the rest still runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----- Conda -----------------------------------------------------------------
# pointops / pointnet2_ops live in torch311; without it PointTransformerV2,
# PCT and PointNet2_MSG fail to build. conda's activation hooks reference
# unset vars, so `set -u` is relaxed around them.
if [ "${CONDA_DEFAULT_ENV:-}" != "torch311" ]; then
    if [ -f ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh ]; then
        set +u
        # shellcheck disable=SC1091
        source ${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh
        conda activate torch311
        set -u
    fi
fi

# ----- Config (override via env) ---------------------------------------------
EPOCHS_CLS="${EPOCHS_CLS:-200}"     # classifier ensembles (early stop, patience 30)
EPOCHS_IMP="${EPOCHS_IMP:-100}"     # WaveFill-Net imputers
SEEDS="${SEEDS:-0,1,2,3,4}"
NUM_MODELS="${NUM_MODELS:-5}"
IMPUTERS="${IMPUTERS:-mlp,relukan,fastkan}"
BACKBONES="${BACKBONES:-PointTransformerV2 DGCNN DeepGCN PCT PointNet}"
CLASSIFIER_ROOT="${CLASSIFIER_ROOT:-runs/all_16d_backbones_1024pts_5seed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/method}"
SKIP_CLASSIFIERS="${SKIP_CLASSIFIERS:-0}"
SKIP_TABLE="${SKIP_TABLE:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# --- Ablations (stages 3 and 4) ----------------------------------------------
# Both need the stage-1 classifiers. They are independent of stage 2, so a
# failure in one does not affect the others.
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"      # 0 = main experiment only
# Teacher for exp 2 and 3. Defaults to PointTransformerV2, but falls back to
# the first entry of BACKBONES when PTv2 is not among the trained backbones --
# otherwise a reduced run (e.g. BACKBONES="PointNet") looks for a classifier
# that was never trained and both ablations abort.
ABL_TEACHER="${ABL_TEACHER:-}"
if [ -z "${ABL_TEACHER}" ]; then
    case " ${BACKBONES} " in
        *" PointTransformerV2 "*) ABL_TEACHER="PointTransformerV2" ;;
        *) ABL_TEACHER="${BACKBONES%% *}" ;;
    esac
fi
ABL_SEEDS_COMP="${ABL_SEEDS_COMP:-0,1,2,3,4}"      # exp 2: multi-seed
ABL_SEEDS_HP="${ABL_SEEDS_HP:-0}"                  # exp 3: single seed (25 configs)
# FastKAN hyperparameters: grid/k were ReLU-KAN's and no longer exist.
ABL_PARAMS="${ABL_PARAMS:-lambda_recon,lambda_cls,num_grids,hidden}"
# Keep ablation outputs beside the main results, so a scratch OUTPUT_ROOT
# keeps a smoke run fully self-contained.
_ABL_BASE="$(dirname "${OUTPUT_ROOT}")"
[ "${_ABL_BASE}" = "." ] && _ABL_BASE="runs"
ABL_OUT_COMP="${ABL_OUT_COMP:-${_ABL_BASE}/ablation_components_kan}"
ABL_OUT_HP="${ABL_OUT_HP:-${_ABL_BASE}/ablation_hparam_kan}"

RC=0
banner() { echo; echo "################################################################"
           echo "# $*"; echo "################################################################"; }

# ----- Guard against a second concurrent run ---------------------------------
# Two instances writing the same runs/ tree silently corrupt each other's
# checkpoints -- this has bitten this project before.
RUNNING=$(pgrep -f "run_imputation_experiments.py" 2>/dev/null | wc -l)
if [ "${RUNNING:-0}" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "ERROR: run_imputation_experiments.py is already running (${RUNNING} process(es))."
    echo "       Two runs writing the same output tree corrupt each other."
    echo "       Stop it first:  pkill -f run_imputation_experiments.py"
    exit 1
fi

banner "Main imputation experiment"
echo "  Backbones        : ${BACKBONES}"
echo "  Seeds            : ${SEEDS}"
echo "  Imputers         : ${IMPUTERS}  (param-matched: MLP h=256, ReLU-KAN h=88, FastKAN h=84)"
echo "  Classifier epochs: ${EPOCHS_CLS}  (ensemble of ${NUM_MODELS}, early stop)"
echo "  Imputer epochs   : ${EPOCHS_IMP}"
echo "  Classifier root  : ${CLASSIFIER_ROOT}"
echo "  Output root      : ${OUTPUT_ROOT}"
echo "  Ablations        : ${RUN_ABLATIONS}  (teacher ${ABL_TEACHER}, "\
     "exp2 seeds ${ABL_SEEDS_COMP}, exp3 seeds ${ABL_SEEDS_HP})"
echo "  Force retrain    : ${FORCE}"
echo "  Dry run          : ${DRY_RUN}"

# ----- Stage 1: classifier ensembles -----------------------------------------
if [ "$SKIP_CLASSIFIERS" = "1" ]; then
    banner "Stage 1/5  classifiers -- SKIPPED (SKIP_CLASSIFIERS=1)"
else
    banner "Stage 1/5  train_classifiers.sh   (${EPOCHS_CLS} epochs)"
    BACKBONES_OVERRIDE="${BACKBONES}" EPOCHS="${EPOCHS_CLS}" \
        NUM_MODELS="${NUM_MODELS}" OUTPUT_ROOT="${CLASSIFIER_ROOT}" \
        FORCE="${FORCE}" DRY_RUN="${DRY_RUN}" bash train_classifiers.sh
    rc=$?
    [ $rc -ne 0 ] && { echo ">> Stage 1 rc=${rc} (per-backbone failures are logged)"; RC=$rc; }
fi

# ----- Stage 2: imputation experiment ----------------------------------------
banner "Stage 2/5  run_imputation_experiments.py   (${EPOCHS_IMP} epochs)"
IMP_ARGS=(--backbones "${BACKBONES// /,}"
          --seeds "${SEEDS}"
          --imputers "${IMPUTERS}"
          --epochs "${EPOCHS_IMP}"
          --classifier-root "${CLASSIFIER_ROOT}"
          --output-root "${OUTPUT_ROOT}")
[ "$FORCE" = "1" ] && IMP_ARGS+=(--force-train)

if [ "$DRY_RUN" = "1" ]; then
    echo "   [dry-run] would run:"
    echo "   python run_imputation_experiments.py ${IMP_ARGS[*]}"
else
    python -u run_imputation_experiments.py "${IMP_ARGS[@]}"
    rc=$?
    [ $rc -ne 0 ] && { echo ">> Stage 2 FAILED rc=${rc}"; RC=$rc; }
fi

# ----- Stage 3: component ablation (Experiment 2) ----------------------------
if [ "$RUN_ABLATIONS" != "1" ]; then
    banner "Stages 3-4  ablations -- SKIPPED (RUN_ABLATIONS=0)"
else
    banner "Stage 3/5  ablation_components_kan.py   (Experiment 2, ${EPOCHS_IMP} epochs)"
    ABL_ARGS=(--teacher "${ABL_TEACHER}"
              --evaluators "${BACKBONES// /,}"
              --seeds "${ABL_SEEDS_COMP}"
              --epochs "${EPOCHS_IMP}"
              --classifier-root "${CLASSIFIER_ROOT}"
              --out-dir "${ABL_OUT_COMP}")
    [ "$FORCE" = "1" ] && ABL_ARGS+=(--force-train)
    if [ "$DRY_RUN" = "1" ]; then
        echo "   [dry-run] python ablation_components_kan.py ${ABL_ARGS[*]}"
    else
        python -u ablation_components_kan.py "${ABL_ARGS[@]}"
        rc=$?
        [ $rc -ne 0 ] && { echo ">> Stage 3 (exp 2) rc=${rc}"; RC=$rc; }
    fi

    # ----- Stage 4: hyperparameter ablation (Experiment 3) -------------------
    banner "Stage 4/5  ablation_hparam_kan.py   (Experiment 3, ${EPOCHS_IMP} epochs)"
    HP_ARGS=(--teacher "${ABL_TEACHER}"
             --params "${ABL_PARAMS}"
             --seeds "${ABL_SEEDS_HP}"
             --epochs "${EPOCHS_IMP}"
             --classifier-root "${CLASSIFIER_ROOT}"
             --out-dir "${ABL_OUT_HP}")
    [ "$FORCE" = "1" ] && HP_ARGS+=(--force-train)
    if [ "$DRY_RUN" = "1" ]; then
        echo "   [dry-run] python ablation_hparam_kan.py ${HP_ARGS[*]}"
    else
        python -u ablation_hparam_kan.py "${HP_ARGS[@]}"
        rc=$?
        [ $rc -ne 0 ] && { echo ">> Stage 4 (exp 3) rc=${rc}"; RC=$rc; }
    fi
fi

# ----- Stage 5: tables --------------------------------------------------------
if [ "$SKIP_TABLE" = "1" ]; then
    banner "Stage 5/5  tables -- SKIPPED (SKIP_TABLE=1)"
elif [ "$DRY_RUN" = "1" ]; then
    banner "Stage 5/5  build_master_table.py"
    echo "   [dry-run] would run: OUTPUT_ROOT=${OUTPUT_ROOT} python build_master_table.py"
else
    banner "Stage 5/5  build_master_table.py"
    OUTPUT_ROOT="${OUTPUT_ROOT}" python -u build_master_table.py
    rc=$?
    [ $rc -ne 0 ] && { echo ">> Stage 5 rc=${rc}"; RC=$rc; }
fi

# ----- Summary ----------------------------------------------------------------
banner "Done."
echo "  Per-backbone results : ${OUTPUT_ROOT}/<BB>/summary.json  (+ summary.csv)"
echo "  Per-seed raw         : ${OUTPUT_ROOT}/<BB>/seed<S>/eval_results.json"
echo "  Exp 2 (components)   : ${ABL_OUT_COMP}/results{,_aggregated}.csv"
echo "  Exp 3 (hyperparams)  : ${ABL_OUT_HP}/results{,_aggregated}.csv"
echo "  LaTeX table          : master_comparison.tex (+ .pdf)"
echo "  Classifier status    : ${CLASSIFIER_ROOT}/_logs/classifier_status.csv"
if [ $RC -ne 0 ]; then
    echo
    echo "  NOTE: a stage returned non-zero -- check the logs above."
fi
exit $RC
