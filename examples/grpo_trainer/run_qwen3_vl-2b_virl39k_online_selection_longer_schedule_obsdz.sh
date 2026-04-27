#!/usr/bin/env bash
# DISCOVER headline configuration on ViRL39K:
#   slow discovery cadence + observed dead-zone (tau_obs=0.80) + asymmetric utility.
#
# This is the canonical "DISCOVER + obs. dead-zone (default)" row of Table 1
# in the paper draft.
#
# Background (see docs/PAPER_STATUS_AND_PLAN_2026-04-26.md):
#   - The ViRL39K longer_schedule run (without obs-dead-zone) is the best on
#     record at 0.512 peak / 0.507 final, no entropy collapse.
#   - The ViRL39K vdr_1234 run (with obs-dead-zone but FAST cadence) collapsed
#     at 0.479 final despite the obs-dead-zone — confirming that obs-dead-zone
#     delays but does not prevent collapse under fast cadence.
#   - This script combines slow cadence with obs-dead-zone — the configuration
#     the paper actually claims as default. We need this measurement to
#     attribute the §3.6 contribution cleanly.
#
# Three possible outcomes:
#   (A) GAIN  vs slow-cadence-only (0.507 final): §3.6 ships as headline contribution.
#   (B) NEUTRAL: §3.6 ships as no-cost robustness mechanism (kills 17% text-leaky
#       tail at zero accuracy penalty).
#   (C) HURT: drop §3.6 from headline; report only as corpus-characterisation result.
#
# Required prerequisite:
#   cd /workspace/rl_data_selection/benchmark/rl_data_selection
#   bash dataset_prep/prepare.sh virl39k
#
# Usage:
#   bash run_qwen3_vl-2b_virl39k_online_selection_longer_schedule_obsdz.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DATASET_NAME="virl39k"
K_FINAL="${K_FINAL:-150}"
N_REPS="${N_REPS:-3}"
export WANDB_API_KEY='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'
TRAIN_SPLIT="${TRAIN_SPLIT:-train_90_100}"
TEST_SPLIT="${TEST_SPLIT:-test_10_100}"

DS_ROOT="${DATA_ROOT}/${DATASET_NAME}"
TRAIN_PARQUET="${DS_ROOT}/parquet/${TRAIN_SPLIT}.parquet"
VAL_PARQUET="${DS_ROOT}/parquet/${TEST_SPLIT}.parquet"
CLUSTER_ARRAYS="${DS_ROOT}/cluster_arrays_90_100/outputs_K${K_FINAL}_r${N_REPS}/cluster_arrays.npz"
DATASET_JSON="${DS_ROOT}/records/all.jsonl"

for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CLUSTER_ARRAYS" "$DATASET_JSON"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required file: $p" >&2
        echo "       Run: cd $(cd "$DS_ROOT/.." && pwd) && bash dataset_prep/prepare.sh $DATASET_NAME" >&2
        exit 1
    fi
done

# ----- Slow discovery cadence (same as the existing longer_schedule script) -----
# Total discovery budget exhausted at ~step 272 (vs step 176 on the original
# fast-cadence schedule). See docs/PAPER_STATUS_AND_PLAN_2026-04-26.md §2.1.
unset BUDGET_SCHEDULE
export BUDGET_SCHEDULE='[{until_budget_pct:50,per_round_pct:0.7,interval:5},{until_budget_pct:85,per_round_pct:0.25,interval:15},{until_budget_pct:100,per_round_pct:0.10,interval:25}]'

export SELECTION_BUDGET_PCT="${SELECTION_BUDGET_PCT:-0.33}"
export GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-11.11}"
export RESELECT_INTERVAL="${RESELECT_INTERVAL:-5}"

# ----- Predictor -----
export PREDICTOR_TYPE="${PREDICTOR_TYPE:-knn}"

# ----- Asymmetric utility (paper §3.5 default) -----
export ASYMMETRIC_UTILITY="${ASYMMETRIC_UTILITY:-true}"
export ASYMMETRIC_BIAS="${ASYMMETRIC_BIAS:-0.5}"
export ASYMMETRIC_DEAD_LOW="${ASYMMETRIC_DEAD_LOW:-0.05}"
export ASYMMETRIC_DEAD_HIGH="${ASYMMETRIC_DEAD_HIGH:-0.95}"

# ----- Observed dead-zone (paper §3.6, the new contribution) -----
# tau_obs = 0.80 is the headline default. The 2K text-only audit (§4.4)
# identified a 24.2% text-leaky fraction; the kill-rate under this threshold
# converges toward ~17–20% on the discovered pool, since the discovery
# mechanism preferentially picks ZPD samples (lower mean reward) than the
# unselected corpus.
OBS_DEAD_ZONE_HIGH="${OBS_DEAD_ZONE_HIGH:-0.80}"

# ----- Other knobs -----
export NORMALIZE_VARIANCE="${NORMALIZE_VARIANCE:-true}"
export COUNT_MEDOIDS_IN_BUDGET="${COUNT_MEDOIDS_IN_BUDGET:-false}"

if [[ "${1:-}" != --* && "${1:-}" != *=* && -n "${1:-}" ]]; then
    ENGINE="${1}"
    shift
else
    ENGINE="${ENGINE:-vllm}"
fi
VARIANT="${VARIANT:-interpolated_weighted}"

export WANDB_PROJECT_NAME_OVERRIDE="verl_grpo_virl39k_baseline"

exec bash "$SCRIPT_DIR/run_qwen3_vl-2b_online_selection.sh" \
    "$ENGINE" \
    "$CLUSTER_ARRAYS" \
    "$VARIANT" \
    "$DATASET_JSON" \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data_selection.cluster.n_clusters="${K_FINAL}" \
    data_selection.cluster.n_reps="${N_REPS}" \
    data_selection.cluster.observed_deadzone_high="${OBS_DEAD_ZONE_HIGH}" \
    trainer.project_name=verl_grpo_virl39k_baseline \
    trainer.experiment_name="${WANDB_EXPERIMENT_NAME:-virl39k_knn_k150_longer_schedule_obsdz080}" \
    +trainer.total_training_steps=300 \
    "$@"
