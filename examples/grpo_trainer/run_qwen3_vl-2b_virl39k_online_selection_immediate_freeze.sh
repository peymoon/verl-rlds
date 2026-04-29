#!/usr/bin/env bash
# Immediate-freeze ablation — isolates the contribution of the discovery phase.
#
# Instead of phased discovery across ~176 steps, this consumes the full 10%
# annotation budget in a single round at step 1, then runs frozen reweight
# for the remaining ~299 steps. Everything the model sees post-step-1 is
# multinomial-sampled from a fixed 10% pool, weighted by predicted variance.
#
# Hypothesis: if this run matches the full online-KNN run, the gain is
# driven by the frozen-reweight mechanism (+ the cluster-selected initial
# 10%), *not* by the incremental discovery. Discovery could then be removed
# from the method section or deferred to an appendix.
#
# Note: this is NOT the same as "random 10% + reweight" — the initial 10%
# pool here is still chosen by cluster-level variance, not uniformly at
# random. A truly random-pool-with-reweight baseline would require code
# changes to seed _frozen_pool_npz from a random subset at init.
#
# Required prerequisite:
#   cd /workspace/rl_data_selection
#   bash dataset_prep/prepare.sh virl39k
#
# Usage:
#   bash run_qwen3_vl-2b_virl39k_online_selection_immediate_freeze.sh
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
CLUSTER_ARRAYS="${CLUSTER_ARRAYS:-${DS_ROOT}/cluster_arrays_90_100/outputs_K${K_FINAL}_r${N_REPS}/cluster_arrays.npz}"
DATASET_JSON="${DATASET_JSON:-${DS_ROOT}/records/${TRAIN_SPLIT}.jsonl}"

for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CLUSTER_ARRAYS" "$DATASET_JSON"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required file: $p" >&2
        echo "       Run: cd $(cd "$DS_ROOT/.." && pwd) && bash dataset_prep/prepare.sh $DATASET_NAME" >&2
        exit 1
    fi
done

# Single-phase schedule: select the full ~11.11% budget in round 0, freeze,
# then reweight within that frozen pool.
#
# BUG FIX 1 (previous run overshot to 18.8%): interval:1 caused the phase to fire
# at step 0 AND step 1 before the global cap was enforced, selecting ~6,590
# samples instead of 3,498. Set interval to a value larger than total training
# steps so only round 0 fires.
#
# BUG FIX 2 (frozen reweight never fired in the old 10% run): per_round_pct
# landed below the global cap, so _selection_frozen never became True and
# should_reselect_step used interval=1000 (never fired in 300 steps).
# per_round_pct must bring total budget to >= global_max.
#
# per_round_pct = GLOBAL_BUDGET_PCT so round 0 training selection hits the cap:
# int(34983 x 11.11%) = 3,886 = global_max -> _selection_frozen=True.
# Medoids are measurement-only and excluded from the budget.
#
# The second phase is only for the post-cap frozen pool.  The selector's
# budget_schedule drives should_reselect_step(), so without this phase the
# interval:1000 cold-start phase would also suppress frozen reweight rounds.
unset BUDGET_SCHEDULE
export BUDGET_SCHEDULE='[{until_budget_pct:100,per_round_pct:11.11,interval:1000},{until_budget_pct:100000,per_round_pct:11.11,interval:16}]'

export SELECTION_BUDGET_PCT="${SELECTION_BUDGET_PCT:-11.11}"
export GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-11.11}"
# Keep the flat fallback aligned with the post-cap schedule phase.
export RESELECT_INTERVAL="${RESELECT_INTERVAL:-16}"

export PREDICTOR_TYPE="${PREDICTOR_TYPE:-knn}"

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
    trainer.project_name=verl_grpo_virl39k_baseline \
    trainer.experiment_name="${WANDB_EXPERIMENT_NAME:-virl39k_immediate_freeze_k150}" \
    +trainer.total_training_steps=300 \
    "$@"
