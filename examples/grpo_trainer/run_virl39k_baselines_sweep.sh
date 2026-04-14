#!/usr/bin/env bash
# ViRL39K de-risk gate: run every baseline back-to-back with seeds and matched
# compute, so we can compare "is selection doing anything?" without seed noise
# swamping the signal.
#
# Compute parity is enforced by setting `total_epochs` per method so that all
# runs see roughly the same number of optimizer steps:
#   train_90 (~35k samples) × 2 epochs   = 546 steps   (full)
#   3.5k samples            × 20 epochs  = 547 steps   (random10 / hard10)
#   ~3.5k cumulative unique × ~20 epochs = ~550 steps  (online)
#
# Methods (toggle via the METHODS env var, space-separated):
#   full         — full ViRL39K, 2 epochs
#   random10     — seeded random 10%, 20 epochs
#   hard10       — seeded random sample of base-failed (acc=0) samples, 20 ep
#   online_knn   — current best online selector at 10% global cap
#   online_mlp   — same with two-head MLP predictor
#
# Seeds (override via SEEDS env var). Methods with no seed dependence
# (just `full`) run once.
#
# Usage:
#   bash run_virl39k_baselines_sweep.sh
#   METHODS="random10 hard10" SEEDS="42 1337" bash run_virl39k_baselines_sweep.sh
#   DRY_RUN=1 bash run_virl39k_baselines_sweep.sh   # print plan, run nothing
#
# Logs land in $LOG_DIR (default /workspace/rl_data_selection/benchmark/rl_data_selection/logs/virl39k_sweep)

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # …/rl_data_selection

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/benchmark/rl_data_selection/data}"
DS_ROOT="${DATA_ROOT}/virl39k"
PARQUET_DIR="${DS_ROOT}/parquet"
RECORDS_DIR="${DS_ROOT}/records"

BASE_ROLLOUTS="${BASE_ROLLOUTS:-/workspace/peyman/outputs/rollouts/virl39k/virl39k_base_rollouts_train_90_100/0.jsonl}"
BASE_ROLLOUTS_N8="${BASE_ROLLOUTS_N8:-/workspace/rl_data_selection/peyman/outputs/virl39k/rollouts/virl39k/virl39k_base_rollouts_train_90_100_n8/0.jsonl}"

LOG_DIR="${LOG_DIR:-/workspace/rl_data_selection/benchmark/rl_data_selection/logs/virl39k_sweep}"
mkdir -p "$LOG_DIR"

# Methods + seeds (override via env)
METHODS="${METHODS:-full random10 hard10 online_knn online_mlp}"
SEEDS="${SEEDS:-42 1337}"
SUBSET_PCT="${SUBSET_PCT:-10}"

# Compute-parity epoch counts (see header).
EPOCHS_FULL="${EPOCHS_FULL:-2}"
EPOCHS_SUBSET="${EPOCHS_SUBSET:-20}"
EPOCHS_ONLINE="${EPOCHS_ONLINE:-20}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ENGINE="${ENGINE:-vllm}"

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_or_echo() {
    # $1 = log file, $@ = command
    local logf="$1"; shift
    echo "[$(ts)] >>> $*" | tee -a "$logf"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[$(ts)] DRY_RUN — skipping" | tee -a "$logf"
        return 0
    fi
    # shellcheck disable=SC2068
    "$@" 2>&1 | tee -a "$logf"
}

ensure_random_subset() {
    # ensure_random_subset <seed>
    # Generates train_random${SUBSET_PCT}_seed${seed}_100.parquet by calling
    # the existing make_random_subset (which writes to a fixed filename) and
    # then renaming both the parquet and the sidecar JSONL.
    local seed="$1"
    local sp="$SUBSET_PCT"
    local target_parquet="${PARQUET_DIR}/train_random${sp}_seed${seed}_100.parquet"
    local target_jsonl="${RECORDS_DIR}/train_random${sp}_seed${seed}_100.jsonl"
    if [ -e "$target_parquet" ] && [ -e "$target_jsonl" ]; then
        echo "[$(ts)] random subset seed=$seed already exists — skipping generation"
        return 0
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "[$(ts)] DRY_RUN — would generate random seed=$seed"
        return 0
    fi
    ( cd "$REPO_ROOT" && python -m dataset_prep.make_random_subset \
        --dataset virl39k --subset_pct "$sp" --seed "$seed" )
    mv "${PARQUET_DIR}/train_random${sp}_100.parquet" "$target_parquet"
    mv "${RECORDS_DIR}/train_random${sp}_100.jsonl"   "$target_jsonl"
    echo "[$(ts)] wrote $target_parquet"
}

ensure_hard_subset() {
    local seed="$1"
    local sp="$SUBSET_PCT"
    local target_parquet="${PARQUET_DIR}/train_hard${sp}_seed${seed}_100.parquet"
    if [ -e "$target_parquet" ]; then
        echo "[$(ts)] hard subset seed=$seed already exists — skipping generation"
        return 0
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "[$(ts)] DRY_RUN — would generate hard seed=$seed"
        return 0
    fi
    ( cd "$REPO_ROOT" && python -m dataset_prep.make_hard_subset \
        --dataset virl39k \
        --rollouts_jsonl "$BASE_ROLLOUTS" \
        --mode hard --hard_threshold 0.5 \
        --subset_pct "$sp" --seed "$seed" )
    echo "[$(ts)] wrote $target_parquet"
}

ensure_maxvar_subset() {
    local seed="$1"
    local sp="$SUBSET_PCT"
    local target_parquet="${PARQUET_DIR}/train_maxvar${sp}_seed${seed}_100.parquet"
    if [ -e "$target_parquet" ]; then
        echo "[$(ts)] maxvar subset seed=$seed already exists — skipping generation"
        return 0
    fi
    if [ ! -e "$BASE_ROLLOUTS_N8" ]; then
        echo "[$(ts)] ERROR: maxvar requires n=8 rollouts at $BASE_ROLLOUTS_N8" >&2
        echo "       Run run_qwen3_vl-2b_virl39k_base_rollouts.sh first." >&2
        return 1
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "[$(ts)] DRY_RUN — would generate maxvar seed=$seed"
        return 0
    fi
    ( cd "$REPO_ROOT" && python -m dataset_prep.make_hard_subset \
        --dataset virl39k \
        --rollouts_jsonl "$BASE_ROLLOUTS_N8" \
        --mode maxvar \
        --subset_pct "$sp" --seed "$seed" )
    echo "[$(ts)] wrote $target_parquet"
}

# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------
run_full() {
    local exp="virl39k_full_seed0"   # full has no seed
    local logf="${LOG_DIR}/${exp}.log"
    run_or_echo "$logf" \
        env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
            EXP_NAME="$exp" \
            TOTAL_EPOCHS="$EPOCHS_FULL" \
            DATA_ROOT="$DATA_ROOT" \
        bash "$SCRIPT_DIR/run_qwen3_vl-2b_virl39k_full.sh" "$ENGINE"
}

run_random10() {
    local seed="$1"
    ensure_random_subset "$seed"
    local sp="$SUBSET_PCT"
    local parquet="${PARQUET_DIR}/train_random${sp}_seed${seed}_100.parquet"
    local exp="virl39k_random${sp}_seed${seed}"
    local logf="${LOG_DIR}/${exp}.log"
    # The vanilla random10 launcher reads the parquet from its env-derived
    # path; we override data.train_files via the inner Hydra "$@" passthrough.
    run_or_echo "$logf" \
        env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
            EXP_NAME="$exp" \
            TOTAL_EPOCHS="$EPOCHS_SUBSET" \
            DATA_ROOT="$DATA_ROOT" \
            SUBSET_PCT="$sp" \
        bash "$SCRIPT_DIR/run_qwen3_vl-2b_virl39k_random10.sh" "$ENGINE" \
            data.train_files="$parquet"
}

run_subset_with_parquet() {
    # run_subset_with_parquet <exp_name> <parquet_path>
    local exp="$1"
    local parquet="$2"
    local logf="${LOG_DIR}/${exp}.log"
    run_or_echo "$logf" \
        env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
            EXP_NAME="$exp" \
            TOTAL_EPOCHS="$EPOCHS_SUBSET" \
            DATA_ROOT="$DATA_ROOT" \
            SUBSET_PCT="$SUBSET_PCT" \
        bash "$SCRIPT_DIR/run_qwen3_vl-2b_virl39k_random10.sh" "$ENGINE" \
            data.train_files="$parquet"
}

run_hard10() {
    local seed="$1"
    ensure_hard_subset "$seed"
    local sp="$SUBSET_PCT"
    local parquet="${PARQUET_DIR}/train_hard${sp}_seed${seed}_100.parquet"
    run_subset_with_parquet "virl39k_hard${sp}_seed${seed}" "$parquet"
}

run_maxvar10() {
    local seed="$1"
    ensure_maxvar_subset "$seed" || return 0
    local sp="$SUBSET_PCT"
    local parquet="${PARQUET_DIR}/train_maxvar${sp}_seed${seed}_100.parquet"
    run_subset_with_parquet "virl39k_maxvar${sp}_seed${seed}" "$parquet"
}

_unused_run_hard10_old() {
    local seed="$1"
    local sp="$SUBSET_PCT"
    local parquet="${PARQUET_DIR}/train_hard${sp}_seed${seed}_100.parquet"
    local exp="virl39k_hard${sp}_seed${seed}"
    local logf="${LOG_DIR}/${exp}.log"
    # Reuse the random10 launcher with the parquet path overridden — same
    # hyperparams, same compute budget, only the subset differs. This keeps
    # the comparison apples-to-apples.
    run_or_echo "$logf" \
        env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
            EXP_NAME="$exp" \
            TOTAL_EPOCHS="$EPOCHS_SUBSET" \
            DATA_ROOT="$DATA_ROOT" \
            SUBSET_PCT="$sp" \
        bash "$SCRIPT_DIR/run_qwen3_vl-2b_virl39k_random10.sh" "$ENGINE" \
            data.train_files="$parquet"
}

run_online() {
    # run_online <predictor_type> <seed>
    local predictor="$1"
    local seed="$2"
    local exp="virl39k_online_${predictor}_seed${seed}"
    local logf="${LOG_DIR}/${exp}.log"
    # The inner VLAA launcher hardcodes total_epochs=170; we override via
    # the trailing "$@" Hydra passthrough that the wrapper forwards.
    # Note: ClusterSelector currently has no seed knob — `seed` here only
    # parameterises the EXP_NAME and any trainer-side stochasticity that
    # honours `actor_rollout_ref.actor.seed`. If you want true selection
    # determinism, expose a selector seed in cluster_selector.py (see
    # _rng = np.random.default_rng(...) in __init__).
    run_or_echo "$logf" \
        env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
            DATA_ROOT="$DATA_ROOT" \
            PREDICTOR_TYPE="$predictor" \
            GLOBAL_BUDGET_PCT="$SUBSET_PCT" \
        bash "$SCRIPT_DIR/run_qwen3_vl-2b_virl39k_online_selection.sh" \
            trainer.experiment_name="$exp" \
            trainer.total_epochs="$EPOCHS_ONLINE"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
echo "[$(ts)] Sweep starting"
echo "  METHODS=$METHODS"
echo "  SEEDS=$SEEDS"
echo "  SUBSET_PCT=$SUBSET_PCT"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  LOG_DIR=$LOG_DIR"
echo "  DRY_RUN=$DRY_RUN"
echo

for method in $METHODS; do
    case "$method" in
        full)
            run_full
            ;;
        random10)
            for s in $SEEDS; do run_random10 "$s"; done
            ;;
        hard10)
            for s in $SEEDS; do run_hard10 "$s"; done
            ;;
        online_knn)
            for s in $SEEDS; do run_online knn "$s"; done
            ;;
        online_mlp)
            for s in $SEEDS; do run_online mlp "$s"; done
            ;;
        maxvar10)
            for s in $SEEDS; do run_maxvar10 "$s"; done
            ;;
        online_ridge)
            for s in $SEEDS; do run_online ridge "$s"; done
            ;;
        *)
            echo "[$(ts)] unknown method: $method (skipping)"
            ;;
    esac
done

echo "[$(ts)] Sweep finished. Logs in $LOG_DIR"
