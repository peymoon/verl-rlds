#!/usr/bin/env bash
# VADE comparison run for ViRL39K + Qwen3-VL-2B-Instruct.
#
# This runs the VADE dynamic beta sampler through our online data-selection
# interface:
#   base rollouts -> initialize a/b counts -> train one selected batch ->
#   update a/b from rollout acc -> reselect the next batch.
#
# Prereq:
#   SPLIT=train_90_100 bash run_qwen3_vl-2b_virl39k_base_rollouts.sh
#
# Common env vars:
#   DATA_ROOT                  root containing data/virl39k
#   TRAIN_SPLIT                default train_90_100
#   TEST_SPLIT                 default test_10_100
#   VADE_INITIAL_ROLLOUTS      base-rollout JSONL directory or VADE JSON file
#   VADE_SAMPLE_METHOD         thompson | UCB | direct (default thompson)
#   VADE_P_FUNCTION            1 for p(1-p), 2 for p(1-p)^2
#   VADE_AB_UPDATE_STRATEGY    1, 2, or 3 (default 1)
#   TRAIN_BATCH_SIZE           default 128
#   SELECTION_BUDGET           default TRAIN_BATCH_SIZE
#   GLOBAL_BUDGET_PCT          default 11.11 cumulative unique cap
#   BUDGET_SCHEDULE            fast phased schedule when GLOBAL_BUDGET_PCT is set
#   SAVE_FREQ                  default 10
#   TEST_FREQ                  default 10
#   ROLLOUT_N                  default 8
#   CUDA_VISIBLE_DEVICES       default 3,4,5,7

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" != --* && "${1:-}" != *=* && -n "${1:-}" ]]; then
    ENGINE="$1"
    shift
else
    ENGINE="${ENGINE:-vllm}"
fi

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DATASET_NAME="virl39k"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_90_100}"
TEST_SPLIT="${TEST_SPLIT:-test_10_100}"

DS_ROOT="${DATA_ROOT}/${DATASET_NAME}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DS_ROOT}/parquet/${TRAIN_SPLIT}.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DS_ROOT}/parquet/${TEST_SPLIT}.parquet}"

DEFAULT_INITIAL="/workspace/rl_data_selection/peyman/outputs/virl39k/rollouts/virl39k/virl39k_base_rollouts_${TRAIN_SPLIT}_n8"
CHUNKED_INITIAL="/workspace/rl_data_selection/peyman/outputs/virl39k/rollouts/${TRAIN_SPLIT}_n8_chunked/merged.jsonl"
if [ ! -e "$DEFAULT_INITIAL" ] && [ -e "$CHUNKED_INITIAL" ]; then
    DEFAULT_INITIAL="$CHUNKED_INITIAL"
fi
VADE_INITIAL_ROLLOUTS="${VADE_INITIAL_ROLLOUTS:-$DEFAULT_INITIAL}"

for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$VADE_INITIAL_ROLLOUTS"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required path: $p" >&2
        echo "       For VADE_INITIAL_ROLLOUTS, first run run_qwen3_vl-2b_virl39k_base_rollouts.sh or override the env var." >&2
        exit 1
    fi
done

export WANDB_RESUME="${WANDB_RESUME:-allow}"

PROJECT_NAME="${PROJECT_NAME:-verl_grpo_virl39k_baseline}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
SELECTION_BUDGET="${SELECTION_BUDGET:-$TRAIN_BATCH_SIZE}"
RESELECT_INTERVAL="${RESELECT_INTERVAL:-1}"
GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-11.11}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"

# Match the cluster data selector's fast phased discovery schedule.  The
# schedule is only forwarded when GLOBAL_BUDGET_PCT is set, because the phase
# transitions are defined as a percentage of the global cap.
BUDGET_SCHEDULE=${BUDGET_SCHEDULE:-'[{until_budget_pct:50,per_round_pct:1.0,interval:4},{until_budget_pct:85,per_round_pct:0.3,interval:10},{until_budget_pct:100,per_round_pct:0.15,interval:16}]'}
if [ -n "$GLOBAL_BUDGET_PCT" ]; then
    GLOBAL_BUDGET_OVERRIDE="data_selection.global_budget_pct=$GLOBAL_BUDGET_PCT"
else
    GLOBAL_BUDGET_OVERRIDE=""
fi
if [ -n "$GLOBAL_BUDGET_PCT" ] && [ "$BUDGET_SCHEDULE" != "none" ] && [ -n "$BUDGET_SCHEDULE" ]; then
    BUDGET_SCHEDULE_NORMALIZED=$(printf '%s' "$BUDGET_SCHEDULE" | sed -E 's/"([A-Za-z_][A-Za-z0-9_]*)"[[:space:]]*:/\1:/g' | tr -d '[:space:]')
    BUDGET_SCHEDULE_OVERRIDE="+data_selection.vade.budget_schedule=${BUDGET_SCHEDULE_NORMALIZED}"
else
    BUDGET_SCHEDULE_OVERRIDE=""
fi

VADE_T="${VADE_T:-1.0}"
VADE_LAM1="${VADE_LAM1:-0.2}"
VADE_LAM2="${VADE_LAM2:-0.999}"
VADE_G="${VADE_G:-8}"
VADE_SAMPLE_METHOD="${VADE_SAMPLE_METHOD:-thompson}"
VADE_P_FUNCTION="${VADE_P_FUNCTION:-1}"
VADE_ALPHA="${VADE_ALPHA:-0.95}"
VADE_SEED="${VADE_SEED:-42}"
VADE_AB_UPDATE_STRATEGY="${VADE_AB_UPDATE_STRATEGY:-1}"
VADE_TAU="${VADE_TAU:-1.0}"
VADE_K="${VADE_K:-100.0}"
SELECTION_REWARD_KEY="${SELECTION_REWARD_KEY:-acc}"
CORRECTNESS_THRESHOLD="${CORRECTNESS_THRESHOLD:-0.5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3,4,5}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-$TENSOR_MODEL_PARALLEL_SIZE}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-$N_GPUS_PER_NODE}"

for int_var in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU SELECTION_BUDGET N_GPUS_PER_NODE ROLLOUT_N ROLLOUT_AGENT_NUM_WORKERS VADE_G; do
    if ! [[ "${!int_var}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $int_var must be a positive integer, got '${!int_var}'." >&2
        exit 1
    fi
done

if (( VADE_G != ROLLOUT_N )); then
    echo "ERROR: VADE_G (${VADE_G}) must match ROLLOUT_N (${ROLLOUT_N}) so correct_count uses the same number of rollouts VADE expects." >&2
    exit 1
fi

suggest_train_batch_size() {
    local candidate="$1"
    while (( candidate < PPO_MINI_BATCH_SIZE ||
             candidate % PPO_MINI_BATCH_SIZE != 0 ||
             (candidate * ROLLOUT_N) % N_GPUS_PER_NODE != 0 )); do
        candidate=$((candidate + 1))
    done
    echo "$candidate"
}

suggest_ppo_mini_batch_size() {
    local candidate="$1"
    while (( (candidate * ROLLOUT_N) % N_GPUS_PER_NODE != 0 ||
             ((candidate * ROLLOUT_N) / N_GPUS_PER_NODE) % PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )); do
        candidate=$((candidate + 1))
    done
    echo "$candidate"
}

REAL_TRAIN_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ROLLOUT_N))

if (( (PPO_MINI_BATCH_SIZE * ROLLOUT_N) % N_GPUS_PER_NODE != 0 ||
      ((PPO_MINI_BATCH_SIZE * ROLLOUT_N) / N_GPUS_PER_NODE) % PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
    SUGGESTED_PPO_MINI_BATCH_SIZE="$(suggest_ppo_mini_batch_size "$PPO_MINI_BATCH_SIZE")"
    echo "ERROR: PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE}) is not compatible with ROLLOUT_N=${ROLLOUT_N}, N_GPUS_PER_NODE=${N_GPUS_PER_NODE}, and PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU}." >&2
    echo "       Try PPO_MINI_BATCH_SIZE=${SUGGESTED_PPO_MINI_BATCH_SIZE}." >&2
    exit 1
fi

if (( REAL_TRAIN_BATCH_SIZE % N_GPUS_PER_NODE != 0 )); then
    SUGGESTED_TRAIN_BATCH_SIZE="$(suggest_train_batch_size "$TRAIN_BATCH_SIZE")"
    echo "ERROR: data.train_batch_size (${TRAIN_BATCH_SIZE}) * rollout.n (${ROLLOUT_N}) = ${REAL_TRAIN_BATCH_SIZE} must be divisible by N_GPUS_PER_NODE (${N_GPUS_PER_NODE})." >&2
    echo "       Try TRAIN_BATCH_SIZE=${SUGGESTED_TRAIN_BATCH_SIZE}." >&2
    exit 1
fi

if (( TRAIN_BATCH_SIZE < PPO_MINI_BATCH_SIZE || TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
    SUGGESTED_TRAIN_BATCH_SIZE="$(suggest_train_batch_size "$TRAIN_BATCH_SIZE")"
    echo "ERROR: TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be a multiple of PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE})." >&2
    echo "       Try TRAIN_BATCH_SIZE=${SUGGESTED_TRAIN_BATCH_SIZE}." >&2
    exit 1
fi

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
if [ -z "${WANDB_RUN_ID:-}" ]; then
    WANDB_RUN_ID="virl39k_vade_${VADE_SAMPLE_METHOD}_pf${VADE_P_FUNCTION}_s${VADE_AB_UPDATE_STRATEGY}_seed${VADE_SEED}_$(date +%Y%m%d_%H%M%S)"
fi
export WANDB_RUN_ID
export WANDB_EXPERIMENT_NAME="${WANDB_EXPERIMENT_NAME:-${WANDB_RUN_ID}}"
export WANDB_PROJECT_NAME_OVERRIDE="$PROJECT_NAME"

EXP_NAME="${EXP_NAME:-${WANDB_RUN_ID}_qwen3vl2b_b${SELECTION_BUDGET}_i${RESELECT_INTERVAL}}"
WANDB_DISPLAY_NAME="${WANDB_EXPERIMENT_NAME:-${EXP_NAME}}"
CKPT_DIR="${CKPT_DIR:-/workspace/rl_data_selection/peyman/outputs/checkpoints/vade/${WANDB_DISPLAY_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/workspace/rl_data_selection/peyman/outputs/rollouts/vade/${WANDB_DISPLAY_NAME}}"
VADE_LOG_STATE_DIR="${VADE_LOG_STATE_DIR:-/workspace/rl_data_selection/peyman/outputs/vade_state/${WANDB_DISPLAY_NAME}}"

echo "[VADE] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, ROLLOUT_N=${ROLLOUT_N}, real_train_batch_size=${REAL_TRAIN_BATCH_SIZE}, SELECTION_BUDGET=${SELECTION_BUDGET}"
echo "[VADE] initial_rollouts=${VADE_INITIAL_ROLLOUTS}, method=${VADE_SAMPLE_METHOD}, p_function=${VADE_P_FUNCTION}, strategy=${VADE_AB_UPDATE_STRATEGY}"
echo "[VADE] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, SAVE_FREQ=${SAVE_FREQ}, TEST_FREQ=${TEST_FREQ}"
if [ -n "$GLOBAL_BUDGET_PCT" ]; then
    echo "[VADE] GLOBAL_BUDGET_PCT=${GLOBAL_BUDGET_PCT}, BUDGET_SCHEDULE=${BUDGET_SCHEDULE}"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data_selection.method=vade \
    data_selection.reselect_schedule=step \
    data_selection.reselect_interval="$RESELECT_INTERVAL" \
    data_selection.selection_budget="$SELECTION_BUDGET" \
    ${GLOBAL_BUDGET_OVERRIDE:-} \
    data_selection.vade.initial_rollout_data_path="$VADE_INITIAL_ROLLOUTS" \
    data_selection.vade.log_state_dir="$VADE_LOG_STATE_DIR" \
    data_selection.vade.T="$VADE_T" \
    data_selection.vade.lam1="$VADE_LAM1" \
    data_selection.vade.lam2="$VADE_LAM2" \
    data_selection.vade.G="$VADE_G" \
    data_selection.vade.sample_method="$VADE_SAMPLE_METHOD" \
    data_selection.vade.p_function="$VADE_P_FUNCTION" \
    data_selection.vade.alpha="$VADE_ALPHA" \
    data_selection.vade.seed="$VADE_SEED" \
    data_selection.vade.a_b_update_strategy="$VADE_AB_UPDATE_STRATEGY" \
    data_selection.vade.tau="$VADE_TAU" \
    data_selection.vade.k="$VADE_K" \
    data_selection.vade.selection_reward_key="$SELECTION_REWARD_KEY" \
    data_selection.vade.correctness_threshold="$CORRECTNESS_THRESHOLD" \
    ${BUDGET_SCHEDULE_OVERRIDE:-} \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-2B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.01}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$WANDB_DISPLAY_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs=300 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_local_dir="$CKPT_DIR" \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_NUM_WORKERS" \
    trainer.rollout_data_dir="$ROLLOUT_DIR" \
    trainer.val_before_train=False \
    trainer.validation_data_dir="${ROLLOUT_DIR}_val" \
    "$@"
