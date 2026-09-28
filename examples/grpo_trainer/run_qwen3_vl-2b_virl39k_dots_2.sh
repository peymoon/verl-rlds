#!/usr/bin/env bash
# DOTS comparison run for ViRL39K + Qwen3-VL-2B-Instruct.
#
# This uses the multimodal-safe DOTS selector in verl/trainer/ppo/data_selector:
# random reference rollouts -> DOTS embedding predictor -> target-difficulty
# softmax sampling.  If DOTS_TEACHER_CHECKPOINT is set, the learned DOTS
# regressor is loaded exactly.  If it is empty, the script runs the labeled
# identity-attention fallback so the comparison pipeline is still executable
# with Qwen3-VL embeddings.
#
# Common env vars:
#   DATA_ROOT                  root containing data/virl39k
#   TRAIN_SPLIT                default train_90_100_stratified_seed1234
#   TEST_SPLIT                 default test_10_100_stratified_seed1234
#   EMBEDDINGS_FILE            default Qwen-VL embeddings for TRAIN_SPLIT
#   DATASET_JSON               default records/TRAIN_SPLIT.jsonl
#   DOTS_TEACHER_CHECKPOINT    optional learned DOTS teacher checkpoint
#   DOTS_REQUIRE_TEACHER       true to fail if no checkpoint is supplied
#   DOTS_TEACHER_DEVICE        auto | cpu | cuda | cuda:0 ...
#   DOTS_REF_SIZE              default 64
#   DOTS_ALPHA                 default 0.5
#   DOTS_SELECTION_TAU         default 0.02
#   DOTS_TEACHER_TAU           default 1.0
#   SELECTION_BUDGET_PCT       default 0.37 (~one 128-sample batch on ViRL39K)
#   GLOBAL_BUDGET_PCT          default 10.0
#   RESELECT_INTERVAL          default 4
#   DOTS_COUNT_REFERENCES_IN_BUDGET default true
#   TRAIN_BATCH_SIZE           default 128 (2-GPU divisible with rollout.n=8)
#   PPO_MINI_BATCH_SIZE        default 32 (keeps four PPO minibatches per 128 prompts)
#   PPO_MICRO_BATCH_SIZE_PER_GPU default 8
#   ROLLOUT_N                  default 8
#   CUDA_VISIBLE_DEVICES       default 0,5
#   N_GPUS_PER_NODE            default 2
#   ROLLOUT_TP_SIZE            default 1
#
# Usage:
#   bash run_qwen3_vl-2b_virl39k_dots_2.sh
#   DOTS_TEACHER_CHECKPOINT=/path/to/qwen3vl_dots_teacher.pt \
#       bash run_qwen3_vl-2b_virl39k_dots_2.sh

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" != --* && "${1:-}" != *=* && -n "${1:-}" ]]; then
    ENGINE="$1"
    shift
else
    ENGINE="${ENGINE:-vllm}"
fi

DEFAULT_DATA_ROOT="/workspace/rl_data_selection/data"
if [ ! -d "$DEFAULT_DATA_ROOT" ] && [ -d "/data/smalekmohammadi/peymoon/rl_data_selection/data" ]; then
    DEFAULT_DATA_ROOT="/data/smalekmohammadi/peymoon/rl_data_selection/data"
fi

DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
DATASET_NAME="virl39k"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_90_100_stratified_seed1234}"
TEST_SPLIT="${TEST_SPLIT:-test_10_100_stratified_seed1234}"

DS_ROOT="${DATA_ROOT}/${DATASET_NAME}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DS_ROOT}/parquet/${TRAIN_SPLIT}.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DS_ROOT}/parquet/${TEST_SPLIT}.parquet}"
EMBEDDINGS_FILE="${EMBEDDINGS_FILE:-${DS_ROOT}/embeddings/qwen_embeddings_${TRAIN_SPLIT}.npz}"
DATASET_JSON="${DATASET_JSON:-${DS_ROOT}/records/${TRAIN_SPLIT}.jsonl}"

for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$EMBEDDINGS_FILE" "$DATASET_JSON"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing required file: $p" >&2
        echo "       Run dataset_prep/prepare.sh virl39k or override DATA_ROOT/TRAIN_SPLIT/EMBEDDINGS_FILE." >&2
        exit 1
    fi
done

DOTS_TEACHER_CHECKPOINT="${DOTS_TEACHER_CHECKPOINT:-}"
if [ -n "$DOTS_TEACHER_CHECKPOINT" ] && [ ! -e "$DOTS_TEACHER_CHECKPOINT" ]; then
    echo "ERROR: DOTS_TEACHER_CHECKPOINT does not exist: $DOTS_TEACHER_CHECKPOINT" >&2
    exit 1
fi

if [ -n "$DOTS_TEACHER_CHECKPOINT" ]; then
    DOTS_TEACHER_OVERRIDE="data_selection.dots.teacher_checkpoint=$DOTS_TEACHER_CHECKPOINT"
    DOTS_MODE_SUFFIX="teacher"
else
    DOTS_TEACHER_OVERRIDE="data_selection.dots.teacher_checkpoint=null"
    DOTS_MODE_SUFFIX="identity_attention"
fi

PROJECT_NAME="${PROJECT_NAME:-verl_grpo_virl39k_baseline}"
SELECTION_BUDGET_PCT="${SELECTION_BUDGET_PCT:-0.37}"
GLOBAL_BUDGET_PCT="${GLOBAL_BUDGET_PCT:-10.0}"
RESELECT_INTERVAL="${RESELECT_INTERVAL:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
EXCLUDE_ALREADY_SELECTED="${EXCLUDE_ALREADY_SELECTED:-true}"
SKIP_REFERENCE_WHEN_FROZEN="${SKIP_REFERENCE_WHEN_FROZEN:-true}"
SELECTION_REWARD_KEY="${SELECTION_REWARD_KEY:-acc}"

DOTS_REF_SIZE="${DOTS_REF_SIZE:-64}"
DOTS_ALPHA="${DOTS_ALPHA:-0.5}"
DOTS_SELECTION_TAU="${DOTS_SELECTION_TAU:-0.02}"
DOTS_TEACHER_TAU="${DOTS_TEACHER_TAU:-1.0}"
DOTS_TEACHER_BATCH_SIZE="${DOTS_TEACHER_BATCH_SIZE:-128}"
DOTS_TEACHER_DEVICE="${DOTS_TEACHER_DEVICE:-auto}"
DOTS_TEACHER_TOP_K="${DOTS_TEACHER_TOP_K:-32}"
DOTS_REQUIRE_TEACHER="${DOTS_REQUIRE_TEACHER:-false}"
DOTS_NORMALIZE_EMBEDDINGS="${DOTS_NORMALIZE_EMBEDDINGS:-true}"
DOTS_COUNT_REFERENCES_IN_BUDGET="${DOTS_COUNT_REFERENCES_IN_BUDGET:-true}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,5}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-$N_GPUS_PER_NODE}"

for int_var in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU N_GPUS_PER_NODE ROLLOUT_N ROLLOUT_AGENT_NUM_WORKERS DOTS_REF_SIZE; do
    if ! [[ "${!int_var}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $int_var must be a positive integer, got '${!int_var}'." >&2
        exit 1
    fi
done

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

suggest_dots_ref_size() {
    local candidate="$1"
    while (( (candidate * ROLLOUT_N) % ROLLOUT_AGENT_NUM_WORKERS != 0 )); do
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

NORMALIZED_PPO_MINI=$((PPO_MINI_BATCH_SIZE * ROLLOUT_N / N_GPUS_PER_NODE))

if (( REAL_TRAIN_BATCH_SIZE % N_GPUS_PER_NODE != 0 )); then
    SUGGESTED_TRAIN_BATCH_SIZE="$(suggest_train_batch_size "$TRAIN_BATCH_SIZE")"
    echo "ERROR: data.train_batch_size (${TRAIN_BATCH_SIZE}) * rollout.n (${ROLLOUT_N}) = ${REAL_TRAIN_BATCH_SIZE} must be divisible by N_GPUS_PER_NODE (${N_GPUS_PER_NODE})." >&2
    echo "       Try TRAIN_BATCH_SIZE=${SUGGESTED_TRAIN_BATCH_SIZE} for this ${N_GPUS_PER_NODE}-GPU run." >&2
    exit 1
fi

if (( TRAIN_BATCH_SIZE < PPO_MINI_BATCH_SIZE || TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
    SUGGESTED_TRAIN_BATCH_SIZE="$(suggest_train_batch_size "$TRAIN_BATCH_SIZE")"
    echo "ERROR: TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be a multiple of PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE})." >&2
    echo "       Try TRAIN_BATCH_SIZE=${SUGGESTED_TRAIN_BATCH_SIZE}." >&2
    exit 1
fi

if (( (DOTS_REF_SIZE * ROLLOUT_N) % ROLLOUT_AGENT_NUM_WORKERS != 0 )); then
    SUGGESTED_DOTS_REF_SIZE="$(suggest_dots_ref_size "$DOTS_REF_SIZE")"
    echo "ERROR: DOTS_REF_SIZE (${DOTS_REF_SIZE}) * ROLLOUT_N (${ROLLOUT_N}) must be divisible by ROLLOUT_AGENT_NUM_WORKERS (${ROLLOUT_AGENT_NUM_WORKERS})." >&2
    echo "       Try DOTS_REF_SIZE=${SUGGESTED_DOTS_REF_SIZE} for this worker count." >&2
    exit 1
fi

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
EXP_NAME="${EXP_NAME:-virl39k_dots_${DOTS_MODE_SUFFIX}_qwen3vl2b_b${SELECTION_BUDGET_PCT}_g${GLOBAL_BUDGET_PCT}_i${RESELECT_INTERVAL}}"
CKPT_DIR="${CKPT_DIR:-/workspace/rl_data_selection/outputs/virl39k/checkpoints/virl39k/${EXP_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/workspace/rl_data_selection/outputs/virl39k/rollouts/virl39k/${EXP_NAME}}"

echo "[DOTS] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, ROLLOUT_N=${ROLLOUT_N}, real_train_batch_size=${REAL_TRAIN_BATCH_SIZE}, PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}, normalized_ppo_mini=${NORMALIZED_PPO_MINI}"
echo "[DOTS] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, N_GPUS_PER_NODE=${N_GPUS_PER_NODE}, ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS}, DOTS_REF_SIZE=${DOTS_REF_SIZE}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
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
    data_selection.method=dots \
    data_selection.reselect_schedule=step \
    data_selection.reselect_interval="$RESELECT_INTERVAL" \
    data_selection.selection_budget_pct="$SELECTION_BUDGET_PCT" \
    data_selection.global_budget_pct="$GLOBAL_BUDGET_PCT" \
    data_selection.dots.embeddings_file="$EMBEDDINGS_FILE" \
    data_selection.dots.dataset_json_file="$DATASET_JSON" \
    data_selection.dots.normalize_embeddings="$DOTS_NORMALIZE_EMBEDDINGS" \
    data_selection.dots.ref_size="$DOTS_REF_SIZE" \
    data_selection.dots.alpha="$DOTS_ALPHA" \
    data_selection.dots.tau="$DOTS_SELECTION_TAU" \
    data_selection.dots.exclude_already_selected="$EXCLUDE_ALREADY_SELECTED" \
    data_selection.dots.skip_reference_when_frozen="$SKIP_REFERENCE_WHEN_FROZEN" \
    data_selection.dots.count_references_in_budget="$DOTS_COUNT_REFERENCES_IN_BUDGET" \
    "$DOTS_TEACHER_OVERRIDE" \
    data_selection.dots.teacher_batch_size="$DOTS_TEACHER_BATCH_SIZE" \
    data_selection.dots.teacher_device="$DOTS_TEACHER_DEVICE" \
    data_selection.dots.teacher_top_k="$DOTS_TEACHER_TOP_K" \
    data_selection.dots.teacher_tau="$DOTS_TEACHER_TAU" \
    data_selection.dots.require_teacher_checkpoint="$DOTS_REQUIRE_TEACHER" \
    data_selection.dots.selection_reward_key="$SELECTION_REWARD_KEY" \
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
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-10}" \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.total_epochs=300 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_local_dir="$CKPT_DIR" \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_NUM_WORKERS" \
    trainer.rollout_data_dir="$ROLLOUT_DIR" \
    trainer.val_before_train=False \
    trainer.validation_data_dir="${ROLLOUT_DIR}_val" \
    "$@"
