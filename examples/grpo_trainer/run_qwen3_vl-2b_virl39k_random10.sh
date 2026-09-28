#!/usr/bin/env bash
# Compute-matched random-subset baseline for the canonical ViRL39K split.
#
# SUBSET_SEED chooses the fixed random subset; RUN_SEED controls every training
# RNG. They are separate and both appear in the run identity.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DS_ROOT="${DATA_ROOT}/virl39k"
SUBSET_PCT="${SUBSET_PCT:-10}"
SUBSET_SEED="${SUBSET_SEED:-42}"
RUN_SEED="${RUN_SEED:-$SUBSET_SEED}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train_90_100_stratified_seed1234}"
TEST_SPLIT="${TEST_SPLIT:-test_10_100_stratified_seed1234}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
FULL_DETERMINISM="${FULL_DETERMINISM:-true}"
RESUME_MODE="${RESUME_MODE:-disable}"

for value_name in SUBSET_SEED RUN_SEED; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $value_name must be a non-negative integer, got '$value'" >&2
        exit 1
    fi
done
case "$RESUME_MODE" in
    disable) WANDB_RESUME_DEFAULT=never ;;
    auto|resume_path) WANDB_RESUME_DEFAULT=allow ;;
    *)
        echo "ERROR: RESUME_MODE must be disable, auto, or resume_path" >&2
        exit 1
        ;;
esac
export PYTHONHASHSEED="$RUN_SEED"
export WANDB_RESUME="${WANDB_RESUME:-$WANDB_RESUME_DEFAULT}"

SOURCE_TAG="${SOURCE_SPLIT#train_}"
DEFAULT_PARQUET="${DS_ROOT}/parquet/train_random${SUBSET_PCT}_seed${SUBSET_SEED}_from_${SOURCE_TAG}_100.parquet"
TRAIN_PARQUET="${TRAIN_PARQUET:-$DEFAULT_PARQUET}"
VAL_PARQUET="${DS_ROOT}/parquet/${TEST_SPLIT}.parquet"

for path in "$TRAIN_PARQUET" "$VAL_PARQUET"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: missing $path" >&2
        echo "Generate the subset with:" >&2
        echo "  python -m dataset_prep.make_random_subset --dataset virl39k --data_root $DATA_ROOT --train_split $SOURCE_SPLIT --subset_pct $SUBSET_PCT --seed $SUBSET_SEED" >&2
        exit 1
    fi
done

echo "[random baseline] Input artifact SHA-256:"
sha256sum "$TRAIN_PARQUET" "$VAL_PARQUET"

DEFAULT_NAME="virl39k_random${SUBSET_PCT}_subsetseed${SUBSET_SEED}_trainseed${RUN_SEED}_steps${TOTAL_TRAINING_STEPS}"
EXP_NAME="${EXP_NAME:-$DEFAULT_NAME}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_virl39k_seeded_clean}"
ENGINE="${1:-vllm}"

CKPT_DIR="/workspace/rl_data_selection/outputs/virl39k/checkpoints/virl39k/${EXP_NAME}"
ROLLOUT_DIR="/workspace/rl_data_selection/outputs/virl39k/rollouts/virl39k/${EXP_NAME}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"} \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.seed="$RUN_SEED" \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-2B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.seed="$RUN_SEED" \
    actor_rollout_ref.actor.fsdp_config.full_determinism="$FULL_DETERMINISM" \
    actor_rollout_ref.actor.data_loader_seed="$RUN_SEED" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.seed="$RUN_SEED" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.seed="$RUN_SEED" \
    actor_rollout_ref.ref.fsdp_config.full_determinism="$FULL_DETERMINISM" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-10}" \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.total_epochs=300 \
    ++trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.default_local_dir="$CKPT_DIR" \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    trainer.rollout_data_dir="$ROLLOUT_DIR" \
    trainer.val_before_train=True \
    trainer.validation_data_dir="${ROLLOUT_DIR}_val" "${@:2}"
