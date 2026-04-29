#!/usr/bin/env bash
# Baseline: GRPO on the full ViRL39K 90% train split. No data selection.
#
# Prereq:
#   cd /workspace/rl_data_selection
#   bash dataset_prep/prepare.sh virl39k
#
# Env vars (overridable):
#   DATA_ROOT     (default: /workspace/rl_data_selection/data)
#   TRAIN_SPLIT   (default: train_90_100)
#   TEST_SPLIT    (default: test_10_100)
#   TOTAL_EPOCHS  (default: 2 — full 39k × 2 ≈ same compute as random10 × 20)
#   CUDA_VISIBLE_DEVICES

set -x
set -euo pipefail


DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DS_ROOT="${DATA_ROOT}/virl39k"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_90_100}"
TEST_SPLIT="${TEST_SPLIT:-test_10_100}"
export WANDB_API_KEY='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'
TRAIN_PARQUET="${DS_ROOT}/parquet/${TRAIN_SPLIT}.parquet"
VAL_PARQUET="${DS_ROOT}/parquet/${TEST_SPLIT}.parquet"

for p in "$TRAIN_PARQUET" "$VAL_PARQUET"; do
    [ -e "$p" ] || { echo "ERROR: missing $p — run dataset_prep/prepare.sh virl39k" >&2; exit 1; }
done

EXP_NAME="${WANDB_EXPERIMENT_NAME:-${EXP_NAME:-virl39k_full_90_qwen3_vl_2b}}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_virl39k_baseline}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.default_local_dir="$CKPT_DIR" \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    trainer.rollout_data_dir="$ROLLOUT_DIR" \
    trainer.val_before_train=False \
    trainer.validation_data_dir="${ROLLOUT_DIR}_val" "${@:2}" \
    +trainer.total_training_steps=300
