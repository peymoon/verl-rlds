#!/usr/bin/env bash
# Run base-model rollouts on the full ViRL39K training set — no training,
# no weight updates. Useful for diagnosing the difficulty distribution before
# any RL and for seeding the rollout-history buffer analysis.
#
# Mechanism: total_epochs=0 + val_before_train=True + val_files=<train parquet>
# The validation loop rolls out every sample and writes results to
# validation_data_dir. No gradients, no checkpoints.
#
# IMPORTANT: validation rollouts are governed by `rollout.val_kwargs`, NOT
# `rollout.n`. The defaults in verl's rollout.yaml are n=1, do_sample=False,
# temperature=0 (greedy), so without overriding val_kwargs you get a single
# greedy decode per prompt and zero per-sample variance. We override below to
# n=8 stochastic samples (temperature=1.0, top_p=1.0) so the JSONL contains
# 8 rows per prompt and we can compute per-prompt sample variance for the
# max-variance subset construction.
#
# Prereq:
#   bash dataset_prep/prepare.sh virl39k
#
# Env vars:
#   DATA_ROOT, SPLIT (default: train_90_100), CUDA_VISIBLE_DEVICES
#
# Output:
#   /workspace/peyman/outputs/rollouts/virl39k/base_rollouts_<SPLIT>/

set -x
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/rl_data_selection/data}"
DS_ROOT="${DATA_ROOT}/virl39k"
SPLIT="${SPLIT:-train_90_100}"
PARQUET="${DS_ROOT}/parquet/${SPLIT}.parquet"
ENGINE="${1:-vllm}"

[ -e "$PARQUET" ] || { echo "ERROR: missing $PARQUET — run dataset_prep/prepare.sh virl39k" >&2; exit 1; }

# Output dir includes _n8 so we don't clobber the original n=1 greedy file.
EXP_NAME="virl39k_base_rollouts_${SPLIT}_n8"
ROLLOUT_DIR="/workspace/rl_data_selection/peyman/outputs/virl39k/rollouts/virl39k/${EXP_NAME}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2,4,5,7"} \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$PARQUET" \
    data.val_files="$PARQUET" \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-2B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_virl39k_baseline' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=1 \
    trainer.total_epochs=0 \
    trainer.default_local_dir="${ROLLOUT_DIR}_ckpt" \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    trainer.val_before_train=True \
    trainer.validation_data_dir="$ROLLOUT_DIR" "${@:2}"
