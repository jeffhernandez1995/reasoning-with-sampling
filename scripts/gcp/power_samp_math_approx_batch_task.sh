#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${BATCH_TASK_INDEX:-}" ]]; then
  echo "BATCH_TASK_INDEX is not set. This script must run inside Google Cloud Batch." >&2
  exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-5}"
NUM_SEEDS="${NUM_SEEDS:-8}"
TOTAL_TASKS=$((NUM_SHARDS * NUM_SEEDS))

if (( BATCH_TASK_INDEX >= TOTAL_TASKS )); then
  echo "BATCH_TASK_INDEX=${BATCH_TASK_INDEX} out of range for TOTAL_TASKS=${TOTAL_TASKS}" >&2
  exit 1
fi

SEED=$((BATCH_TASK_INDEX % NUM_SEEDS))
BATCH_IDX=$((BATCH_TASK_INDEX / NUM_SEEDS))

REPO_ROOT="${REPO_ROOT:-/app}"
SAVE_STR="${SAVE_STR:-/mnt/disks/rws/results}"
mkdir -p "${SAVE_STR}"

# Hugging Face cache defaults to local disk for better runtime performance.
export HF_HOME="${HF_HOME:-/tmp/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/models}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-True}"

CPU_COUNT="$(nproc)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_COUNT}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-qwen_math}"
TEMP="${TEMP:-0.25}"
TOP_K="${TOP_K:-8}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-32}"
ROLLOUTS_PER_CANDIDATE="${ROLLOUTS_PER_CANDIDATE:-8}"
LOOKAHEAD_TOKENS="${LOOKAHEAD_TOKENS:-}"
BLOCK_SIZE="${BLOCK_SIZE:-1}"
USE_JACKKNIFE="${USE_JACKKNIFE:-true}"

EXTRA_ARGS=()
if [[ -n "${LOOKAHEAD_TOKENS}" ]]; then
  EXTRA_ARGS+=(--lookahead_tokens "${LOOKAHEAD_TOKENS}")
fi

echo "== Batch task metadata =="
echo "HOSTNAME=$(hostname)"
echo "BATCH_TASK_INDEX=${BATCH_TASK_INDEX}"
echo "TOTAL_TASKS=${TOTAL_TASKS} (NUM_SHARDS=${NUM_SHARDS}, NUM_SEEDS=${NUM_SEEDS})"
echo "BATCH_IDX=${BATCH_IDX} SEED=${SEED}"
echo "MODEL=${MODEL} TEMP=${TEMP}"
echo "TOP_K=${TOP_K} CANDIDATE_POOL_SIZE=${CANDIDATE_POOL_SIZE} ROLLOUTS_PER_CANDIDATE=${ROLLOUTS_PER_CANDIDATE}"
echo "LOOKAHEAD_TOKENS=${LOOKAHEAD_TOKENS} BLOCK_SIZE=${BLOCK_SIZE} USE_JACKKNIFE=${USE_JACKKNIFE}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "SAVE_STR=${SAVE_STR}"
echo

cd "${REPO_ROOT}"
python3 "${REPO_ROOT}/power_samp_math_approx.py" \
  --batch_idx "${BATCH_IDX}" \
  --temp "${TEMP}" \
  --seed "${SEED}" \
  --model "${MODEL}" \
  --save_str "${SAVE_STR}" \
  --top_k "${TOP_K}" \
  --candidate_pool_size "${CANDIDATE_POOL_SIZE}" \
  --rollouts_per_candidate "${ROLLOUTS_PER_CANDIDATE}" \
  --block_size "${BLOCK_SIZE}" \
  --use_jackknife "${USE_JACKKNIFE}" \
  "${EXTRA_ARGS[@]}"
