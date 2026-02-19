#!/bin/bash
#SBATCH -J psamp-math-approx
#SBATCH -p commons
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --array=0-39
#SBATCH -o /scratch/%u/logs/%x-%A_%a.out
#SBATCH -e /scratch/%u/logs/%x-%A_%a.err

set -euo pipefail

# --- map array id -> (batch_idx, seed) ---

NUM_SHARDS=5
NUM_SEEDS=8
TOTAL_TASKS=$((NUM_SHARDS * NUM_SEEDS))
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for ${TOTAL_TASKS} tasks" >&2
  exit 1
fi
SEED=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
BATCH_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

# --- Hugging Face cache & token ---
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/models}"
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.env" ]]; then
  export HF_TOKEN="$(grep -E '^HUGGINGFACE_API_KEY=' "$HOME/.env" | cut -d= -f2- | tr -d '"')"
fi
export HF_HUB_DISABLE_XET=1
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

# --- Runtime tuning ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-26}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# --- Paths / run params ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/power_samp_math_approx.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p "/scratch/$USER/logs" "${HF_HOME}"

MODEL="${MODEL:-qwen_math}"
TEMP="${TEMP:-0.25}"
TOP_K="${TOP_K:-8}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-8}"
ROLLOUTS_PER_CANDIDATE="${ROLLOUTS_PER_CANDIDATE:-4}"
LOOKAHEAD_TOKENS="${LOOKAHEAD_TOKENS:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
USE_JACKKNIFE="${USE_JACKKNIFE:-true}"
SAVE_STR="${SAVE_STR:-/scratch/$USER/reasoning-with-sampling/results}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
SAVE_EVERY="${SAVE_EVERY:-5}"
DEBUG_VERBOSE="${DEBUG_VERBOSE:-true}"
CUDA_SYNC="${CUDA_SYNC:-false}"
mkdir -p "${SAVE_STR}"

EXTRA_ARGS=""
if [[ -n "${LOOKAHEAD_TOKENS}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --lookahead_tokens ${LOOKAHEAD_TOKENS}"
fi
if [[ -n "${MAX_QUESTIONS}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --max_questions ${MAX_QUESTIONS}"
fi
EXTRA_ARGS="${EXTRA_ARGS} --save_every ${SAVE_EVERY} --debug_verbose ${DEBUG_VERBOSE} --cuda_sync ${CUDA_SYNC}"

echo "== Node =="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID} ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"
echo "BATCH_IDX=${BATCH_IDX} SEED=${SEED}"
echo "MODEL=${MODEL} TEMP=${TEMP}"
echo "TOP_K=${TOP_K} CANDIDATE_POOL_SIZE=${CANDIDATE_POOL_SIZE} ROLLOUTS_PER_CANDIDATE=${ROLLOUTS_PER_CANDIDATE}"
echo "LOOKAHEAD_TOKENS=${LOOKAHEAD_TOKENS} BLOCK_SIZE=${BLOCK_SIZE} USE_JACKKNIFE=${USE_JACKKNIFE}"
echo "MAX_QUESTIONS=${MAX_QUESTIONS} SAVE_EVERY=${SAVE_EVERY} DEBUG_VERBOSE=${DEBUG_VERBOSE} CUDA_SYNC=${CUDA_SYNC}"
echo "PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER} TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES} CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "SAVE_STR=${SAVE_STR}"
echo

RUN_CMD="python \"${REPO_ROOT}/power_samp_math_approx.py\" \
  --batch_idx \"${BATCH_IDX}\" \
  --temp \"${TEMP}\" \
  --seed \"${SEED}\" \
  --model \"${MODEL}\" \
  --save_str \"${SAVE_STR}\" \
  --top_k \"${TOP_K}\" \
  --candidate_pool_size \"${CANDIDATE_POOL_SIZE}\" \
  --rollouts_per_candidate \"${ROLLOUTS_PER_CANDIDATE}\" \
  --block_size \"${BLOCK_SIZE}\" \
  --use_jackknife \"${USE_JACKKNIFE}\" \
  ${EXTRA_ARGS}"

srun --ntasks=1 bash -lc "
source \"\$(conda info --base)/etc/profile.d/conda.sh\" &&
conda activate psamp &&
cd \"${REPO_ROOT}\" &&
echo \"CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}\" &&
python -V &&
(nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader || true) &&
${RUN_CMD}
"
