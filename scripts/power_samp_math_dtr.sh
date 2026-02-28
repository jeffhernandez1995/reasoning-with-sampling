#!/bin/bash
#SBATCH -J psamp-math-dtr
#SBATCH -p commons
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --array=0-9
#SBATCH -o /scratch/%u/logs/%x-%A_%a.out
#SBATCH -e /scratch/%u/logs/%x-%A_%a.err
#SBATCH --mail-user=jeh16@rice.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# --- map array id -> (batch_idx, seed) ---
NUM_SHARDS=5
NUM_SEEDS=2
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
export NCCL_DEBUG="WARN"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export PYTHONFAULTHANDLER="1"
export TORCH_SHOW_CPP_STACKTRACES="1"
export CUDA_LAUNCH_BLOCKING="0"

# --- Paths / run params ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/power_samp_math_dtr.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p "/scratch/$USER/logs" "${HF_HOME}"

# Force run config in-script to avoid inheriting submit-shell overrides.
SAMPLING_METHOD="power_dtr"
MODEL="qwen_math"
TEMP="0.25"
MAX_NEW_TOKENS="3072"
TOP_K="8"
LOOKAHEAD_TOKENS="16"
BRANCH_FACTOR="8"
BEAM_WIDTH="16"
INCLUDE_EOS_IN_BRANCH="true"
PRUNE_LOGW_MARGIN=""
MAX_FORWARD_BATCH_SIZE=""
SAVE_STR="/scratch/$USER/reasoning-with-sampling/results"
MAX_QUESTIONS=""
SAVE_EVERY="5"
DEBUG_VERBOSE="true"
CUDA_SYNC="false"
mkdir -p "${SAVE_STR}"

EXTRA_ARGS=""
if [[ -n "${PRUNE_LOGW_MARGIN}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --prune_logw_margin ${PRUNE_LOGW_MARGIN}"
fi
if [[ -n "${MAX_FORWARD_BATCH_SIZE}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --max_forward_batch_size ${MAX_FORWARD_BATCH_SIZE}"
fi
if [[ -n "${MAX_QUESTIONS}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --max_questions ${MAX_QUESTIONS}"
fi
EXTRA_ARGS="${EXTRA_ARGS} --save_every ${SAVE_EVERY} --debug_verbose ${DEBUG_VERBOSE} --cuda_sync ${CUDA_SYNC}"

echo "== Node =="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID} ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"
echo "BATCH_IDX=${BATCH_IDX} SEED=${SEED}"
echo "SAMPLING_METHOD=${SAMPLING_METHOD} MODEL=${MODEL} TEMP=${TEMP} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "TOP_K=${TOP_K} LOOKAHEAD_TOKENS=${LOOKAHEAD_TOKENS} BRANCH_FACTOR=${BRANCH_FACTOR} BEAM_WIDTH=${BEAM_WIDTH}"
echo "INCLUDE_EOS_IN_BRANCH=${INCLUDE_EOS_IN_BRANCH} PRUNE_LOGW_MARGIN=${PRUNE_LOGW_MARGIN} MAX_FORWARD_BATCH_SIZE=${MAX_FORWARD_BATCH_SIZE}"
echo "MAX_QUESTIONS=${MAX_QUESTIONS} SAVE_EVERY=${SAVE_EVERY} DEBUG_VERBOSE=${DEBUG_VERBOSE} CUDA_SYNC=${CUDA_SYNC}"
echo "PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER} TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES} CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "SAVE_STR=${SAVE_STR}"
echo

RUN_CMD="python \"${REPO_ROOT}/power_samp_math_dtr.py\" \
  --sampling_method \"${SAMPLING_METHOD}\" \
  --batch_idx \"${BATCH_IDX}\" \
  --temp \"${TEMP}\" \
  --seed \"${SEED}\" \
  --model \"${MODEL}\" \
  --save_str \"${SAVE_STR}\" \
  --max_new_tokens \"${MAX_NEW_TOKENS}\" \
  --top_k \"${TOP_K}\" \
  --lookahead_tokens \"${LOOKAHEAD_TOKENS}\" \
  --branch_factor \"${BRANCH_FACTOR}\" \
  --beam_width \"${BEAM_WIDTH}\" \
  --include_eos_in_branch \"${INCLUDE_EOS_IN_BRANCH}\" \
  ${EXTRA_ARGS}"

srun --ntasks=1 bash -lc "
source \"$HOME/miniconda3/etc/profile.d/conda.sh\" &&
conda activate psamp &&
unset LD_PRELOAD &&
if [[ -n \"\${LD_LIBRARY_PATH:-}\" ]]; then
  CLEAN_LD_LIBRARY_PATH=\$(printf '%s' \"\$LD_LIBRARY_PATH\" | tr ':' '\n' | grep -v '^/opt/apps/xalt/default/lib64$' | paste -sd: -);
else
  CLEAN_LD_LIBRARY_PATH='';
fi &&
export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\${CLEAN_LD_LIBRARY_PATH:+:\$CLEAN_LD_LIBRARY_PATH}\" &&
export SSL_CERT_FILE=\"\$CONDA_PREFIX/ssl/cert.pem\" &&
echo \"CONDA_PREFIX=\$CONDA_PREFIX\" &&
echo \"LD_LIBRARY_PATH=\$LD_LIBRARY_PATH\" &&
cd \"${REPO_ROOT}\" &&
echo \"CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}\" &&
python -V &&
(nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader || true) &&
${RUN_CMD}
"
