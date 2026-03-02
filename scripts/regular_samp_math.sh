#!/bin/bash
#SBATCH -J regular-math
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

# --- Paths / run params ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/regular_samp_math.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p "/scratch/$USER/logs" "${HF_HOME}"

# Run config (override via environment when calling sbatch).
HF_MODEL_ID="${HF_MODEL_ID:-stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150}"
ENABLE_THINKING="${ENABLE_THINKING:-auto}"
PY_VENV_PATH="${PY_VENV_PATH:-/scratch/$USER/venvs/psamp-qwen3}"
SAVE_STR="${SAVE_STR:-/scratch/$USER/reasoning-with-sampling/results}"

if [[ "${HF_MODEL_ID}" == Qwen/Qwen3-* ]]; then
  TEMP="${TEMP:-0.6}"
  STANDARD_TEMPERATURE="${STANDARD_TEMPERATURE:-0.6}"
  TOP_P="${TOP_P:-0.95}"
  TOP_K="${TOP_K:-20}"
  MIN_P="${MIN_P:-0.0}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
  THINKING_CONTROL_MODE="${THINKING_CONTROL_MODE:-multi_pass}"
else
  TEMP="${TEMP:-0.25}"
  STANDARD_TEMPERATURE="${STANDARD_TEMPERATURE:-none}"
  TOP_P="${TOP_P:-none}"
  TOP_K="${TOP_K:-none}"
  MIN_P="${MIN_P:-none}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
  THINKING_CONTROL_MODE="${THINKING_CONTROL_MODE:-none}"
fi

THINKING_ANSWER_BUDGET_TOKENS="${THINKING_ANSWER_BUDGET_TOKENS:-900}"
MAX_THINKING_TOKENS="${MAX_THINKING_TOKENS:-none}"
MIN_THINKING_TOKENS="${MIN_THINKING_TOKENS:-0}"
IGNORE_EOT_ATTEMPTS="${IGNORE_EOT_ATTEMPTS:-0}"
EOT_TRIGGER_TOPK="${EOT_TRIGGER_TOPK:-1}"
WAIT_TEXT="${WAIT_TEXT:-\\nWait\\n}"
EARLY_STOPPING_TEXT="${EARLY_STOPPING_TEXT:-\\n\\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\\n</think>\\n\\n}"
THINKING_EXTRA_TOKENS="${THINKING_EXTRA_TOKENS:-0}"
mkdir -p "${SAVE_STR}"

echo "== Node =="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID} ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"
echo "BATCH_IDX=${BATCH_IDX} SEED=${SEED}"
echo "HF_MODEL_ID=${HF_MODEL_ID} TEMP=${TEMP} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} ENABLE_THINKING=${ENABLE_THINKING}"
echo "STANDARD_TEMPERATURE=${STANDARD_TEMPERATURE} TOP_P=${TOP_P} TOP_K=${TOP_K} MIN_P=${MIN_P}"
echo "THINKING_CONTROL_MODE=${THINKING_CONTROL_MODE} THINKING_ANSWER_BUDGET_TOKENS=${THINKING_ANSWER_BUDGET_TOKENS}"
echo "MAX_THINKING_TOKENS=${MAX_THINKING_TOKENS} MIN_THINKING_TOKENS=${MIN_THINKING_TOKENS} IGNORE_EOT_ATTEMPTS=${IGNORE_EOT_ATTEMPTS}"
echo "EOT_TRIGGER_TOPK=${EOT_TRIGGER_TOPK} THINKING_EXTRA_TOKENS=${THINKING_EXTRA_TOKENS}"
echo "PY_VENV_PATH=${PY_VENV_PATH}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "SAVE_STR=${SAVE_STR}"
echo

srun --ntasks=1 bash -lc "
source \"$HOME/miniconda3/etc/profile.d/conda.sh\" &&
conda activate psamp &&
if [[ ! -f \"${PY_VENV_PATH}/bin/activate\" ]]; then
  echo \"Missing overlay env at ${PY_VENV_PATH}. Expected ${PY_VENV_PATH}/bin/activate\" >&2 &&
  exit 1;
fi &&
source \"${PY_VENV_PATH}/bin/activate\" &&
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
python \"${REPO_ROOT}/regular_samp_math.py\" \
  --batch_idx \"${BATCH_IDX}\" \
  --temp \"${TEMP}\" \
  --max_new_tokens \"${MAX_NEW_TOKENS}\" \
  --standard_temperature \"${STANDARD_TEMPERATURE}\" \
  --top_p \"${TOP_P}\" \
  --top_k \"${TOP_K}\" \
  --min_p \"${MIN_P}\" \
  --seed \"${SEED}\" \
  --hf_model_id \"${HF_MODEL_ID}\" \
  --enable_thinking \"${ENABLE_THINKING}\" \
  --thinking_control_mode \"${THINKING_CONTROL_MODE}\" \
  --thinking_answer_budget_tokens \"${THINKING_ANSWER_BUDGET_TOKENS}\" \
  --max_thinking_tokens \"${MAX_THINKING_TOKENS}\" \
  --min_thinking_tokens \"${MIN_THINKING_TOKENS}\" \
  --ignore_eot_attempts \"${IGNORE_EOT_ATTEMPTS}\" \
  --eot_trigger_topk \"${EOT_TRIGGER_TOPK}\" \
  --wait_text \"${WAIT_TEXT}\" \
  --early_stopping_text \"${EARLY_STOPPING_TEXT}\" \
  --thinking_extra_tokens \"${THINKING_EXTRA_TOKENS}\" \
  --save_str \"${SAVE_STR}\"
"
