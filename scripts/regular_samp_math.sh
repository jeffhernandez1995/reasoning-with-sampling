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

# Force run config in-script to avoid inheriting submit-shell overrides.
HF_MODEL_ID="stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150"
TEMP="0.25"
MAX_NEW_TOKENS="3072"
SAVE_STR="/scratch/$USER/reasoning-with-sampling/results"
mkdir -p "${SAVE_STR}"

echo "== Node =="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID} ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"
echo "BATCH_IDX=${BATCH_IDX} SEED=${SEED}"
echo "HF_MODEL_ID=${HF_MODEL_ID} TEMP=${TEMP} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "SAVE_STR=${SAVE_STR}"
echo

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
python \"${REPO_ROOT}/regular_samp_math.py\" \
  --batch_idx \"${BATCH_IDX}\" \
  --temp \"${TEMP}\" \
  --max_new_tokens \"${MAX_NEW_TOKENS}\" \
  --seed \"${SEED}\" \
  --hf_model_id \"${HF_MODEL_ID}\" \
  --save_str \"${SAVE_STR}\"
"
