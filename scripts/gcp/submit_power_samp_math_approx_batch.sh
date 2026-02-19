#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Submit power_samp_math_approx.py to Google Cloud Batch.

Required:
  --project-id <PROJECT_ID>
  --region <REGION>
  --image-uri <ARTIFACT_REGISTRY_IMAGE_URI>
  --bucket-path <BUCKET_OR_BUCKET/PREFIX>   (without gs://)
  --profile <a100-40|a100-80|h100|h200>

Optional:
  --job-name <name>                         (default: psamp-math-approx-<profile>-<timestamp>)
  --task-count <int>                        (default: 40)
  --parallelism <int>                       (default depends on profile)
  --provisioning-model <STANDARD|SPOT|FLEX_START>
  --allowed-locations <csv>                 (default: regions/<region>)
  --max-run-duration <seconds+s>            (default: 86400s)
  --hf-secret <secret-version-resource>     (optional Secret Manager version path)
  --wandb-secret <secret-version-resource>  (optional Secret Manager version path)
  --dry-run                                 (generate JSON only, do not submit)

Experiment overrides (all optional):
  --model <model_alias>
  --temp <float>
  --top-k <int>
  --candidate-pool-size <int>
  --rollouts-per-candidate <int>
  --lookahead-tokens <int>
  --block-size <int>
  --use-jackknife <true|false>
  --num-shards <int>                        (default: 5)
  --num-seeds <int>                         (default: 8)
  --wandb-project <name>
  --wandb-entity <name>

Examples:
  scripts/gcp/submit_power_samp_math_approx_batch.sh \
    --project-id tti-rava-vicenteor \
    --region us-east1 \
    --image-uri us-east1-docker.pkg.dev/tti-rava-vicenteor/rws/psamp:v1 \
    --bucket-path my-bucket/reasoning-with-sampling \
    --profile a100-80

  scripts/gcp/submit_power_samp_math_approx_batch.sh \
    --project-id tti-rava-vicenteor \
    --region us-east1 \
    --image-uri us-east1-docker.pkg.dev/tti-rava-vicenteor/rws/psamp:v1 \
    --bucket-path my-bucket/reasoning-with-sampling \
    --profile h100 \
    --parallelism 2 \
    --hf-secret projects/tti-rava-vicenteor/secrets/hf-token/versions/latest \
    --wandb-secret projects/tti-rava-vicenteor/secrets/wandb-api-key/versions/latest
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-}"
IMAGE_URI="${IMAGE_URI:-}"
BUCKET_PATH="${BUCKET_PATH:-}"
PROFILE="${PROFILE:-}"
JOB_NAME="${JOB_NAME:-}"
TASK_COUNT="${TASK_COUNT:-40}"
PARALLELISM="${PARALLELISM:-}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-}"
ALLOWED_LOCATIONS="${ALLOWED_LOCATIONS:-}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-86400s}"
HF_SECRET="${HF_SECRET:-}"
WANDB_SECRET="${WANDB_SECRET:-}"
DRY_RUN=0

MODEL="${MODEL:-qwen_math}"
TEMP="${TEMP:-0.25}"
TOP_K="${TOP_K:-8}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-32}"
ROLLOUTS_PER_CANDIDATE="${ROLLOUTS_PER_CANDIDATE:-8}"
LOOKAHEAD_TOKENS="${LOOKAHEAD_TOKENS:-}"
BLOCK_SIZE="${BLOCK_SIZE:-1}"
USE_JACKKNIFE="${USE_JACKKNIFE:-true}"
NUM_SHARDS="${NUM_SHARDS:-5}"
NUM_SEEDS="${NUM_SEEDS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-reasoning-with-sampling}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-id)
      PROJECT_ID="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --image-uri)
      IMAGE_URI="$2"
      shift 2
      ;;
    --bucket-path)
      BUCKET_PATH="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --task-count)
      TASK_COUNT="$2"
      shift 2
      ;;
    --parallelism)
      PARALLELISM="$2"
      shift 2
      ;;
    --provisioning-model)
      PROVISIONING_MODEL="$2"
      shift 2
      ;;
    --allowed-locations)
      ALLOWED_LOCATIONS="$2"
      shift 2
      ;;
    --max-run-duration)
      MAX_RUN_DURATION="$2"
      shift 2
      ;;
    --hf-secret)
      HF_SECRET="$2"
      shift 2
      ;;
    --wandb-secret)
      WANDB_SECRET="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --temp)
      TEMP="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --candidate-pool-size)
      CANDIDATE_POOL_SIZE="$2"
      shift 2
      ;;
    --rollouts-per-candidate)
      ROLLOUTS_PER_CANDIDATE="$2"
      shift 2
      ;;
    --lookahead-tokens)
      LOOKAHEAD_TOKENS="$2"
      shift 2
      ;;
    --block-size)
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --use-jackknife)
      USE_JACKKNIFE="$2"
      shift 2
      ;;
    --num-shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    --num-seeds)
      NUM_SEEDS="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "${PROJECT_ID}" ]] || die "--project-id is required"
[[ -n "${REGION}" ]] || die "--region is required"
[[ -n "${IMAGE_URI}" ]] || die "--image-uri is required"
[[ -n "${BUCKET_PATH}" ]] || die "--bucket-path is required"
[[ -n "${PROFILE}" ]] || die "--profile is required"

is_int "${TASK_COUNT}" || die "--task-count must be an integer"
is_int "${NUM_SHARDS}" || die "--num-shards must be an integer"
is_int "${NUM_SEEDS}" || die "--num-seeds must be an integer"
if [[ -n "${PARALLELISM}" ]]; then
  is_int "${PARALLELISM}" || die "--parallelism must be an integer"
fi

BUCKET_PATH="${BUCKET_PATH#gs://}"
BUCKET_PATH="${BUCKET_PATH%/}"

TOTAL_TASKS=$((NUM_SHARDS * NUM_SEEDS))
if (( TASK_COUNT > TOTAL_TASKS )); then
  die "--task-count (${TASK_COUNT}) cannot exceed num-shards*num-seeds (${TOTAL_TASKS})"
fi

case "${PROFILE}" in
  a100-40)
    MACHINE_TYPE="a2-highgpu-1g"
    CPU_MILLI=12000
    MEMORY_MIB=80000
    BOOT_DISK_MIB=150000
    DEFAULT_PARALLELISM=4
    DEFAULT_PROVISIONING_MODEL="STANDARD"
    DEFAULT_OMP_THREADS=12
    ;;
  a100-80)
    MACHINE_TYPE="a2-ultragpu-1g"
    CPU_MILLI=12000
    MEMORY_MIB=160000
    BOOT_DISK_MIB=150000
    DEFAULT_PARALLELISM=4
    DEFAULT_PROVISIONING_MODEL="STANDARD"
    DEFAULT_OMP_THREADS=12
    ;;
  h100)
    MACHINE_TYPE="a3-highgpu-1g"
    CPU_MILLI=26000
    MEMORY_MIB=220000
    BOOT_DISK_MIB=200000
    DEFAULT_PARALLELISM=2
    DEFAULT_PROVISIONING_MODEL="SPOT"
    DEFAULT_OMP_THREADS=26
    ;;
  h200)
    MACHINE_TYPE="a3-ultragpu-8g"
    CPU_MILLI=26000
    MEMORY_MIB=220000
    BOOT_DISK_MIB=250000
    DEFAULT_PARALLELISM=1
    DEFAULT_PROVISIONING_MODEL="SPOT"
    DEFAULT_OMP_THREADS=26
    ;;
  *)
    die "Unsupported profile '${PROFILE}'. Use one of: a100-40, a100-80, h100, h200"
    ;;
esac

if [[ -z "${PARALLELISM}" ]]; then
  PARALLELISM="${DEFAULT_PARALLELISM}"
fi
if [[ -z "${PROVISIONING_MODEL}" ]]; then
  PROVISIONING_MODEL="${DEFAULT_PROVISIONING_MODEL}"
fi
if [[ -z "${OMP_NUM_THREADS}" ]]; then
  OMP_NUM_THREADS="${DEFAULT_OMP_THREADS}"
fi

if [[ -z "${ALLOWED_LOCATIONS}" ]]; then
  ALLOWED_LOCATIONS="regions/${REGION}"
fi

if [[ -z "${JOB_NAME}" ]]; then
  JOB_NAME="psamp-math-approx-${PROFILE}-$(date +%Y%m%d-%H%M%S)"
fi
JOB_NAME="$(echo "${JOB_NAME}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9-' '-')"
JOB_NAME="${JOB_NAME#-}"
JOB_NAME="${JOB_NAME%-}"
JOB_NAME="${JOB_NAME:0:63}"
[[ -n "${JOB_NAME}" ]] || die "Derived --job-name is empty after sanitization"

JOB_DIR="${REPO_ROOT}/scripts/gcp/jobs"
mkdir -p "${JOB_DIR}"
CONFIG_PATH="${JOB_DIR}/${JOB_NAME}.json"

SAVE_STR="/mnt/disks/rws/results"

LOCATION_POLICY_JSON="$(
  jq -cn \
    --arg locations "${ALLOWED_LOCATIONS}" \
    '
      if ($locations | length) == 0 then
        {}
      else
        {
          location: {
            allowedLocations: (
              $locations
              | split(",")
              | map(gsub("^ +| +$"; ""))
              | map(select(length > 0))
            )
          }
        }
      end
    '
)"

SECRET_JSON="$(
  jq -cn \
    --arg hf "${HF_SECRET}" \
    --arg wandb "${WANDB_SECRET}" \
    '
      reduce [
        (if $hf == "" then {} else {HF_TOKEN: $hf} end),
        (if $wandb == "" then {} else {WANDB_API_KEY: $wandb} end)
      ][] as $obj ({}; . + $obj)
    '
)"

jq -n \
  --arg image_uri "${IMAGE_URI}" \
  --arg bucket_path "${BUCKET_PATH}" \
  --arg mount_path "/mnt/disks/rws" \
  --arg save_str "${SAVE_STR}" \
  --arg model "${MODEL}" \
  --arg temp "${TEMP}" \
  --arg top_k "${TOP_K}" \
  --arg candidate_pool_size "${CANDIDATE_POOL_SIZE}" \
  --arg rollouts_per_candidate "${ROLLOUTS_PER_CANDIDATE}" \
  --arg lookahead_tokens "${LOOKAHEAD_TOKENS}" \
  --arg block_size "${BLOCK_SIZE}" \
  --arg use_jackknife "${USE_JACKKNIFE}" \
  --arg num_shards "${NUM_SHARDS}" \
  --arg num_seeds "${NUM_SEEDS}" \
  --arg wandb_project "${WANDB_PROJECT}" \
  --arg wandb_entity "${WANDB_ENTITY}" \
  --arg omp_num_threads "${OMP_NUM_THREADS}" \
  --arg machine_type "${MACHINE_TYPE}" \
  --arg provisioning_model "${PROVISIONING_MODEL}" \
  --arg max_run_duration "${MAX_RUN_DURATION}" \
  --argjson task_count "${TASK_COUNT}" \
  --argjson parallelism "${PARALLELISM}" \
  --argjson cpu_milli "${CPU_MILLI}" \
  --argjson memory_mib "${MEMORY_MIB}" \
  --argjson boot_disk_mib "${BOOT_DISK_MIB}" \
  --argjson location_policy "${LOCATION_POLICY_JSON}" \
  --argjson secret_variables "${SECRET_JSON}" \
  '
  {
    taskGroups: [
      {
        taskSpec: {
          runnables: [
            {
              container: {
                imageUri: $image_uri,
                entrypoint: "bash",
                commands: [
                  "-lc",
                  "chmod +x /app/scripts/gcp/power_samp_math_approx_batch_task.sh && /app/scripts/gcp/power_samp_math_approx_batch_task.sh"
                ]
              }
            }
          ],
          computeResource: {
            cpuMilli: $cpu_milli,
            memoryMib: $memory_mib,
            bootDiskMib: $boot_disk_mib
          },
          maxRunDuration: $max_run_duration,
          maxRetryCount: 1,
          volumes: [
            {
              gcs: {
                remotePath: $bucket_path
              },
              mountPath: $mount_path
            }
          ],
          environment:
            (
              {
                variables: {
                  REPO_ROOT: "/app",
                  SAVE_STR: $save_str,
                  MODEL: $model,
                  TEMP: $temp,
                  TOP_K: $top_k,
                  CANDIDATE_POOL_SIZE: $candidate_pool_size,
                  ROLLOUTS_PER_CANDIDATE: $rollouts_per_candidate,
                  LOOKAHEAD_TOKENS: $lookahead_tokens,
                  BLOCK_SIZE: $block_size,
                  USE_JACKKNIFE: $use_jackknife,
                  NUM_SHARDS: $num_shards,
                  NUM_SEEDS: $num_seeds,
                  OMP_NUM_THREADS: $omp_num_threads,
                  WANDB_PROJECT: $wandb_project,
                  WANDB_ENTITY: $wandb_entity,
                  HF_HOME: "/tmp/hf_home",
                  HF_HUB_CACHE: "/tmp/hf_home/hub",
                  HF_DATASETS_CACHE: "/tmp/hf_home/datasets",
                  TRANSFORMERS_CACHE: "/tmp/hf_home/models",
                  HF_HUB_DISABLE_XET: "1",
                  HF_ALLOW_CODE_EVAL: "1",
                  HF_DATASETS_TRUST_REMOTE_CODE: "True",
                  TOKENIZERS_PARALLELISM: "false",
                  NCCL_DEBUG: "WARN",
                  PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:256",
                  CUDA_VISIBLE_DEVICES: "0"
                }
              }
              + (if ($secret_variables | length) > 0 then {secretVariables: $secret_variables} else {} end)
            )
        },
        taskCount: $task_count,
        parallelism: $parallelism
      }
    ],
    allocationPolicy:
      (
        {
          instances: [
            {
              policy: {
                machineType: $machine_type,
                provisioningModel: $provisioning_model,
                installGpuDrivers: true
              }
            }
          ]
        }
        + $location_policy
      ),
    logsPolicy: {
      destination: "CLOUD_LOGGING"
    }
  }
  ' > "${CONFIG_PATH}"

echo "Wrote Batch config:"
echo "  ${CONFIG_PATH}"
echo
echo "Profile: ${PROFILE}"
echo "Machine type: ${MACHINE_TYPE}"
echo "Provisioning model: ${PROVISIONING_MODEL}"
echo "Task count: ${TASK_COUNT}"
echo "Parallelism: ${PARALLELISM}"
echo "Bucket mount: gs://${BUCKET_PATH} -> /mnt/disks/rws"
echo "Image: ${IMAGE_URI}"

if (( DRY_RUN == 1 )); then
  echo
  echo "Dry run enabled; not submitting."
  exit 0
fi

gcloud batch jobs submit "${JOB_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --config="${CONFIG_PATH}"

echo
echo "Submitted job '${JOB_NAME}'."
echo "Describe with:"
echo "  gcloud batch jobs describe ${JOB_NAME} --project=${PROJECT_ID} --location=${REGION}"
