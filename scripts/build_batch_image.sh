#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-rws}"
IMAGE_NAME="${IMAGE_NAME:-psamp}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Set PROJECT_ID first, e.g. export PROJECT_ID=my-gcp-project" >&2
  exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${TAG}"

if ! gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}"
fi

echo "Building and pushing ${IMAGE_URI}"
gcloud builds submit --project="${PROJECT_ID}" --tag "${IMAGE_URI}" .
echo
echo "Done. Image URI:"
echo "${IMAGE_URI}"
