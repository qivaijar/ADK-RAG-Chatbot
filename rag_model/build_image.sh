#!/usr/bin/env bash
# =========================================================
# Cloud Run GPU Deployment Script for vLLM Model
# =========================================================
set -e # Exit if any command fails

# --- Load environment variables from .env ---
if [ -f .env ]; then
  echo "Loading environment variables from .env..."
  set -o allexport
  source .env
  set +o allexport
else
  echo "No .env file found. Please create one first."
  exit 1
fi

# --- Derived values ---

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_VERSION}"
gcloud builds submit --tag ${IMAGE_URI} .
