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

echo "Starting Cloud Run deployment for ${SERVICE_NAME}..."
echo "Image: ${IMAGE_URI}"
echo "Region: ${REGION}"

# --- Deploy to Cloud Run ---
gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 8 \
  --memory 32Gi \
  --gpu 1 \
  --no-cpu-boost \
  --max-instances 1 \
  --min-instances 0 \
  --gpu-type nvidia-l4 \
  --no-cpu-throttling \
  --concurrency 1 \
  --timeout 300 \
  --service-account "${SERVICE_ACCOUNT}" \
  --ingress all \
  --no-gpu-zonal-redundancy \
  --set-env-vars "MODEL_NAME=${MODEL_NAME},HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN},MAX_MODEL_LEN=${MAX_MODEL_LEN},MAX_NUM_SEQS=${MAX_NUM_SEQS}" \
  --startup-probe=timeoutSeconds=10,tcpSocket.port=8080,initialDelaySeconds=240,periodSeconds=60,failureThreshold=20

echo "Deployment complete!"
echo
echo "Service URL:"
gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --format="value(status.url)"
