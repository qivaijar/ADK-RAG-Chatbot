#!/usr/bin/env bash
set -e

export $(grep -v '^#' .env | xargs)

ENV_VARS=$(grep -v '^#' .env | xargs | sed 's/ /,/g')
HOST=0.0.0.0
PORT=8000

# Deploy to Cloud Run
echo "🚀 Deploying $SERVICE_NAME..."
gcloud run deploy "$SERVICE_NAME" \
  --image="$GOOGLE_CLOUD_LOCATION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/$REPO_NAME/$IMAGE_NAME:latest" \
  --region="$GOOGLE_CLOUD_LOCATION" \
  --platform=managed \
  --allow-unauthenticated \
  --port="$PORT" \
  --cpu=2 \
  --memory=2Gi \
  --timeout=900 \
  --command="adk" \
  --args="api_server,.,--host,$HOST,--port,$PORT" \
  --set-env-vars="PYTHONUNBUFFERED=1${ENV_VARS:+,$ENV_VARS}"

echo "✅ Deployment complete!"
