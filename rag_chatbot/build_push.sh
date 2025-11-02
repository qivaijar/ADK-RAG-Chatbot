#!/usr/bin/env bash
set -e

export $(grep -v '^#' .env | xargs)

echo "🏗️ Building and pushing $IMAGE_NAME..."

# Build the Docker image
gcloud builds submit --tag "$GOOGLE_CLOUD_LOCATION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/$REPO_NAME/$IMAGE_NAME:latest" .

echo "✅ Build and push complete!"
