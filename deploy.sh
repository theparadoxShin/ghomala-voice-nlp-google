#!/bin/bash
# NAM SA' — Deploy to Cloud Run (one command)
# Usage: ./deploy.sh YOUR_PROJECT_ID

set -e

PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="us-central1"
SERVICE="nam-sa"

echo "🌅 Deploying NAM SA' to Cloud Run..."
echo "   Project: $PROJECT_ID"
echo "   Region:  $REGION"
echo ""

# Enable required APIs
echo "📦 Enabling APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  --project="$PROJECT_ID" --quiet

# Copy agent + dictionary into backend build context
echo "📂 Preparing build context..."
cp -r data/dictionary backend/data/ 2>/dev/null || mkdir -p backend/data/dictionary

# Deploy directly from source (Cloud Build + Cloud Run)
echo "🚀 Building and deploying..."
cd backend
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=$REGION,GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION,GEMINI_TUNED_MODEL=projects/976647416990/locations/us-central1/endpoints/3643166357693923328" \
  --project="$PROJECT_ID"

# Get the URL
URL=$(gcloud run services describe "$SERVICE" \
  --region "$REGION" \
  --project="$PROJECT_ID" \
  --format='value(status.url)')

echo ""
echo "✅ Deployed!"
echo "   URL: $URL"
echo "   Health: $URL/health"
echo ""
echo "   Update mobile/src/services/api.js with:"
echo "   API_BASE = '$URL'"
echo "   WS_BASE  = '${URL/https/wss}'"
