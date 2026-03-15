# GCP Setup for Vertex AI Fine-Tuning

## Step 1: Enable APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  --project=YOUR_PROJECT_ID
```

## Step 2: Authenticate

**Option A — API Key (simple, for dev):**
1. Go to Google Cloud Console → APIs & Services → Credentials
2. Create an API key
3. Set `GOOGLE_API_KEY=your-key` in `backend/nam_sa_agent/.env`

**Option B — Vertex AI (recommended for production):**
1. `gcloud auth application-default login`
2. Set in `.env`:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=TRUE
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   ```

## Step 3: Create a GCS Bucket for Training Data

```bash
gsutil mb -l us-central1 gs://nam-sa-ghomala-training/
```

Or let the upload script create it automatically:
```bash
python data/scripts/03_upload_to_gcs.py --bucket nam-sa-ghomala-training
```

## Step 4: Grant Permissions

The service account running the fine-tuning job needs:
- `roles/aiplatform.user` — to create tuning jobs
- `roles/storage.objectViewer` — to read training data from GCS

```bash
# If using your user account (dev):
gcloud auth application-default login

# If using a service account (prod):
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:SA_EMAIL" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:SA_EMAIL" \
  --role="roles/storage.objectViewer"
```

## Step 5: Launch Fine-Tuning

```bash
cd data/scripts
python 04_launch_fine_tuning.py --mode sft --project YOUR_PROJECT_ID
```

## Step 6: Monitor

- **Console:** https://console.cloud.google.com/vertex-ai/tuning
- **CLI:** `gcloud ai tuning-jobs list --region=us-central1`

## Step 7: Deploy Backend

```bash
./deploy.sh YOUR_PROJECT_ID
```

This deploys the backend to Cloud Run and prints the URL to set in the mobile app.
