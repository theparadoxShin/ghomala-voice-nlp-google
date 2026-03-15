"""
============================================================================
Script 04: Launch Fine-Tuning Job on Vertex AI (Gemini SFT)
============================================================================
Launches a Supervised Fine-Tuning (SFT) job for Gemini on Vertex AI.

SFT (Supervised Fine-Tuning):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  HOW:    You give the model PAIRS of (input → expected output)
  LIKE:   A student learning from a textbook with answers
  DATA:   "systemInstruction + contents" JSONL in GCS
  GOOD:   Teaching domain-specific knowledge (Ghomala' vocabulary)
  
  Supported base models for SFT:
   - gemini-2.5-flash       (fast, cost-effective — recommended)
   - gemini-2.5-flash-lite  (fastest, cheapest)
   - gemini-2.5-pro         (highest quality, slower)

  OUR STRATEGY:
  1. SFT with our MAFAND-MT + dictionary data (teach Ghomala' knowledge)
  2. Evaluate with Vertex AI Model Evaluation
  3. Deploy the tuned model for inference

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prerequisites:
  - gcloud CLI authenticated: `gcloud auth application-default login`
  - Training data uploaded to GCS (run 03_upload_to_gcs.py first)
  - Vertex AI API enabled: `gcloud services enable aiplatform.googleapis.com`

Usage:
  python 04_launch_fine_tuning.py --mode sft
  python 04_launch_fine_tuning.py --mode test --endpoint ENDPOINT_ID
============================================================================
"""

import argparse
import time
from datetime import datetime

from google.cloud import aiplatform

# ============================================================================
# CONFIGURATION
# ============================================================================
GCP_REGION = "us-central1"
BUCKET_NAME = "nam-sa-ghomala-training"
GCS_PREFIX = "fine-tuning/ghomala-v1"

# GCS URIs (from script 03)
TRAIN_GCS_URI = f"gs://{BUCKET_NAME}/{GCS_PREFIX}/train.jsonl"
VAL_GCS_URI = f"gs://{BUCKET_NAME}/{GCS_PREFIX}/val.jsonl"

# Base model for SFT
BASE_MODEL = "gemini-2.5-flash-001"


# ============================================================================
# SFT: Supervised Fine-Tuning via Vertex AI
# ============================================================================
def launch_sft_job(project_id: str):
    """
    Launch a Supervised Fine-Tuning job on Vertex AI for Gemini.

    What happens:
    1. Vertex AI reads your train.jsonl from GCS
    2. For each (systemInstruction, user, model) example, it adjusts weights
       so the model learns to respond like the "model" role in your data
    3. After training, you get a tuned model endpoint for inference

    Typical duration: 1-4 hours depending on data size.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = f"nam-sa-ghomala-sft-{timestamp}"

    print(f"\nLaunching Vertex AI SFT Job")
    print(f"   Display name:  {display_name}")
    print(f"   Base model:    {BASE_MODEL}")
    print(f"   Train data:    {TRAIN_GCS_URI}")
    print(f"   Validation:    {VAL_GCS_URI}")
    print(f"   Project:       {project_id}")
    print(f"   Region:        {GCP_REGION}")

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=GCP_REGION)

    try:
        sft_tuning_job = aiplatform.TuningJob.create(
            source_model=BASE_MODEL,
            training_dataset=TRAIN_GCS_URI,
            validation_dataset=VAL_GCS_URI,
            tuned_model_display_name=display_name,
            epochs=3,
            learning_rate_multiplier=1.0,
        )

        print(f"\n   ✅ SFT Job created!")
        print(f"   Job resource name: {sft_tuning_job.resource_name}")
        print(f"   Tuned model name:  {display_name}")

        return sft_tuning_job

    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        print(f"\n   Common fixes:")
        print(f"   1. Run: gcloud auth application-default login")
        print(f"   2. Enable API: gcloud services enable aiplatform.googleapis.com")
        print(f"   3. Make sure GCS URIs are accessible")
        print(f"   4. Check Vertex AI quotas in your project")
        raise


# ============================================================================
# Monitor job status
# ============================================================================
def monitor_job(tuning_job):
    """Poll the fine-tuning job status until completion."""
    print(f"\nMonitoring job: {tuning_job.resource_name}")
    print(f"   (This can take 1-4 hours)")
    print(f"   Console: https://console.cloud.google.com/vertex-ai/tuning\n")

    while True:
        tuning_job.refresh()
        state = tuning_job.state.name if hasattr(tuning_job.state, 'name') else str(tuning_job.state)

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"   [{timestamp}] State: {state}")

        if state == "JOB_STATE_SUCCEEDED":
            print(f"\n   ✅ Fine-tuning COMPLETE!")
            tuned_model = tuning_job.tuned_model
            if tuned_model:
                print(f"   Tuned model endpoint: {tuned_model.endpoint}")
                print(f"   Tuned model:          {tuned_model.model}")
            return tuning_job

        elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            print(f"\n   ❌ Job {state}!")
            if hasattr(tuning_job, 'error') and tuning_job.error:
                print(f"   Error: {tuning_job.error}")
            return tuning_job

        # Wait 60 seconds before checking again
        time.sleep(60)


# ============================================================================
# Test the fine-tuned model
# ============================================================================
def test_model(project_id: str, endpoint_id: str):
    """
    Test the fine-tuned model with Ghomala' prompts.
    Uses the google-genai SDK for inference.
    """
    print(f"\nTesting fine-tuned model via endpoint: {endpoint_id}")

    from google import genai

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=GCP_REGION,
    )

    test_prompts = [
        "Comment dit-on 'bonjour' en Ghomala' ?",
        "Traduis en Ghomala' : Le marché est ouvert aujourd'hui",
        "Parle-moi de la culture Bamiléké",
        "How do you say 'thank you' in Ghomala'?",
    ]

    system_instruction = (
        "Tu es NAM SA', un agent de préservation du Ghomala'. "
        "Tu te comportes comme un ancien bienveillant du village."
    )

    for prompt in test_prompts:
        print(f"\n   USER: {prompt}")

        try:
            response = client.models.generate_content(
                model=endpoint_id,
                contents=prompt,
                config={
                    "system_instruction": system_instruction,
                    "temperature": 0.7,
                    "max_output_tokens": 300,
                },
            )

            answer = response.text
            print(f"   MODEL: {answer[:200]}...")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n   ✅ Testing complete!")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Launch Vertex AI Gemini fine-tuning")
    parser.add_argument(
        "--mode",
        choices=["sft", "test"],
        default="sft",
        help="sft = Supervised Fine-Tuning, test = Test tuned model"
    )
    parser.add_argument(
        "--project",
        default=None,
        help="GCP Project ID (default: from gcloud config)"
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Tuned model endpoint for testing"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Wait for job completion"
    )

    args = parser.parse_args()

    # Get project ID
    project_id = args.project
    if not project_id:
        import subprocess
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True
        )
        project_id = result.stdout.strip()
        if not project_id:
            print("❌ No project ID. Use --project flag or `gcloud config set project PROJECT_ID`")
            return

    print("NAM SA' — Vertex AI Gemini Fine-Tuning Pipeline")
    print("=" * 60)
    print(f"   Project: {project_id}")

    if args.mode == "sft":
        print("\n   Mode: Supervised Fine-Tuning (SFT)")
        print("   The model learns from example conversations")
        tuning_job = launch_sft_job(project_id)

        if args.monitor:
            monitor_job(tuning_job)

    elif args.mode == "test":
        if not args.endpoint:
            print("❌ Please provide --endpoint for testing")
            return
        test_model(project_id, args.endpoint)

    print("\n" + "=" * 60)
    print("USEFUL COMMANDS:")
    print("=" * 60)
    print("   Monitor in console:")
    print("   → https://console.cloud.google.com/vertex-ai/tuning")
    print("")
    print("   List tuning jobs:")
    print("   → gcloud ai tuning-jobs list --region=us-central1")
    print("")
    print("   Describe a job:")
    print("   → gcloud ai tuning-jobs describe JOB_ID --region=us-central1")
    print("")
    print("   Test model:")
    print("   → python 04_launch_fine_tuning.py --mode test --endpoint ENDPOINT_ID")
    print("=" * 60)


if __name__ == "__main__":
    main()
