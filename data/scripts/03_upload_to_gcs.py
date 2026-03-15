"""
============================================================================
Script 03: Upload training data to Google Cloud Storage (GCS)
============================================================================
Uploads train.jsonl and val.jsonl to your GCS bucket for Vertex AI
Gemini supervised fine-tuning.

Prerequisites:
  - gcloud CLI installed and authenticated: `gcloud auth login`
  - Application Default Credentials set: `gcloud auth application-default login`
  - Or run from a GCE/Cloud Shell environment with proper IAM

Usage:
  python 03_upload_to_gcs.py
  python 03_upload_to_gcs.py --bucket my-custom-bucket --project my-project
============================================================================
"""

import argparse
import json
from pathlib import Path

from google.cloud import storage

# ============================================================================
# CONFIGURATION — Modify these values for your GCP project
# ============================================================================
DEFAULT_BUCKET_NAME = "nam-sa-ghomala-training"
GCS_PREFIX = "fine-tuning/ghomala-v1"

# Local paths
PROCESSED_DIR = Path(__file__).parent.parent / "processed"
TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VAL_FILE = PROCESSED_DIR / "val.jsonl"


def create_bucket_if_needed(client: storage.Client, bucket_name: str, location: str):
    """Create the GCS bucket if it doesn't exist."""
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"   ✅ Bucket '{bucket_name}' exists (location: {bucket.location})")
    except Exception:
        print(f"   Creating bucket '{bucket_name}' in {location}...")
        bucket = client.bucket(bucket_name)
        bucket.storage_class = "STANDARD"
        bucket = client.create_bucket(bucket, location=location)
        print(f"   ✅ Bucket created in {bucket.location}!")
    return bucket


def upload_file(bucket, local_path: Path, gcs_key: str):
    """Upload a single file to GCS with progress info."""
    file_size = local_path.stat().st_size / 1024
    gcs_uri = f"gs://{bucket.name}/{gcs_key}"
    print(f"   Uploading {local_path.name} ({file_size:.1f} KB) → {gcs_uri}")

    blob = bucket.blob(gcs_key)
    blob.upload_from_filename(str(local_path), content_type="application/jsonl")
    print(f"   ✅ Uploaded!")
    return gcs_uri


def validate_jsonl_quick(file_path: Path):
    """Quick validation that the JSONL is well-formed Vertex AI format."""
    print(f"   Validating {file_path.name}...")
    line_count = 0
    errors = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                assert "contents" in obj, "Missing 'contents'"
                assert len(obj["contents"]) >= 2, "Need at least user + model"
                assert obj["contents"][0]["role"] == "user", "First content must be user"
                assert obj["contents"][-1]["role"] == "model", "Last content must be model"
                line_count += 1
            except Exception as e:
                print(f"      ❌ Line {i}: {e}")
                errors += 1

    if errors == 0:
        print(f"   ✅ Validation passed: {line_count} valid conversations")
    else:
        print(f"   ⚠️  {errors} errors found out of {line_count + errors} lines")

    return errors == 0


def main():
    parser = argparse.ArgumentParser(description="Upload training data to GCS")
    parser.add_argument(
        "--bucket", default=DEFAULT_BUCKET_NAME,
        help=f"GCS bucket name (default: {DEFAULT_BUCKET_NAME})"
    )
    parser.add_argument(
        "--project", default=None,
        help="GCP project ID (default: from gcloud config)"
    )
    parser.add_argument(
        "--location", default="us-central1",
        help="Bucket location (default: us-central1)"
    )

    args = parser.parse_args()

    print("NAM SA' — GCS Upload Pipeline")
    print("=" * 60)

    # Check files exist
    if not TRAIN_FILE.exists():
        print(f"❌ {TRAIN_FILE} not found. Run 02_transform_to_jsonl.py first!")
        return

    # Validate before uploading
    print("\nValidating JSONL files...")
    if not validate_jsonl_quick(TRAIN_FILE):
        print("❌ Training file has errors. Fix them before uploading.")
        return

    if VAL_FILE.exists():
        validate_jsonl_quick(VAL_FILE)

    # Initialize GCS client
    print("\nConnecting to Google Cloud Storage...")
    client = storage.Client(project=args.project)
    print(f"   Project: {client.project}")

    # Create bucket
    bucket = create_bucket_if_needed(client, args.bucket, args.location)

    # Upload files
    print("\nUploading files...")
    train_gcs_uri = upload_file(
        bucket, TRAIN_FILE,
        f"{GCS_PREFIX}/train.jsonl"
    )

    val_gcs_uri = None
    if VAL_FILE.exists():
        val_gcs_uri = upload_file(
            bucket, VAL_FILE,
            f"{GCS_PREFIX}/val.jsonl"
        )

    # Summary
    print("\n" + "=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)
    print(f"   Bucket:    {args.bucket}")
    print(f"   Location:  {args.location}")
    print(f"   Train URI: {train_gcs_uri}")
    if val_gcs_uri:
        print(f"   Val URI:   {val_gcs_uri}")

    print(f"\n   Save these URIs for the fine-tuning step:")
    print(f"      TRAIN_GCS_URI = \"{train_gcs_uri}\"")
    if val_gcs_uri:
        print(f"      VAL_GCS_URI   = \"{val_gcs_uri}\"")

    print(f"\n   Next step: python 04_launch_fine_tuning.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
