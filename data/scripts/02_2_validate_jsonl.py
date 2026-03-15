"""
============================================================================
Script 02_2: Validate JSONL files for Vertex AI Gemini fine-tuning
============================================================================
Validates that train.jsonl and val.jsonl conform to Vertex AI's
Gemini supervised fine-tuning format before uploading to GCS.

Checks:
  - Valid JSONL (each line is valid JSON)
  - Correct schema: systemInstruction + contents
  - systemInstruction has role "system" + parts with text
  - contents array has at least 2 entries (user + model)
  - Role alternation: user → model → user → model ...
  - Last message has "model" role (not "assistant" — Vertex AI uses "model")
  - No empty text content
  - Sample count within bounds (Vertex AI: min 10, max 10,000,000)

Usage:
  python 02_2_validate_jsonl.py
  python 02_2_validate_jsonl.py --file train.jsonl
============================================================================
"""

import argparse
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
PROCESSED_DIR = Path(__file__).parent.parent / "processed"

# Vertex AI SFT limits
MIN_SAMPLES = 10
MAX_SAMPLES = 10_000_000

# Valid roles
VALID_CONTENT_ROLES = ["user", "model"]
VALID_SYSTEM_ROLE = "system"


# ============================================================================
# VALIDATION
# ============================================================================
def validate_sample(sample: dict, index: int) -> list:
    """
    Validate a single JSONL sample against Vertex AI Gemini SFT format.
    Returns list of error strings (empty = valid).

    Expected format:
    {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": "..."}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": "..."}]},
            {"role": "model", "parts": [{"text": "..."}]}
        ]
    }
    """
    errors = []

    # --- systemInstruction (optional but expected) ---
    if "systemInstruction" in sample:
        si = sample["systemInstruction"]

        if not isinstance(si, dict):
            errors.append(f"Sample {index}: systemInstruction must be an object")
        else:
            role = si.get("role")
            if role != VALID_SYSTEM_ROLE:
                errors.append(
                    f"Sample {index}: systemInstruction.role must be 'system', got '{role}'"
                )

            parts = si.get("parts")
            if not isinstance(parts, list) or len(parts) == 0:
                errors.append(
                    f"Sample {index}: systemInstruction.parts must be a non-empty list"
                )
            else:
                for pi, part in enumerate(parts):
                    if not isinstance(part, dict) or "text" not in part:
                        errors.append(
                            f"Sample {index}: systemInstruction.parts[{pi}] must have 'text'"
                        )
                    elif not part["text"].strip():
                        errors.append(
                            f"Sample {index}: systemInstruction.parts[{pi}].text is empty"
                        )

    # --- contents (required) ---
    if "contents" not in sample:
        errors.append(f"Sample {index}: missing required field 'contents'")
        return errors

    contents = sample["contents"]

    if not isinstance(contents, list):
        errors.append(f"Sample {index}: 'contents' must be a list")
        return errors

    if len(contents) < 2:
        errors.append(
            f"Sample {index}: 'contents' must have at least 2 items (user + model), got {len(contents)}"
        )
        return errors

    # Check role alternation and content
    for ci, content in enumerate(contents):
        if not isinstance(content, dict):
            errors.append(f"Sample {index}: contents[{ci}] must be an object")
            continue

        role = content.get("role")
        if role not in VALID_CONTENT_ROLES:
            errors.append(
                f"Sample {index}: contents[{ci}].role must be 'user' or 'model', got '{role}'"
            )

        # Check alternation: even indices → user, odd indices → model
        expected_role = "user" if ci % 2 == 0 else "model"
        if role != expected_role:
            errors.append(
                f"Sample {index}: contents[{ci}] expected role '{expected_role}', got '{role}'"
            )

        # Check parts
        parts = content.get("parts")
        if not isinstance(parts, list) or len(parts) == 0:
            errors.append(f"Sample {index}: contents[{ci}].parts must be a non-empty list")
        else:
            for pi, part in enumerate(parts):
                if not isinstance(part, dict) or "text" not in part:
                    errors.append(
                        f"Sample {index}: contents[{ci}].parts[{pi}] must have 'text'"
                    )
                elif not part["text"].strip():
                    errors.append(
                        f"Sample {index}: contents[{ci}].parts[{pi}].text is empty"
                    )

    # Last message must be model role
    if contents and contents[-1].get("role") != "model":
        errors.append(
            f"Sample {index}: last content must have role 'model', got '{contents[-1].get('role')}'"
        )

    return errors


def validate_file(file_path: str) -> tuple:
    """
    Validate an entire JSONL file.
    Returns (num_valid, num_errors, error_messages).
    """
    print(f"\nValidating: {file_path}")

    errors_all = []
    num_valid = 0
    line_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            line_count += 1

            # Parse JSON
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                errors_all.append(f"Line {line_number}: Invalid JSON — {e}")
                continue

            # Validate structure
            sample_errors = validate_sample(sample, line_number)

            if sample_errors:
                errors_all.extend(sample_errors)
            else:
                num_valid += 1

    print(f"   Total lines:   {line_count}")
    print(f"   Valid samples: {num_valid}")
    print(f"   Errors:        {len(errors_all)}")

    # Check bounds
    if line_count < MIN_SAMPLES:
        errors_all.append(
            f"Too few samples: {line_count} < minimum {MIN_SAMPLES} for Vertex AI SFT"
        )
    if line_count > MAX_SAMPLES:
        errors_all.append(
            f"Too many samples: {line_count} > maximum {MAX_SAMPLES} for Vertex AI SFT"
        )

    return num_valid, len(errors_all), errors_all


def main():
    parser = argparse.ArgumentParser(
        description="Validate JSONL files for Vertex AI Gemini SFT",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Specific JSONL file to validate. If omitted, validates train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=10,
        help="Max number of errors to display (default: 10)",
    )

    args = parser.parse_args()

    print("NAM SA' — JSONL Validation for Vertex AI Gemini SFT")
    print("=" * 60)
    print("   Expected format: systemInstruction + contents (role: user/model)")
    print()

    files_to_validate = []

    if args.file:
        files_to_validate.append(args.file)
    else:
        train_path = PROCESSED_DIR / "train.jsonl"
        val_path = PROCESSED_DIR / "val.jsonl"

        if train_path.exists():
            files_to_validate.append(str(train_path))
        else:
            print(f"   ⚠️  {train_path} not found")

        if val_path.exists():
            files_to_validate.append(str(val_path))
        else:
            print(f"   ⚠️  {val_path} not found")

    if not files_to_validate:
        print("❌ No files to validate. Run 02_transform_to_jsonl.py first!")
        return

    all_passed = True

    for file_path in files_to_validate:
        num_valid, num_errors, error_msgs = validate_file(file_path)

        if num_errors > 0:
            all_passed = False
            print(f"\n   ❌ ERRORS FOUND:")
            for msg in error_msgs[:args.show_errors]:
                print(f"      • {msg}")
            if num_errors > args.show_errors:
                print(f"      ... and {num_errors - args.show_errors} more errors")
        else:
            print(f"   ✅ All {num_valid} samples valid for Vertex AI Gemini SFT")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL FILES PASSED VALIDATION")
        print(f"   Ready for upload to GCS → python 03_upload_to_gcs.py")
    else:
        print("❌ SOME FILES FAILED VALIDATION")
        print("   Fix the errors and re-run 02_transform_to_jsonl.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
