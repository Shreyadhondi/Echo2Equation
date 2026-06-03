# scripts/merge_datasets.py

"""
Clean and shuffle the final Echo2Equation dataset.

Purpose:
--------
Earlier, this project had multiple source JSONL files such as:
    data/all_formulas.jsonl
    data/bigdata.jsonl
    data/mathspeech_trimmed.jsonl

Those files have now been merged into one final dataset:

    data/full_dataset.jsonl

This script now works only on data/full_dataset.jsonl.

It performs:
1. Load full_dataset.jsonl
2. Remove duplicate (spoken, latex) pairs
3. Skip invalid rows
4. Shuffle the cleaned dataset
5. Save the cleaned dataset back to full_dataset.jsonl

Each JSONL line should look like:
    {"spoken": "a plus b", "latex": "$a + b$"}
"""

import json
import random
from pathlib import Path


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SHUFFLE = True
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "full_dataset.jsonl"


# ---------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """
    Load a JSONL file into a list of dictionaries.

    Empty lines are skipped.
    Invalid JSON lines raise a clear error with the line number.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {e}"
                )

    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    """
    Save a list of dictionaries into a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Main cleaning logic
# ---------------------------------------------------------------------

def main() -> None:
    print(f"Loading dataset from: {DATASET_PATH}")

    rows = load_jsonl(DATASET_PATH)

    print(f"Rows before cleaning: {len(rows):,}")

    seen = set()
    cleaned_rows = []
    duplicate_count = 0
    invalid_count = 0

    for row in rows:
        spoken = str(row.get("spoken", "")).strip()
        latex = str(row.get("latex", "")).strip()

        if not spoken or not latex:
            invalid_count += 1
            continue

        # Lowercase spoken text for duplicate detection.
        # Keep LaTeX case-sensitive because A and a are different in math.
        key = (spoken.lower(), latex)

        if key in seen:
            duplicate_count += 1
            continue

        seen.add(key)
        cleaned_rows.append(
            {
                "spoken": spoken,
                "latex": latex,
            }
        )

    if SHUFFLE:
        random.seed(RANDOM_SEED)
        random.shuffle(cleaned_rows)

    save_jsonl(cleaned_rows, DATASET_PATH)

    print("\nDataset cleaning complete.")
    print(f"Saved to: {DATASET_PATH}")
    print(f"Final rows: {len(cleaned_rows):,}")
    print(f"Duplicate rows removed: {duplicate_count:,}")
    print(f"Invalid rows skipped: {invalid_count:,}")
    print(f"Shuffled: {SHUFFLE} with seed={RANDOM_SEED}")


if __name__ == "__main__":
    main()