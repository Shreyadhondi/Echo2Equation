# backend/latex_parser/tokenization.py

"""
Tokenize Echo2Equation dataset splits using the MathT5 tokenizer
with additional LaTeX-specific tokens.

Purpose:
--------
This script reads train/validation/test JSONL files from:

    data/splits/

Each line should contain:
    {"spoken": "...", "latex": "..."}

It tokenizes:
    spoken text  -> encoder input
    LaTeX text   -> decoder labels

Then it saves tokenized JSONL files into:

    data/tokenized_splits/

Output files:
-------------
    data/tokenized_splits/tokenized_train.jsonl
    data/tokenized_splits/tokenized_val.jsonl
    data/tokenized_splits/tokenized_test.jsonl

Important:
----------
The same LaTeX token list must be used during:
1. tokenization
2. training
3. inference through the saved tokenizer

That is why we import the shared token helper from:

    backend/latex_parser/latex_tokens.py
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------
# Make project imports work when running this file directly
# ---------------------------------------------------------------------
# If we run:
#     python backend/latex_parser/tokenization.py
#
# Python may not automatically know the project root.
# So we add the project root folder to sys.path.
#
# File location:
#     backend/latex_parser/tokenization.py
#
# parents[0] = backend/latex_parser
# parents[1] = backend
# parents[2] = project root: Echo2Equation-git
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.latex_parser.latex_tokens import add_latex_tokens  # noqa: E402


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "jmeadows17/MathT5-base"

SPLITS_DIR = Path("data/splits")
TOKENIZED_DIR = Path("data/tokenized_splits")

MAX_LEN_INPUT = 256
MAX_LEN_LABEL = 256

TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load MathT5 tokenizer and add LaTeX tokens
# ---------------------------------------------------------------------

# The tokenizer converts text into token IDs that the model can understand.
# We start from the MathT5 tokenizer because our model is based on MathT5.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Add shared LaTeX tokens such as \frac, \sqrt, \sum, \int, \alpha, etc.
# This helps the tokenizer treat common LaTeX commands as meaningful units.
num_added_tokens = add_latex_tokens(tokenizer)

print(f"Added {num_added_tokens} LaTeX tokens to tokenizer.")
print(f"Tokenizer vocabulary size after adding tokens: {len(tokenizer)}")


# ---------------------------------------------------------------------
# Load dataset splits
# ---------------------------------------------------------------------

data_files = {}

for split_name in ["train", "val", "test"]:
    split_path = SPLITS_DIR / f"{split_name}.jsonl"

    if split_path.exists():
        data_files[split_name] = str(split_path)

if not data_files:
    raise FileNotFoundError(
        f"No split files found in {SPLITS_DIR}. "
        "Expected train.jsonl, val.jsonl, and/or test.jsonl."
    )

dataset = load_dataset("json", data_files=data_files)


# ---------------------------------------------------------------------
# Tokenization function
# ---------------------------------------------------------------------

def tokenize_example(example):
    """
    Tokenize one dataset example.

    Input:
        spoken text, for example:
            "x square plus y square equals z square"

    Target:
        LaTeX text, for example:
            "$x^2 + y^2 = z^2$"

    Output:
        input_ids       -> token IDs for spoken text
        attention_mask  -> tells model which tokens are real vs padding
        labels          -> token IDs for LaTeX target
        spoken          -> original spoken text, kept for debugging
        latex           -> original LaTeX text, kept for evaluation/audit
    """

    # Tokenize spoken text for the encoder side of T5.
    model_inputs = tokenizer(
        example["spoken"],
        max_length=MAX_LEN_INPUT,
        padding="max_length",
        truncation=True,
    )

    # Tokenize LaTeX target text for the decoder side of T5.
    labels = tokenizer(
        example["latex"],
        max_length=MAX_LEN_LABEL,
        padding="max_length",
        truncation=True,
    )

    # Keep original text for debugging, evaluation, and inspection.
    model_inputs["spoken"] = example["spoken"]
    model_inputs["latex"] = example["latex"]

    # HuggingFace Trainer expects decoder targets under the key "labels".
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# batched=False keeps memory usage low and processes one row at a time.
tokenized = dataset.map(tokenize_example, batched=False)


# ---------------------------------------------------------------------
# Save tokenized splits as JSONL
# ---------------------------------------------------------------------

def save_jsonl(ds_split, filename):
    """
    Save one tokenized dataset split to a JSONL file.

    Each row contains tokenized inputs plus original spoken/latex strings.
    """

    output_path = TOKENIZED_DIR / filename

    with output_path.open("w", encoding="utf-8") as f:
        for example in ds_split:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(ds_split)} rows → {output_path}")


if "train" in tokenized:
    save_jsonl(tokenized["train"], "tokenized_train.jsonl")

if "val" in tokenized:
    save_jsonl(tokenized["val"], "tokenized_val.jsonl")

if "test" in tokenized:
    save_jsonl(tokenized["test"], "tokenized_test.jsonl")


print("Tokenization complete using MathT5 tokenizer with LaTeX special tokens.")