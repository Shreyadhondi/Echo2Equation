# backend/latex_parser/matht5_train.py

"""
Train MathT5 for Echo2Equation.

Purpose:
--------
This script fine-tunes a MathT5 sequence-to-sequence model to convert
spoken-style mathematical text into LaTeX.

Example:
    spoken input : "x square plus y square equals z square"
    model output : "$x^2 + y^2 = z^2$"

Main MLOps features:
--------------------
1. Loads training settings from configs/config_matht5.yaml
2. Loads tokenized train/validation/test datasets
3. Uses the shared LaTeX token list from latex_tokens.py
4. Trains MathT5 using HuggingFace Seq2SeqTrainer
5. Logs parameters, losses, and custom metrics to MLflow
6. Saves the final model to models/matht5_model
7. Logs model artifacts, metrics, logs, and prediction samples to MLflow

Expected input files:
---------------------
data/tokenized_splits/tokenized_train.jsonl
data/tokenized_splits/tokenized_val.jsonl
data/tokenized_splits/tokenized_test.jsonl

Expected model output:
----------------------
models/matht5_model/
"""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# ---------------------------------------------------------------------
# Make project imports work when running this file directly
# ---------------------------------------------------------------------
# If we run:
#     python backend/latex_parser/matht5_train.py
#
# Python may not automatically know the project root.
# So we add the project root folder to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.latex_parser.latex_clean import clean_latex  # noqa: E402
from backend.latex_parser.latex_tokens import add_latex_tokens  # noqa: E402


# ---------------------------------------------------------------------
# 1. Configuration paths
# ---------------------------------------------------------------------

CONFIG_PATH = Path("configs/config_matht5.yaml")


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    Load YAML configuration.

    Keeping hyperparameters and paths in YAML makes the training script
    easier to reproduce and modify without editing Python code.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_yaml_config(CONFIG_PATH)

training_cfg = config["training"]
model_cfg = config["model"]
paths_cfg = config["paths"]
misc_cfg = config.get("misc", {})


# ---------------------------------------------------------------------
# 2. Reproducibility setup
# ---------------------------------------------------------------------

SEED = int(training_cfg.get("seed", 42))


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Full GPU reproducibility is not always guaranteed across hardware,
    but setting seeds makes runs more stable and explainable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ---------------------------------------------------------------------
# 3. Logging setup
# ---------------------------------------------------------------------

LOG_DIR = Path(paths_cfg["log_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "training.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w",
)

logger = logging.getLogger(__name__)


def log_and_print(message: str) -> None:
    """
    Print important messages to terminal and also store them in the log file.
    """
    print(message)
    logger.info(message)


# ---------------------------------------------------------------------
# 4. Read training settings from YAML
# ---------------------------------------------------------------------

MODEL_NAME = model_cfg["model_name"]
TOKENIZER_NAME = model_cfg["tokenizer_name"]

MAX_INPUT_LENGTH = int(model_cfg.get("max_input_length", 256))
MAX_TARGET_LENGTH = int(model_cfg.get("max_target_length", 256))

EPOCHS = int(training_cfg["epochs"])
BATCH_SIZE = int(training_cfg.get("batch_size", 16))
LEARNING_RATE = float(training_cfg["learning_rate"])
WEIGHT_DECAY = float(training_cfg["weight_decay"])
LOGGING_STEPS = int(training_cfg.get("logging_steps", 10))

SAVE_STRATEGY = training_cfg.get("save_strategy", "epoch")
EVAL_STRATEGY = training_cfg.get("evaluation_strategy", "epoch")
REPORT_TO = training_cfg.get("report_to", "mlflow")

TOKENIZED_DATASET_DIR = Path(paths_cfg["tokenized_dataset"])
MODEL_OUTPUT_DIR = Path(paths_cfg.get("model_output_dir", "models/matht5_model"))

MLFLOW_TRACKING_URI = paths_cfg.get("mlflow_tracking_uri", "mlruns")
MLFLOW_EXPERIMENT_NAME = paths_cfg.get(
    "mlflow_experiment_name", "Echo2Equation-MathT5"
)

USE_CUDA = bool(misc_cfg.get("use_cuda", True))
DEVICE = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# 5. Load tokenizer, add LaTeX tokens, and load model
# ---------------------------------------------------------------------

log_and_print(f"Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Add the same shared LaTeX tokens used during tokenization.
# This is important because tokenized files were created with these added tokens.
num_added_tokens = add_latex_tokens(tokenizer)

log_and_print(f"Added {num_added_tokens} LaTeX tokens to tokenizer.")
log_and_print(f"Tokenizer vocabulary size: {len(tokenizer)}")

log_and_print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# The model embedding table must be resized after adding new tokenizer tokens.
# Without this, the model cannot learn the new token IDs.
model.resize_token_embeddings(len(tokenizer))

model.to(DEVICE)

log_and_print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------
# 6. Load tokenized dataset
# ---------------------------------------------------------------------

train_file = TOKENIZED_DATASET_DIR / "tokenized_train.jsonl"
val_file = TOKENIZED_DATASET_DIR / "tokenized_val.jsonl"
test_file = TOKENIZED_DATASET_DIR / "tokenized_test.jsonl"

for file_path in [train_file, val_file, test_file]:
    if not file_path.exists():
        raise FileNotFoundError(f"Tokenized dataset file not found: {file_path}")

log_and_print("Loading tokenized dataset...")

dataset: DatasetDict = load_dataset(
    "json",
    data_files={
        "train": str(train_file),
        "val": str(val_file),
        "test": str(test_file),
    },
)

log_and_print(
    "Dataset loaded: "
    f"train={len(dataset['train'])}, "
    f"val={len(dataset['val'])}, "
    f"test={len(dataset['test'])}"
)


# ---------------------------------------------------------------------
# 7. Prepare labels for loss calculation
# ---------------------------------------------------------------------

def prepare_labels(example: dict[str, Any]) -> dict[str, Any]:
    """
    Replace padding token IDs in labels with -100.

    HuggingFace ignores label value -100 while computing loss.
    This prevents padded positions from affecting the training loss.
    """
    labels = example["labels"]

    example["labels"] = [
        -100 if token_id == tokenizer.pad_token_id else token_id
        for token_id in labels
    ]

    return example


dataset = dataset.map(prepare_labels)


# ---------------------------------------------------------------------
# 8. Metric helper functions
# ---------------------------------------------------------------------

def sanitize_token_ids(token_ids) -> np.ndarray:
    """
    Clean token IDs before decoding.

    Why this is needed:
    -------------------
    During evaluation, HuggingFace may return padded or invalid token IDs.
    tokenizer.batch_decode() cannot handle negative IDs or IDs larger than
    the tokenizer vocabulary size. Invalid IDs are replaced with pad_token_id.

    This prevents errors like:
        OverflowError: out of range integral type conversion attempted
    """
    token_ids = np.asarray(token_ids)

    # If predictions accidentally come as logits, convert logits to token IDs.
    if token_ids.ndim == 3:
        token_ids = np.argmax(token_ids, axis=-1)

    token_ids = np.where(
        (token_ids < 0) | (token_ids >= len(tokenizer)),
        tokenizer.pad_token_id,
        token_ids,
    )

    return token_ids.astype(np.int64)


def normalize_latex_for_metric(text: str) -> str:
    """
    Normalize LaTeX before comparison.

    This reduces unfair mismatches caused by spacing differences.
    """
    if text is None:
        return ""

    text = clean_latex(str(text), keep_dollars=True)
    text = text.strip()

    # Remove all whitespace for stricter symbolic comparison.
    text = re.sub(r"\s+", "", text)

    return text


def levenshtein_distance(a: str, b: str) -> int:
    """
    Compute edit distance between two strings.

    Edit distance counts how many insertions, deletions, or substitutions
    are needed to convert one string into another.
    """
    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous_row = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current_row = [i]

        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (char_a != char_b)

            current_row.append(min(insert_cost, delete_cost, replace_cost))

        previous_row = current_row

    return previous_row[-1]


def has_balanced_braces(text: str) -> bool:
    """
    Simple validity check for LaTeX-like strings.

    This does not fully parse LaTeX. It only checks whether braces are balanced.
    It helps identify obvious generation mistakes like:
        "\\sqrt{x+1}}"
    """
    balance = 0

    for char in text:
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1

        if balance < 0:
            return False

    return balance == 0


def is_likely_valid_latex(text: str) -> bool:
    """
    Lightweight validity check for generated LaTeX.

    A full LaTeX compiler check would be expensive here.
    This basic check catches common brace mistakes.
    """
    if not text:
        return False

    return has_balanced_braces(text)


def compute_metrics(eval_preds) -> dict[str, float]:
    """
    Compute custom validation/test metrics.

    Metrics:
    --------
    exact_match_accuracy:
        Fraction of predictions exactly matching target LaTeX after cleaning.

    avg_edit_distance:
        Average character-level edit distance.

    avg_normalized_edit_distance:
        Edit distance divided by target length.

    valid_latex_rate:
        Fraction of predictions with balanced braces.
    """
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Clean predictions and labels before decoding.
    predictions = sanitize_token_ids(predictions)
    labels = sanitize_token_ids(labels)

    decoded_preds = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    exact_matches = 0
    edit_distances = []
    normalized_edit_distances = []
    valid_latex_count = 0

    for pred, label in zip(decoded_preds, decoded_labels):
        pred_norm = normalize_latex_for_metric(pred)
        label_norm = normalize_latex_for_metric(label)

        if pred_norm == label_norm:
            exact_matches += 1

        distance = levenshtein_distance(pred_norm, label_norm)
        edit_distances.append(distance)

        denominator = max(len(label_norm), 1)
        normalized_edit_distances.append(distance / denominator)

        if is_likely_valid_latex(pred_norm):
            valid_latex_count += 1

    total = max(len(decoded_labels), 1)

    return {
        "exact_match_accuracy": exact_matches / total,
        "avg_edit_distance": float(np.mean(edit_distances)),
        "avg_normalized_edit_distance": float(np.mean(normalized_edit_distances)),
        "valid_latex_rate": valid_latex_count / total,
    }


# ---------------------------------------------------------------------
# 9. Data collator
# ---------------------------------------------------------------------

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)


# ---------------------------------------------------------------------
# 10. MLflow setup
# ---------------------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

log_and_print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
log_and_print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")


# ---------------------------------------------------------------------
# 11. Training arguments
# ---------------------------------------------------------------------

PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = max(1, BATCH_SIZE // PER_DEVICE_BATCH_SIZE)

training_args = Seq2SeqTrainingArguments(
    output_dir="models/tmp_matht5_checkpoints",

    evaluation_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=EPOCHS,

    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=4,

    logging_dir=str(LOG_DIR / "tensorboard"),
    logging_steps=LOGGING_STEPS,

    report_to=REPORT_TO,

    load_best_model_at_end=True,
    save_total_limit=2,

    remove_unused_columns=True,

    fp16=(DEVICE == "cuda"),

    seed=SEED,
)


# ---------------------------------------------------------------------
# 12. Trainer
# ---------------------------------------------------------------------

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ---------------------------------------------------------------------
# 13. Save prediction samples
# ---------------------------------------------------------------------

def save_prediction_samples(
    prediction_output,
    output_path: Path,
    max_samples: int = 100,
) -> None:
    """
    Save a small set of model predictions for inspection.

    The evaluator can inspect:
        spoken input
        target LaTeX
        predicted LaTeX
        exact-match result
    """
    predictions = prediction_output.predictions
    labels = prediction_output.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = sanitize_token_ids(predictions)
    labels = sanitize_token_ids(labels)

    decoded_preds = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_rows = min(max_samples, len(decoded_preds))

    with output_path.open("w", encoding="utf-8") as f:
        for i in range(num_rows):
            spoken = dataset["test"][i].get("spoken", "")

            pred_clean = clean_latex(decoded_preds[i], keep_dollars=True)
            label_clean = clean_latex(decoded_labels[i], keep_dollars=True)

            row = {
                "spoken": spoken,
                "target_latex": label_clean,
                "predicted_latex": pred_clean,
                "exact_match": normalize_latex_for_metric(pred_clean)
                == normalize_latex_for_metric(label_clean),
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# 14. Train, evaluate, log, save
# ---------------------------------------------------------------------

with mlflow.start_run() as run:
    run_id = run.info.run_id

    log_and_print(f"Started MLflow run: {run_id}")

    mlflow.log_params(
        {
            "model_name": MODEL_NAME,
            "tokenizer_name": TOKENIZER_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": PER_DEVICE_BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_input_length": MAX_INPUT_LENGTH,
            "max_target_length": MAX_TARGET_LENGTH,
            "seed": SEED,
            "device": DEVICE,
            "train_samples": len(dataset["train"]),
            "val_samples": len(dataset["val"]),
            "test_samples": len(dataset["test"]),
            "latex_tokens_added": num_added_tokens,
            "tokenizer_vocab_size": len(tokenizer),
        }
    )

    log_and_print(f"Training for {EPOCHS} epoch(s)...")

    train_result = trainer.train()

    log_and_print("Training completed.")

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    mlflow.log_metrics({f"final_train_{k}": v for k, v in train_metrics.items()})

    log_and_print(f"Training metrics: {train_metrics}")

    log_and_print("Evaluating on validation set...")
    val_metrics = trainer.evaluate(
        eval_dataset=dataset["val"],
        metric_key_prefix="val",
    )

    trainer.log_metrics("val", val_metrics)
    trainer.save_metrics("val", val_metrics)
    mlflow.log_metrics(val_metrics)

    log_and_print(f"Validation metrics: {val_metrics}")

    log_and_print("Evaluating on test set...")
    test_predictions = trainer.predict(
        test_dataset=dataset["test"],
        metric_key_prefix="test",
    )

    test_metrics = test_predictions.metrics

    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    mlflow.log_metrics(test_metrics)

    log_and_print(f"Test metrics: {test_metrics}")

    prediction_sample_path = LOG_DIR / "test_prediction_samples.jsonl"
    save_prediction_samples(
        prediction_output=test_predictions,
        output_path=prediction_sample_path,
        max_samples=100,
    )

    mlflow.log_artifact(str(prediction_sample_path), artifact_path="predictions")

    metrics_summary = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    metrics_summary_path = LOG_DIR / "metrics_summary.json"

    with metrics_summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    mlflow.log_artifact(str(metrics_summary_path), artifact_path="metrics")

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for item in MODEL_OUTPUT_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    log_and_print(f"Saved trained model to: {MODEL_OUTPUT_DIR}")

    mlflow.log_artifacts(str(MODEL_OUTPUT_DIR), artifact_path="model")

    if LOG_FILE.exists():
        mlflow.log_artifact(str(LOG_FILE), artifact_path="logs")

    log_and_print("Logged model, metrics, predictions, and logs to MLflow.")

log_and_print("MLflow run ended successfully.")