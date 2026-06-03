# backend/latex_parser/test_matht5.py
import random, sys
from pathlib import Path

# --- make project root importable when running as a file ---
ROOT = Path(__file__).resolve().parents[2]  # <repo root>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging as hf_logging

# your cleaner (now supports keep_dollars=True)
from backend.latex_parser.latex_clean import clean_latex

hf_logging.set_verbosity_error()  # silence HF info/warnings

# ---------- resolve model directory ----------
BASE_MODEL_DIR = Path("models/matht5_model")

def resolve_model_dir(base: Path) -> Path:
    # current layout: files directly in models/matht5_model
    if (base / "config.json").exists() and (base / "model.safetensors").exists():
        return base
    # old layout: pick most recent subfolder
    subs = [p for p in base.glob("*") if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No model found in {base}")
    return max(subs, key=lambda p: p.stat().st_mtime)

MODEL_DIR = resolve_model_dir(BASE_MODEL_DIR)
print(f"📂 Using model from: {MODEL_DIR}")

# ---------- load model/tokenizer ----------
print("📥 Loading MathT5 model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
print(f"🧠 Device: {device}")

# ---------- find test file ----------
CANDIDATES = [
    Path("data/tokenized_splits/tokenized_test.jsonl"),
    Path("data/tokenized_splits/tokenized_test.json"),
    Path("data/splits/test.jsonl"),
    Path("data/tokenized_data/test.json"),
    Path("data/test.json"),
]
TEST_PATH = next((p for p in CANDIDATES if p.exists()), None)
if TEST_PATH is None:
    raise FileNotFoundError("Could not find a test file. Checked:\n" + "\n".join(map(str, CANDIDATES)))

ds = load_dataset("json", data_files={"test": str(TEST_PATH)})["test"]
print(f"✅ Loaded {len(ds)} test samples from: {TEST_PATH}")

# ---------- config ----------
NUM_SAMPLES = 20
BEAMS = 5
SEED = 0
random.seed(SEED)

indices = random.sample(range(len(ds)), min(NUM_SAMPLES, len(ds)))

print("\n🔍 Sample Predictions:")
with torch.no_grad():
    for idx in indices:
        spoken = ds[idx].get("spoken") or ds[idx].get("text") or ""
        gt = (ds[idx].get("latex") or "").strip()
        if not spoken:
            continue

        enc = tokenizer(spoken, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_length=128,
            num_beams=BEAMS,
            early_stopping=True,
        )
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # keep_dollars=True -> re-wrap cleaned output in $...$
        cleaned = clean_latex(pred, keep_dollars=True)

        print("\n🗣 Spoken:      ", spoken)
        print("✅ Ground Truth:", gt)
        print("🤖 Predicted:   ", pred)
        print("🧼 Cleaned:     ", cleaned)
