# backend/latex_parser/interactive_matht5.py
import sys, time
from pathlib import Path

# make repo root importable when running as a file
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging as hf_logging
from backend.latex_parser.latex_clean import clean_latex

hf_logging.set_verbosity_error()  # hush transformers logs

# -------- resolve model directory (works with your current layout) --------
BASE = Path("models/matht5_model")
def resolve_model_dir(base: Path) -> Path:
    if (base / "config.json").exists() and (base / "model.safetensors").exists():
        return base
    subs = [p for p in base.glob("*") if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No model found in {base}")
    return max(subs, key=lambda p: p.stat().st_mtime)

MODEL_DIR = resolve_model_dir(BASE)

# -------- load model/tokenizer --------
print(f"📂 Model: {MODEL_DIR}")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
print(f"🧠 Device: {device}")

# -------- generation config --------
MAX_LEN = 128
BEAMS = 5

def render_latex(latex: str, save_path: Path | None = None):
    """Render LaTeX string with Matplotlib; show and optionally save."""
    plt.figure(figsize=(6, 1.6))
    plt.axis("off")
    plt.text(0.5, 0.5, latex, fontsize=22, ha="center", va="center")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"💾 Saved preview → {save_path}")
    plt.show()

print("\nType spoken-style math and press Enter.")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    spoken = input("🗣  Spoken > ").strip()
    if not spoken:
        continue
    if spoken.lower() in {"exit", "quit"}:
        print("bye!")
        break

    # tokenize & generate
    enc = tok(spoken, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=MAX_LEN,
            num_beams=BEAMS,
            early_stopping=True,
        )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    pred_raw = tok.decode(out[0], skip_special_tokens=True).strip()
    pred_clean = clean_latex(pred_raw, keep_dollars=True)

    print(f"🤖 Raw     : {pred_raw}")
    print(f"🧼 Cleaned : {pred_clean}")
    print(f"⏱️  Latency : {dt_ms:.1f} ms")

    # render (and save a copy)
    render_latex(pred_clean, save_path=Path("reports/last_preview.png"))
