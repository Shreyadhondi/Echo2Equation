import random, sys, html
from pathlib import Path

# ensure package import works when run as a file
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging as hf_logging
from backend.latex_parser.latex_clean import clean_latex

hf_logging.set_verbosity_error()

BASE = Path("models/matht5_model")

def resolve_model_dir(base: Path) -> Path:
    if (base / "config.json").exists() and (base / "model.safetensors").exists():
        return base
    subs = [p for p in base.glob("*") if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No model found in {base}")
    return max(subs, key=lambda p: p.stat().st_mtime)

CANDIDATES = [
    Path("data/tokenized_splits/tokenized_test.jsonl"),
    Path("data/tokenized_splits/tokenized_test.json"),
    Path("data/splits/test.jsonl"),
    Path("data/tokenized_data/test.json"),
    Path("data/test.json"),
]

def find_test_path():
    for p in CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find a test file.\nChecked:\n" + "\n".join(map(str, CANDIDATES)))

def norm_for_match(s: str) -> str:
    """Normalize for fair string comparison (no dollars, collapse spaces)."""
    if not s: return ""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    return " ".join(s.split())

def main(samples=200, beams=5, seed=0, outfile="reports/matht5_preview.html"):
    model_dir = resolve_model_dir(BASE)
    print("📂 Model:", model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print("🧠 Device:", device)

    test_path = find_test_path()
    ds = load_dataset("json", data_files={"test": str(test_path)})["test"]
    print(f"✅ Loaded {len(ds)} samples from: {test_path}")

    n = min(samples, len(ds))
    random.seed(seed)
    idxs = random.sample(range(len(ds)), n)

    rows = []
    em_raw = em_clean = 0

    with torch.no_grad():
        for i in idxs:
            spoken = ds[i].get("spoken") or ds[i].get("text") or ""
            gt = (ds[i].get("latex") or "").strip()
            enc = tok(spoken, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.generate(**enc, max_length=128, num_beams=beams, early_stopping=True)
            pred_raw = tok.decode(out[0], skip_special_tokens=True).strip()
            pred_clean = clean_latex(pred_raw, keep_dollars=True)

            # metrics (normalize both sides without dollars)
            if norm_for_match(pred_raw) == norm_for_match(gt):
                em_raw += 1
            if norm_for_match(pred_clean) == norm_for_match(gt):
                em_clean += 1

            rows.append((spoken, gt, pred_raw, pred_clean))

    em_raw_pct = 100.0 * em_raw / n
    em_clean_pct = 100.0 * em_clean / n
    print(f"📊 Exact match (raw):   {em_raw}/{n} = {em_raw_pct:.2f}%")
    print(f"📊 Exact match (clean): {em_clean}/{n} = {em_clean_pct:.2f}%")

    # write HTML
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>MathT5 Preview</title>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
  .meta {{ margin-bottom: 12px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
  th {{ background: #f7f7f7; position: sticky; top: 0; z-index: 1; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .ok {{ background: #e9f7ef; }}
  .bad {{ background: #fdecea; }}
  code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body>
  <div class="meta">
    <h2>MathT5 Preview</h2>
    <div>Model dir: <code>{html.escape(str(model_dir))}</code></div>
    <div>Test file: <code>{html.escape(str(test_path))}</code></div>
    <div>Samples: {n} &nbsp;|&nbsp; Beams: {beams}</div>
    <div><b>Exact match</b> — raw: {em_raw}/{n} = {em_raw_pct:.2f}% &nbsp;|&nbsp; cleaned: {em_clean}/{n} = {em_clean_pct:.2f}%</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Spoken</th>
        <th>Ground Truth</th>
        <th>Predicted (raw)</th>
        <th>Predicted (cleaned)</th>
        <th>Raw LaTeX (GT / Pred)</th>
      </tr>
    </thead>
    <tbody>
""")
        for i, (spoken, gt, pred_raw, pred_clean) in enumerate(rows, 1):
            is_match = (norm_for_match(pred_clean) == norm_for_match(gt))
            cls = "ok" if is_match else "bad"
            f.write(f"""
      <tr class="{cls}">
        <td>{i}</td>
        <td>{html.escape(spoken)}</td>
        <td>{gt}</td>
        <td>{pred_raw}</td>
        <td>{pred_clean}</td>
        <td>
          <div><b>GT:</b> <code>{html.escape(gt)}</code></div>
          <div><b>PR:</b> <code>{html.escape(pred_clean)}</code></div>
        </td>
      </tr>
""")
        f.write("""
    </tbody>
  </table>
</body>
</html>
""")
    print("📄 Wrote:", outfile)

if __name__ == "__main__":
    # default args; tweak if you like
    main(samples=200, beams=5, seed=0, outfile="reports/matht5_preview.html")
