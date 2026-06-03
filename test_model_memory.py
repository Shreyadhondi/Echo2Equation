import os
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def print_memory(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)
    print(f"📦 Memory after {stage}: {mem:.2f} GB")

# Step 1: Load tokenizer and model
model_name = "jmeadows17/MathT5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print_memory("loading model")

# Step 2: Add special tokens
special_tokens = [
    "\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\sin", "\\cos", "\\tan",
    "\\log", "\\ln", "\\pi", "\\theta", "\\alpha", "\\beta", "\\gamma",
    "\\delta", "\\epsilon", "\\infty", "\\pm", "\\cdot", "\\times",
    "\\rightarrow", "\\left", "\\right", "^", "_", "{", "}"
]
num_added = tokenizer.add_tokens(list(set(special_tokens)))
print(f"🧩 Added {num_added} new tokens.")
print_memory("adding special tokens")

# Step 3: Resize model embeddings
model.resize_token_embeddings(len(tokenizer))
print_memory("resizing embeddings")

print("✅ Test script finished.")
