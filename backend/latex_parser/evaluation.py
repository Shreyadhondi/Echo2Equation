import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

# Path to the trained model
model_path = "models/t5_model/t5_model_run"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Load tokenized test set
dataset = load_dataset("json", data_files={"test": "data/tokenized_data/test.json"})

# Data collator for padding
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Dummy training args for evaluation only
args = Seq2SeqTrainingArguments(
    output_dir="tmp_eval",  # Temporary folder
    per_device_eval_batch_size=16,
    predict_with_generate=True
)

# Trainer for evaluation
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Run evaluation
results = trainer.evaluate()
print("📊 Evaluation Results:", results)

# ---------------------------------
# Show 5 random predictions
# ---------------------------------
print("\n🔍 Sample Predictions from Test Set:")
sample_indices = random.sample(range(len(dataset["test"])), 5)

for idx in sample_indices:
    sample = dataset["test"][idx]
    spoken_text = sample["spoken"]
    label_ids = sample["labels"]

    # Decode ground truth LaTeX
    ground_truth = tokenizer.decode(label_ids, skip_special_tokens=True)

    # Tokenize input and generate prediction
    inputs = tokenizer(spoken_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=82)
    predicted_latex = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n🗣 Spoken: {spoken_text}")
    print(f"✅ Ground Truth: {ground_truth}")
    print(f"🤖 Predicted:   {predicted_latex}")
