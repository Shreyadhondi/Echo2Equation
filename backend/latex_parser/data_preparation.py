import json
import random
from pathlib import Path

def load_jsonl(file_path):
    """Loads a .jsonl file into a list of dictionaries"""
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(data, file_path):
    """Saves a list of dictionaries to a .jsonl file"""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def prepare_data_splits(seed=42):
    # Define paths
    base_dir = Path(__file__).resolve().parents[2]  # Go 2 levels up from backend/latex_parser
    data_file = base_dir / 'data' / 'full_dataset.jsonl'
    split_dir = base_dir / 'data' / 'splits'

    # Ensure the splits folder exists
    split_dir.mkdir(parents=True, exist_ok=True)

    # Load and shuffle data
    data = load_jsonl(data_file)
    random.seed(seed)
    random.shuffle(data)

    # Split ratios
    total = len(data)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Save the splits
    save_jsonl(train_data, split_dir / 'train.jsonl')
    save_jsonl(val_data, split_dir / 'val.jsonl')
    save_jsonl(test_data, split_dir / 'test.jsonl')

    # Summary
    print(f"Data preparation complete.")
    print(f"   → Total samples: {total}")
    print(f"   → Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"   → Splits saved in: {split_dir}")

if __name__ == '__main__':
    prepare_data_splits()

#OUTPUT:
#-------------------------------------------------------------------
#Data preparation complete.
#   → Total samples: 12621
#   → Train: 10096, Val: 1262, Test: 1263
#   → Splits saved in: /home/shreya/Echo2Equation/data/splits
#--------------------------------------------------------------------