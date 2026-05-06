import json
import os
import glob

RAW_DIR  = "data/raw"
OUT_PATH = "data/processed/dataset.json"

DATASET_FILES = [
    "data.json",
    "math_dataset.json",
    "coding_dataset.json",
    "reasoning_dataset.json",
    "story_dataset.json",
    "sentiment_dataset.json",
    "instruction_dataset.json",
]

def build():
    os.makedirs("data/processed", exist_ok=True)
    processed = []

    for filename in DATASET_FILES:
        filepath = os.path.join(RAW_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[!] Skipping missing file: {filepath}")
            continue

        with open(filepath, "r") as f:
            raw = json.load(f)

        count = 0
        for item in raw:
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            category = item.get("type", "general")
            if inp and out:
                text = f"User: {inp}\nBot: {out}"
                processed.append({"text": text, "category": category})
                count += 1

        print(f"[+] {filename}: {count} samples loaded")

    with open(OUT_PATH, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"\n[\u2713] Total dataset built: {len(processed)} samples -> {OUT_PATH}")

if __name__ == "__main__":
    build()
