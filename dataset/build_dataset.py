import json
import os

RAW_DIR  = "data/raw"
OUT_PATH = "data/processed/dataset.json"

# All dataset files to merge (add new ones here)
DATASET_FILES = [
    # Custom hand-crafted datasets
    "data.json",
    "math_dataset.json",
    "coding_dataset.json",
    "reasoning_dataset.json",
    "story_dataset.json",
    "sentiment_dataset.json",
    "instruction_dataset.json",
    # HuggingFace imported datasets
    "hf_squad.json",
    "hf_daily_dialog.json",
    "hf_trivia.json",
    "hf_openbookqa.json",
    "hf_alpaca.json",
    "hf_commonsense.json",
]

def build():
    os.makedirs("data/processed", exist_ok=True)
    processed = []
    stats = {}

    for filename in DATASET_FILES:
        filepath = os.path.join(RAW_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[!] Skipping (not found): {filename}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        count = 0
        for item in raw:
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            category = item.get("type", "general")
            # Filter junk
            if inp and out and len(inp) > 3 and len(out) > 3:
                text = f"User: {inp}\nBot: {out}"
                processed.append({"text": text, "category": category})
                count += 1
                stats[category] = stats.get(category, 0) + 1

        print(f"[+] {filename}: {count:,} samples")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"[✓] Total samples: {len(processed):,}")
    print(f"[✓] Saved to: {OUT_PATH}")
    print(f"\nBreakdown by category:")
    for cat, cnt in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<20} {cnt:>6,}")

if __name__ == "__main__":
    build()
