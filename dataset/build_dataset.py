"""build_dataset.py
Merges all raw JSON datasets (custom + HuggingFace) into one processed dataset.
Auto-discovers any hf_*.json files in data/raw/ that are not in the explicit list.
Run: python dataset/build_dataset.py
"""

import json
import os
import glob

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
    "hf_conversations.json",     # OpenAssistant conversations
    "hf_trivia.json",
    "hf_openbookqa.json",
    "hf_alpaca.json",
    "hf_commonsense.json",
    "hf_eli5.json",              # Explain Like I'm 5 explanations
    "hf_gsm8k.json",             # Grade school math QA
    "hf_python_code.json",       # Python code instructions
    # New datasets
    "hf_wikitext.json",          # Wikipedia general knowledge
    "hf_dolly.json",             # Databricks Dolly instruction pairs
    "hf_sciq.json",              # Science QA
    "hf_markdown.json",          # Local markdown knowledge
]


def discover_hf_files():
    """Auto-discover any hf_*.json files in data/raw/ not already in the list."""
    known = set(DATASET_FILES)
    discovered = []
    pattern = os.path.join(RAW_DIR, "hf_*.json")
    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        if filename not in known:
            discovered.append(filename)
    return discovered


def build():
    os.makedirs("data/processed", exist_ok=True)

    # Combine explicit list with auto-discovered hf_ files
    all_files = list(DATASET_FILES)
    extra = discover_hf_files()
    if extra:
        print(f"[*] Auto-discovered {len(extra)} extra hf_ dataset(s):")
        for f in extra:
            print(f"    -> {f}")
        all_files.extend(extra)

    processed = []
    stats = {}

    for filename in all_files:
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
    print(f"[v] Total samples: {len(processed):,}")
    print(f"[v] Saved to: {OUT_PATH}")
    print(f"\nBreakdown by category:")
    for cat, cnt in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<20} {cnt:>6,}")


if __name__ == "__main__":
    build()
