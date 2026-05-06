import json
import os

RAW_PATH = "data/raw/data.json"
OUT_PATH = "data/processed/dataset.json"

def build():
    os.makedirs("data/processed", exist_ok=True)

    with open(RAW_PATH, "r") as f:
        raw = json.load(f)

    processed = []
    for item in raw:
        inp = item["input"].strip()
        out = item["output"].strip()
        # Format: "User: <input>\nBot: <output>"
        text = f"User: {inp}\nBot: {out}"
        processed.append({"text": text})

    with open(OUT_PATH, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"[✓] Dataset built: {len(processed)} samples -> {OUT_PATH}")

if __name__ == "__main__":
    build()
