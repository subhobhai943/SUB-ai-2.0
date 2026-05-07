"""import_hf_datasets.py
Downloads and converts large public HuggingFace datasets into the SUB-ai JSON format.
Run: python dataset/import_hf_datasets.py
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)


def save(name, samples):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved {len(samples):,} samples -> {path}")


def clean(text):
    return text.strip().replace("\n", " ").replace("\r", "")


# ── 1. SQuAD v1 ─────────────────────────────────────────────────────────────
# 87k reading comprehension QA pairs from Wikipedia
def import_squad(limit=5000):
    print("\n[*] Importing SQuAD v1...")
    ds = load_dataset("rajpurkar/squad", split="train")
    samples = []
    for row in tqdm(ds.select(range(min(limit, len(ds))))):
        q = clean(row["question"])
        a = clean(row["answers"]["text"][0]) if row["answers"]["text"] else ""
        if q and a:
            samples.append({"type": "squad_qa", "input": q, "output": a})
    save("hf_squad.json", samples)


# ── 2. DailyDialog ───────────────────────────────────────────────────────────
# 13k multi-turn daily conversations
def import_daily_dialog(limit=3000):
    print("\n[*] Importing DailyDialog...")
    ds = load_dataset("daily_dialog", split="train", trust_remote_code=True)
    samples = []
    for row in tqdm(ds.select(range(min(limit, len(ds))))):
        dialog = row["dialog"]
        for i in range(len(dialog) - 1):
            inp = clean(dialog[i])
            out = clean(dialog[i + 1])
            if inp and out and len(inp) > 5 and len(out) > 5:
                samples.append({"type": "daily_dialog", "input": inp, "output": out})
        if len(samples) >= limit:
            break
    save("hf_daily_dialog.json", samples[:limit])


# ── 3. TriviaQA ──────────────────────────────────────────────────────────────
# 95k trivia questions with verified answers
def import_trivia(limit=4000):
    print("\n[*] Importing TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    samples = []
    for row in tqdm(ds.select(range(min(limit * 2, len(ds))))):
        q = clean(row["question"])
        a = clean(row["answer"]["value"]) if row["answer"]["value"] else ""
        if q and a and len(a) < 120:
            samples.append({"type": "trivia_qa", "input": q, "output": a})
        if len(samples) >= limit:
            break
    save("hf_trivia.json", samples)


# ── 4. OpenBookQA ────────────────────────────────────────────────────────────
# 5957 science QA questions with explanations
def import_openbookqa():
    print("\n[*] Importing OpenBookQA...")
    ds = load_dataset("allenai/openbookqa", "main", split="train")
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    samples = []
    for row in tqdm(ds):
        q = clean(row["question_stem"])
        choices = row["choices"]["text"]
        answer_key = row["answerKey"]
        idx = label_map.get(answer_key, 0)
        a = clean(choices[idx]) if idx < len(choices) else ""
        if q and a:
            samples.append({"type": "science_qa", "input": q, "output": a})
    save("hf_openbookqa.json", samples)


# ── 5. Alpaca-style Instructions ─────────────────────────────────────────────
# 52k instruction-following samples (Stanford Alpaca format)
def import_alpaca(limit=5000):
    print("\n[*] Importing Alpaca instructions...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    samples = []
    for row in tqdm(ds.select(range(min(limit, len(ds))))):
        instruction = clean(row["instruction"])
        inp_extra   = clean(row.get("input", ""))
        output      = clean(row["output"])
        full_input  = f"{instruction} {inp_extra}".strip() if inp_extra else instruction
        if full_input and output and len(output) < 300:
            samples.append({"type": "instruction", "input": full_input, "output": output})
    save("hf_alpaca.json", samples)


# ── 6. CommonsenseQA ─────────────────────────────────────────────────────────
# Common sense reasoning questions
def import_commonsense():
    print("\n[*] Importing CommonsenseQA...")
    ds = load_dataset("tau/commonsense_qa", split="train")
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    samples = []
    for row in tqdm(ds):
        q = clean(row["question"])
        choices = row["choices"]["text"]
        key = row.get("answerKey", "A")
        idx = label_map.get(key, 0)
        a = clean(choices[idx]) if idx < len(choices) else ""
        if q and a:
            samples.append({"type": "commonsense", "input": q, "output": a})
    save("hf_commonsense.json", samples)


if __name__ == "__main__":
    print("="*60)
    print(" SUB-ai 2.0 - HuggingFace Dataset Importer")
    print("="*60)

    import_squad(limit=5000)
    import_daily_dialog(limit=3000)
    import_trivia(limit=4000)
    import_openbookqa()
    import_alpaca(limit=5000)
    import_commonsense()

    print("\n[✓] All datasets imported! Now run:")
    print("    python dataset/build_dataset.py")
    print("    python train.py")
