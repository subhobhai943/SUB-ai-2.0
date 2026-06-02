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
    if not isinstance(text, str):
        return ""
    return text.strip().replace("\n", " ").replace("\r", "")[:500]


# ── 1. SQuAD v1 ────────────────────────────────────────────────────────────
# 87k reading comprehension QA pairs from Wikipedia
def import_squad(limit=5000):
    print("\n[*] Importing SQuAD v1...")
    try:
        ds = load_dataset("rajpurkar/squad", split="train")
        samples = []
        for row in tqdm(ds.select(range(min(limit, len(ds))))):
            q = clean(row["question"])
            a = clean(row["answers"]["text"][0]) if row["answers"]["text"] else ""
            if q and a:
                samples.append({"type": "squad_qa", "input": q, "output": a})
        save("hf_squad.json", samples)
    except Exception as e:
        print(f"[!] SQuAD failed: {e}")


# ── 2. OpenAssistant Conversations ──────────────────────────────────────
# Real human conversations - replaces daily_dialog which no longer works
def import_oasst(limit=3000):
    print("\n[*] Importing OpenAssistant conversations...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        # Build turn pairs: prompter -> assistant
        messages = [row for row in ds]
        by_id = {m["message_id"]: m for m in messages}
        samples = []
        for msg in tqdm(messages):
            if msg["role"] == "assistant" and msg["parent_id"] in by_id:
                parent = by_id[msg["parent_id"]]
                if parent["role"] == "prompter":
                    inp = clean(parent["text"])
                    out = clean(msg["text"])
                    if inp and out and len(inp) > 5 and len(out) > 10:
                        samples.append({"type": "conversation", "input": inp, "output": out})
            if len(samples) >= limit:
                break
        save("hf_conversations.json", samples[:limit])
    except Exception as e:
        print(f"[!] OpenAssistant failed: {e}")


# ── 3. TriviaQA ──────────────────────────────────────────────────────────────
def import_trivia(limit=4000):
    print("\n[*] Importing TriviaQA...")
    try:
        # trust_remote_code removed - not needed in newer datasets versions
        ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
        samples = []
        for row in tqdm(ds.select(range(min(limit * 2, len(ds))))):
            q = clean(row["question"])
            a = clean(row["answer"]["value"]) if row["answer"]["value"] else ""
            if q and a and len(a) < 120:
                samples.append({"type": "trivia_qa", "input": q, "output": a})
            if len(samples) >= limit:
                break
        save("hf_trivia.json", samples)
    except Exception as e:
        print(f"[!] TriviaQA failed: {e}")


# ── 4. OpenBookQA ───────────────────────────────────────────────────────────
def import_openbookqa():
    print("\n[*] Importing OpenBookQA...")
    try:
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
    except Exception as e:
        print(f"[!] OpenBookQA failed: {e}")


# ── 5. Alpaca Instructions ───────────────────────────────────────────────────
def import_alpaca(limit=5000):
    print("\n[*] Importing Alpaca instructions...")
    try:
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
    except Exception as e:
        print(f"[!] Alpaca failed: {e}")


# ── 6. CommonsenseQA ─────────────────────────────────────────────────────────
def import_commonsense():
    print("\n[*] Importing CommonsenseQA...")
    try:
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
    except Exception as e:
        print(f"[!] CommonsenseQA failed: {e}")


# ── 7. ELI5 (Explain Like I'm 5) ─────────────────────────────────────────
# Simple plain-English answers to complex questions
def import_eli5(limit=5000):
    print("\n[*] Importing ELI5...")
    try:
        ds = load_dataset("eli5_category", split="train", trust_remote_code=False)
        samples = []
        for row in tqdm(ds.select(range(min(limit * 3, len(ds))))):
            q = clean(row.get("title", ""))
            answers = row.get("answers", {}).get("text", [])
            a = clean(answers[0]) if answers else ""
            if q and a and len(a) > 20 and len(a) < 400:
                samples.append({"type": "eli5", "input": q, "output": a})
            if len(samples) >= limit:
                break
        save("hf_eli5.json", samples)
    except Exception as e:
        print(f"[!] ELI5 failed: {e}")


# ── 8. GSM8K (Grade School Math) ─────────────────────────────────────────
def import_gsm8k(limit=3000):
    print("\n[*] Importing GSM8K (Math)...")
    try:
        ds = load_dataset("gsm8k", "main", split="train")
        samples = []
        for row in tqdm(ds.select(range(min(limit, len(ds))))):
            q = clean(row["question"])
            # The raw answer has steps with final result, let's clean it up slightly
            # We can use it as is for step-by-step reasoning
            a = clean(row["answer"])
            if q and a:
                samples.append({"type": "math", "input": q, "output": a})
        save("hf_gsm8k.json", samples)
    except Exception as e:
        print(f"[!] GSM8K failed: {e}")


# ── 9. Python Code Instructions ──────────────────────────────────────────
def import_python_code(limit=3000):
    print("\n[*] Importing Python Code Instructions...")
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        samples = []
        for row in tqdm(ds.select(range(min(limit, len(ds))))):
            instruction = clean(row["instruction"])
            inp_extra   = clean(row.get("input", ""))
            output      = clean(row["output"])
            full_input  = f"{instruction} {inp_extra}".strip() if inp_extra else instruction
            if full_input and output:
                samples.append({"type": "code", "input": full_input, "output": output})
        save("hf_python_code.json", samples)
    except Exception as e:
        print(f"[!] Python Code failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print(" SUB-ai 2.0 - HuggingFace Dataset Importer")
    print("=" * 60)

    import_squad(limit=10000)         # Wikipedia QA (increased)
    import_oasst(limit=5000)          # Real human conversations (increased)
    import_trivia(limit=5000)         # Trivia facts (increased)
    import_openbookqa()               # Science QA
    import_alpaca(limit=8000)         # Instruction following (increased)
    import_commonsense()              # Common sense reasoning
    import_eli5(limit=5000)           # Simple explanations (increased)
    import_gsm8k(limit=3000)          # Grade School Math (NEW)
    import_python_code(limit=3000)    # Coding (NEW)

    print("\n[v] All datasets imported! Now run:")
    print("    python dataset/build_dataset.py")
    print("    python train.py")
