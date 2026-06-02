"""import_hf_datasets.py
Downloads and converts large public HuggingFace datasets into the SUB-ai JSON format.
Run: python dataset/import_hf_datasets.py
"""

import json
import os
import glob
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


# -- 1. SQuAD v1 ----------------------------------------------------------
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


# -- 2. OpenAssistant Conversations ---------------------------------------
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


# -- 3. TriviaQA ----------------------------------------------------------
def import_trivia(limit=4000):
    print("\n[*] Importing TriviaQA...")
    try:
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


# -- 4. OpenBookQA --------------------------------------------------------
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


# -- 5. Alpaca Instructions -----------------------------------------------
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


# -- 6. CommonsenseQA ------------------------------------------------------
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


# -- 7. ELI5 (Explain Like I'm 5) -----------------------------------------
# Uses rexarski/eli5_category to avoid deprecated dataset script errors
def import_eli5(limit=5000):
    print("\n[*] Importing ELI5...")
    try:
        ds = load_dataset("rexarski/eli5_category", split="train")
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
        print(f"[!] ELI5 failed (rexarski/eli5_category not available): {e}")
        print("[!] Skipping ELI5 gracefully.")


# -- 8. GSM8K (Grade School Math) -----------------------------------------
def import_gsm8k(limit=3000):
    print("\n[*] Importing GSM8K (Math)...")
    try:
        ds = load_dataset("gsm8k", "main", split="train")
        samples = []
        for row in tqdm(ds.select(range(min(limit, len(ds))))):
            q = clean(row["question"])
            a = clean(row["answer"])
            if q and a:
                samples.append({"type": "math", "input": q, "output": a})
        save("hf_gsm8k.json", samples)
    except Exception as e:
        print(f"[!] GSM8K failed: {e}")


# -- 9. Python Code Instructions ------------------------------------------
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


# -- 10. WikiText (General Knowledge) -------------------------------------
# Paragraphs from Wikipedia for general knowledge training
def import_wikitext(limit=5000):
    print("\n[*] Importing WikiText (wikitext-2-raw-v1)...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        samples = []
        for row in tqdm(ds):
            text = row.get("text", "").strip()
            # Skip empty lines, section headers, and very short lines
            if not text or len(text) < 50 or text.startswith("="):
                continue
            paragraph = clean(text)
            if paragraph and len(paragraph) > 50:
                samples.append({
                    "type": "knowledge",
                    "input": "Tell me about: " + paragraph[:80],
                    "output": paragraph,
                })
            if len(samples) >= limit:
                break
        save("hf_wikitext.json", samples)
    except Exception as e:
        print(f"[!] WikiText failed: {e}")


# -- 11. Dolly (Instruction/Response) -------------------------------------
# Databricks Dolly 15k instruction-following pairs
def import_dolly(limit=5000):
    print("\n[*] Importing Dolly 15k...")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        samples = []
        for row in tqdm(ds.select(range(min(limit, len(ds))))):
            instruction = clean(row.get("instruction", ""))
            context     = clean(row.get("context", ""))
            response    = clean(row.get("response", ""))
            full_input  = f"{instruction} {context}".strip() if context else instruction
            if full_input and response and len(response) > 10:
                samples.append({"type": "instruction", "input": full_input, "output": response})
        save("hf_dolly.json", samples)
    except Exception as e:
        print(f"[!] Dolly failed: {e}")


# -- 12. SciQ (Science QA) ------------------------------------------------
# Science question-answer pairs from Allen AI
def import_sciq(limit=3000):
    print("\n[*] Importing SciQ...")
    try:
        ds = load_dataset("allenai/sciq", split="train")
        samples = []
        for row in tqdm(ds.select(range(min(limit, len(ds))))):
            q = clean(row.get("question", ""))
            a = clean(row.get("correct_answer", ""))
            if q and a:
                samples.append({"type": "science_qa", "input": q, "output": a})
        save("hf_sciq.json", samples)
    except Exception as e:
        print(f"[!] SciQ failed: {e}")


# -- 13. Generic CSV Dataset Loader ----------------------------------------
# Load ANY HuggingFace CSV-based dataset by specifying columns
def import_csv_dataset(dataset_name, input_column, output_column,
                       limit=5000, save_name="hf_custom_csv.json",
                       data_type="general"):
    print(f"\n[*] Importing CSV dataset: {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, split="train")
        samples = []
        for row in tqdm(ds.select(range(min(limit, len(ds))))):
            inp = clean(row.get(input_column, ""))
            out = clean(row.get(output_column, ""))
            if inp and out:
                samples.append({"type": data_type, "input": inp, "output": out})
        save(save_name, samples)
    except Exception as e:
        print(f"[!] CSV dataset '{dataset_name}' failed: {e}")


# -- 14. Local Markdown Loader --------------------------------------------
# Reads .md files from data/raw/markdown/, splits into ~300-word chunks
def import_md_dataset(md_dir="data/raw/markdown/", save_name="hf_markdown.json"):
    print(f"\n[*] Importing local Markdown files from {md_dir}...")
    try:
        if not os.path.isdir(md_dir):
            print(f"[!] Directory not found: {md_dir} -- skipping markdown import.")
            return

        md_files = glob.glob(os.path.join(md_dir, "*.md"))
        if not md_files:
            print(f"[!] No .md files found in {md_dir} -- skipping.")
            return

        samples = []
        chunk_size = 300  # approximate words per chunk

        for filepath in tqdm(md_files, desc="Reading .md files"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                continue

            lines = content.strip().split("\n")
            # Use the first non-empty line as the topic title
            topic_title = ""
            for line in lines:
                stripped = line.strip().lstrip("#").strip()
                if stripped:
                    topic_title = stripped
                    break
            if not topic_title:
                topic_title = os.path.basename(filepath).replace(".md", "")

            # Split content into word-based chunks of ~300 words
            words = content.split()
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i : i + chunk_size]
                chunk_text = " ".join(chunk_words).strip()
                if len(chunk_text) < 30:
                    continue
                inp = f"Explain the following topic: {topic_title}"
                out = clean(chunk_text)
                if out:
                    samples.append({"type": "knowledge", "input": inp, "output": out})

        if samples:
            save(save_name, samples)
        else:
            print("[!] No markdown chunks extracted -- nothing saved.")
    except Exception as e:
        print(f"[!] Markdown import failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print(" SUB-ai 2.0 - HuggingFace Dataset Importer")
    print("=" * 60)

    # Original datasets
    import_squad(limit=10000)         # Wikipedia QA
    import_oasst(limit=5000)          # Real human conversations
    import_trivia(limit=5000)         # Trivia facts
    import_openbookqa()               # Science QA
    import_alpaca(limit=8000)         # Instruction following
    import_commonsense()              # Common sense reasoning
    import_eli5(limit=5000)           # Simple explanations (fixed)
    import_gsm8k(limit=3000)          # Grade School Math
    import_python_code(limit=3000)    # Coding

    # New datasets
    import_wikitext(limit=5000)       # General knowledge paragraphs
    import_dolly(limit=5000)          # Instruction/response pairs
    import_sciq(limit=3000)           # Science QA

    # Local markdown (if directory exists)
    import_md_dataset()

    # Generic CSV loader -- uncomment and customize as needed:
    # import_csv_dataset(
    #     dataset_name="csv_dataset_name_on_hf",
    #     input_column="question",
    #     output_column="answer",
    #     limit=5000,
    #     save_name="hf_custom_csv.json",
    #     data_type="general",
    # )

    print("\n[v] All datasets imported! Now run:")
    print("    python dataset/build_dataset.py")
    print("    python train.py")
