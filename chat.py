"""
chat.py - Interactive terminal chat with SUB-ai 2.0
Usage: python chat.py
"""

import json
import torch
import sys
from model.model import SUBaiModel
from model.config import CONFIG
from train import SimpleTokenizer

BANNER = """
┌─────────────────────────────────────────────────────────────┐
│          SUB-ai 2.0  —  Interactive Chat                    │
│          Built by Subhobhai from scratch                    │
│                                                             │
│  Commands:                                                  │
│    /temp <0.1-2.0>  — change temperature                    │
│    /topk <1-100>    — change top-k sampling                 │
│    /len  <tokens>   — change response length                │
│    /reset           — clear conversation history            │
│    /settings        — show current settings                 │
│    /quit or /exit   — exit the chat                         │
└─────────────────────────────────────────────────────────────┘
"""


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=30, device="cpu", max_seq_len=256):
    model.eval()

    # ── Tokenize exactly like training: replace newlines with spaces, then split ──
    clean_prompt = prompt.replace("\n", " ")
    tokens = [tokenizer.word2idx.get("<BOS>", 2)]
    tokens += [tokenizer.word2idx.get(w, tokenizer.word2idx.get("<UNK>", 1))
               for w in clean_prompt.lower().split()]

    # Trim to leave room for generation
    max_prompt_len = max_seq_len - max_new_tokens
    if len(tokens) > max_prompt_len:
        tokens = tokens[:1] + tokens[-(max_prompt_len - 1):]  # keep <BOS> + tail

    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.size(1) >= max_seq_len:
                break

            logits     = model(x)
            next_logits = logits[0, -1, :] / max(temperature, 1e-6)

            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[-1]] = -float("inf")

            probs      = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tok_id = next_token.item()

            # Stop on <EOS> or if model starts a new "user:" turn
            if tok_id == tokenizer.word2idx.get("<EOS>", 3):
                break
            if tokenizer.idx2word.get(tok_id, "") == "user:":
                break

            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    generated = x[0].tolist()[len(tokens):]
    SKIP = {"<PAD>", "<BOS>", "<EOS>", "<UNK>", "user:", "bot:"}
    words = [
        tokenizer.idx2word.get(t, "")
        for t in generated
        if tokenizer.idx2word.get(t, "") not in SKIP
    ]
    return " ".join(words).strip()


def load_model(device):
    import os
    config_path = "checkpoints/config.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = CONFIG

    tokenizer = SimpleTokenizer(cfg["vocab_size"])
    tokenizer.load(cfg["vocab_path"])

    model = SUBaiModel(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        max_seq_len=cfg["max_seq_len"]
    ).to(device)

    model.load_state_dict(
        torch.load(cfg["save_path"], map_location=device, weights_only=True)
    )
    model.eval()
    return model, tokenizer, cfg

def main():
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(BANNER)
    print(f"  Device  : {device_name}")

    # Load
    try:
        model, tokenizer, loaded_cfg = load_model(device)
        vocab_size = len(tokenizer.word2idx)
        print(f"  Vocab   : {vocab_size:,} tokens")
        print(f"  Model   : loaded from {loaded_cfg['save_path']}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not load model: {e}")
        print("  Make sure you have trained the model first: python train.py")
        sys.exit(1)

    # Settings
    temperature  = 0.7
    top_k        = 30
    max_tokens   = 60
    history      = []   # list of (user, bot) tuples

    print("\n  Type your message below. Type /quit to exit.")
    print("-" * 62)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[SUB-ai] Goodbye!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd   = parts[0].lower()

            if cmd in ("/quit", "/exit"):
                print("\n[SUB-ai] Goodbye!")
                break

            elif cmd == "/reset":
                history.clear()
                print("[SUB-ai] Conversation history cleared.")

            elif cmd == "/settings":
                print(f"[SUB-ai] temperature={temperature} | top_k={top_k} | max_tokens={max_tokens}")

            elif cmd == "/temp" and len(parts) == 2:
                try:
                    temperature = float(parts[1])
                    print(f"[SUB-ai] Temperature set to {temperature}")
                except ValueError:
                    print("[SUB-ai] Invalid value. Usage: /temp 0.7")

            elif cmd == "/topk" and len(parts) == 2:
                try:
                    top_k = int(parts[1])
                    print(f"[SUB-ai] Top-k set to {top_k}")
                except ValueError:
                    print("[SUB-ai] Invalid value. Usage: /topk 50")

            elif cmd == "/len" and len(parts) == 2:
                try:
                    max_tokens = int(parts[1])
                    print(f"[SUB-ai] Max response tokens set to {max_tokens}")
                except ValueError:
                    print("[SUB-ai] Invalid value. Usage: /len 60")

            else:
                print("[SUB-ai] Unknown command. Type /quit to exit.")
            continue

        # ── Build prompt with context (last 2 turns) ────────────────────
        # Use spaces instead of newlines — training data newlines become
        # spaces after .split(), so the model never actually learned \n as
        # a separator.  Keeping the "User:" / "Bot:" markers is enough.
        context = ""
        for u, b in history[-2:]:
            context += f"User: {u} Bot: {b} "
        prompt = context + f"User: {user_input} Bot:"

        response = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
            max_seq_len=loaded_cfg["max_seq_len"]
        )

        if not response:
            response = "(no response generated — try rephrasing or retraining)"

        print(f"\nSUB-ai: {response}")
        history.append((user_input, response))


if __name__ == "__main__":
    main()
