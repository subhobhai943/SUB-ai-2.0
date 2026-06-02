"""
chat.py - Interactive terminal chat with SUB-ai 2.0 (Custom & Pretrained modes)
Usage: python chat.py
"""

import json
import torch
import sys
import os
from train import SimpleTokenizer
from model.model import SUBaiModel
from model.config import CONFIG

# Hugging Face imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

BANNER = """
┌─────────────────────────────────────────────────────────────┐
│          SUB-ai 2.0  —  Interactive Chat                    │
│          Built by Subhobhai & Antigravity                   │
│                                                             │
│  Commands:                                                  │
│    /temp <0.1-2.0>  — change temperature                    │
│    /topk <1-100>    — change top-k sampling                 │
│    /len  <tokens>   — change response length                │
│    /reset           — clear conversation history            │
│    /settings        — show current settings                 │
│    /quit or /exit   — exit the chat                         │
│└─────────────────────────────────────────────────────────────┘
"""

def generate_custom(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=30, device="cpu", max_seq_len=256):
    model.eval()

    # Tokenize exactly like training: replace newlines with spaces, then split
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


def generate_pretrained(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50, device="cpu"):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True if temperature > 0.0 else False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


def load_model(device):
    config_path = "checkpoints/config.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = CONFIG

    model_mode = cfg.get("model_mode", "custom")
    
    if model_mode == "pretrained":
        if not HF_AVAILABLE:
            raise ImportError("Pretrained mode requires huggingface libraries. Install transformers, peft, and accelerate.")
            
        base_model_name = cfg["pretrained_model_name"]
        adapter_path = os.path.join("checkpoints", "pretrained_model")
        
        # Determine whether we load the fine-tuned adapter weights or fall back to base model
        if os.path.exists(adapter_path):
            print(f"[v] Loading fine-tuned adapter weights from {adapter_path}...")
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
            
            # Load base model in FP16 on GPU, FP32 on CPU
            torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            base_model = base_model.to(device)
            
            # Wrap with Peft adapter
            model = PeftModel.from_pretrained(base_model, adapter_path)
            print("[v] Pretrained base model with fine-tuned LoRA adapter loaded successfully.")
        else:
            print(f"[!] Fine-tuned weights not found in {adapter_path}. Loading base pre-trained model: {base_model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            model = model.to(device)
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer, cfg

    else:
        # Custom scratch model loading
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
        model_mode = loaded_cfg.get("model_mode", "custom")
        print(f"  Mode    : {model_mode.upper()}")
        if model_mode == "custom":
            vocab_size = len(tokenizer.word2idx)
            print(f"  Vocab   : {vocab_size:,} tokens")
            print(f"  Model   : custom weights loaded from {loaded_cfg['save_path']}")
        else:
            print(f"  Model   : {loaded_cfg['pretrained_model_name']}")
    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        print("  Make sure you have trained the model or config.py is correct.")
        sys.exit(1)

    # Settings
    temperature  = 0.7
    top_k        = 30 if model_mode == "custom" else 50
    max_tokens   = 60 if model_mode == "custom" else 100
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

        # ── Build prompt with context ──────────────────────────────────
        if model_mode == "pretrained":
            # Newlines for pretrained model
            context = ""
            for u, b in history[-2:]:
                context += f"User: {u}\nBot: {b}\n"
            prompt = context + f"User: {user_input}\nBot:"
            
            response = generate_pretrained(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
        else:
            # Spaces for custom model
            context = ""
            for u, b in history[-2:]:
                context += f"User: {u} Bot: {b} "
            prompt = context + f"User: {user_input} Bot:"
            
            response = generate_custom(
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
