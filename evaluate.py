import json
import torch
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


def generate_custom(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8, top_k=40, device="cpu"):
    model.eval()
    cfg = CONFIG

    tokens = [tokenizer.word2idx.get("<BOS>", 2)]
    tokens += [tokenizer.word2idx.get(w, 1) for w in prompt.lower().split()]

    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.size(1) >= cfg["max_seq_len"]:
                break
            logits = model(x)
            next_logits = logits[0, -1, :] / temperature

            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[-1]] = -float("inf")

            probs      = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.word2idx.get("<EOS>", 3):
                break

            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    generated = x[0].tolist()[len(tokens):]
    words = [tokenizer.idx2word.get(t, "<UNK>") for t in generated]
    return " ".join(words).strip()


def generate_pretrained(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8, top_k=40, device="cpu"):
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
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[v] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[!] Using CPU")

    # Load configuration
    config_path = "checkpoints/config.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = CONFIG

    model_mode = cfg.get("model_mode", "custom")
    print(f"[*] Evaluation Mode: {model_mode.upper()}")

    if model_mode == "pretrained":
        if not HF_AVAILABLE:
            raise ImportError("Pretrained mode requires HuggingFace libraries. Install transformers, peft, and accelerate.")
            
        base_model_name = cfg["pretrained_model_name"]
        adapter_path = os.path.join("checkpoints", "pretrained_model")
        
        if os.path.exists(adapter_path):
            print(f"[v] Loading fine-tuned adapter weights from {adapter_path}...")
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
            torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            base_model = base_model.to(device)
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print(f"[!] Fine-tuned weights not found. Loading base pre-trained model: {base_model_name}...")
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

    else:
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

        model.load_state_dict(torch.load(cfg["save_path"], map_location=device, weights_only=True))
        print("[v] Custom Model loaded\n")

    # Format prompts depending on the mode
    if model_mode == "pretrained":
        prompts = [
            "User: What is AI?\nBot:",
            "User: What is machine learning?\nBot:",
            "User: What is Python?\nBot:",
            "User: Tell me a fact.\nBot:",
            "User: What is 12 plus 8?\nBot:",
            "User: What is the capital of India?\nBot:",
            "User: How do you write a function in Python?\nBot:",
        ]
    else:
        # Custom model expected prompts (which internally split spaces, but keeping standard is fine)
        prompts = [
            "User: What is AI?\nBot:",
            "User: What is machine learning?\nBot:",
            "User: What is Python?\nBot:",
            "User: Tell me a fact.\nBot:",
            "User: What is 12 plus 8?\nBot:",
            "User: What is the capital of India?\nBot:",
            "User: How do you write a function in Python?\nBot:",
        ]

    print("\nStarting generation testing:\n" + "="*50)
    for prompt in prompts:
        if model_mode == "pretrained":
            response = generate_pretrained(model, tokenizer, prompt, max_new_tokens=40, temperature=0.7, top_k=50, device=device)
        else:
            response = generate_custom(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8, top_k=40, device=device)
            
        print(f"Prompt  : {prompt.replace(chr(10), ' ')}")
        print(f"Response: {response}")
        print("-" * 50)
