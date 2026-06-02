import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from model.model import SUBaiModel
from model.config import CONFIG
from tqdm import tqdm

# Hugging Face libraries for Pretrained Mode
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ── Tokenizer (For Custom Mode) ──────────────────────────────────────────────
class SimpleTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build_vocab(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(t.lower().split())
        for word, _ in counter.most_common(self.vocab_size - 4):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, indent=2, ensure_ascii=False)
        print(f"[v] Vocabulary saved to {path}")

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}

    def encode(self, text, max_len):
        tokens = [self.word2idx.get("<BOS>", 2)]
        tokens += [self.word2idx.get(w, 1) for w in text.lower().split()]
        tokens.append(self.word2idx.get("<EOS>", 3))
        tokens = tokens[:max_len]
        tokens += [0] * (max_len - len(tokens))
        return tokens

# ── Dataset (For Custom Mode) ───────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.samples = [tokenizer.encode(d["text"], max_len) for d in data]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:],  dtype=torch.long)
        return x, y

# ── Dataset (For Pretrained Fine-Tuning Mode) ─────────────────────────────────
class PretrainedTextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        self.samples = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else ""
        
        for d in tqdm(data, desc="Tokenizing dataset"):
            text = d["text"] + eos_token
            enc = tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].squeeze(0)
            
            # In labels, replace padding token with -100 to ignore loss
            labels = input_ids.clone()
            labels[labels == pad_token_id] = -100
            
            self.samples.append((input_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, labels = self.samples[idx]
        x = input_ids[:-1]
        y = labels[1:]
        return x, y

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    os.makedirs("checkpoints", exist_ok=True)

    # GPU setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[v] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[!] CUDA not available. Using CPU.")

    # Load raw text for vocab/custom datasets
    with open("data/processed/dataset.json", encoding="utf-8") as f:
        raw = json.load(f)
    texts = [d["text"] for d in raw]

    # Model mode checking
    model_mode = cfg.get("model_mode", "custom")
    print(f"[*] Training Mode: {model_mode.upper()}")

    if model_mode == "pretrained":
        if not HF_AVAILABLE:
            raise ImportError("Pretrained mode requires huggingface libraries. Please install transformers, peft, and accelerate.")
        
        model_name = cfg["pretrained_model_name"]
        print(f"[*] Loading pre-trained tokenizer & model for: {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model in half-precision on GPU to save memory, or full-precision on CPU
        if device.type == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(device)

        if cfg.get("use_lora", False):
            print("[*] Wrapping model with LoRA (Parameter Efficient Fine-Tuning)...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.get("lora_r", 8),
                lora_alpha=cfg.get("lora_alpha", 16),
                lora_dropout=cfg.get("lora_dropout", 0.05),
                target_modules=cfg.get("lora_target_modules", None)
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        full_dataset = PretrainedTextDataset("data/processed/dataset.json", tokenizer, cfg["max_seq_len"])
        vocab_size = model.config.vocab_size
        ignore_index = -100

    else:
        # Custom Mode (Word-level Tokenizer & Scratch Transformer)
        tokenizer = SimpleTokenizer(cfg["vocab_size"])
        tokenizer.build_vocab(texts)
        tokenizer.save(cfg["vocab_path"])

        full_dataset = TextDataset("data/processed/dataset.json", tokenizer, cfg["max_seq_len"])
        vocab_size = cfg["vocab_size"]
        ignore_index = 0

        model = SUBaiModel(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            ff_dim=cfg["ff_dim"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"]
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v] Total Parameters: {total_params:,}")
    print(f"[v] Trainable Parameters: {trainable_params:,}")

    # Split dataset
    val_size   = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    best_val_loss = float("inf")

    # Mixed precision scaler (for GPU training speed/memory)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and model_mode == "pretrained"))

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [Train]"):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and model_mode == "pretrained")):
                outputs = model(x)
                # HF models return a tuple/SequenceClassifierOutput, custom model returns logits directly
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and model_mode == "pretrained")):
                    outputs = model(x)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    val_loss += criterion(logits.reshape(-1, vocab_size), y.reshape(-1)).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        scheduler.step()

        print(f"  Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            
            if model_mode == "pretrained":
                save_dir = os.path.join("checkpoints", "pretrained_model")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"  [v] Best pre-trained adapter model saved to {save_dir}")
            else:
                torch.save(model.state_dict(), cfg["save_path"])
                print(f"  [v] Best custom model saved to {cfg['save_path']}")
                
            config_save_path = os.path.join("checkpoints", "config.json")
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)

    print(f"\n[v] Training complete. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
