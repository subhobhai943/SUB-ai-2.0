import json
import os

# Must be set before the CUDA context is created (i.e. before `import torch`).
# Reduces allocator fragmentation, which matters on a T4's tight 16GB budget.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

# -- Tokenizer (For Custom Mode) -----------------------------------------------
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

# -- Dataset (For Custom Mode) -------------------------------------------------
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

# -- Dataset (For Pretrained Fine-Tuning Mode) ---------------------------------
class PretrainedTextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
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
            attention_mask = enc["attention_mask"].squeeze(0)

            # Mask real padding positions via attention_mask, not by comparing
            # token ids -- pad_token_id == eos_token_id for several models
            # (e.g. Phi-3), which would otherwise also mask the real trailing
            # EOS and stop the model from ever learning to end a sequence.
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.samples.append((input_ids, attention_mask, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, attention_mask, labels = self.samples[idx]
        return input_ids, attention_mask, labels

# -- Main ----------------------------------------------------------------------
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

    # Gradient accumulation config
    grad_accum_steps = cfg.get("gradient_accumulation_steps", 4)

    if model_mode == "pretrained":
        if not HF_AVAILABLE:
            raise ImportError(
                "Pretrained mode requires huggingface libraries. "
                "Please install transformers, peft, and accelerate."
            )

        model_name = cfg["pretrained_model_name"]
        print(f"[*] Loading pre-trained tokenizer & model for: {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model -- use `dtype=` instead of deprecated `torch_dtype=`
        # Never chain .to(device) on the same line as from_pretrained
        if device.type == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            model = model.to(device)

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

        if device.type == "cuda":
            # peft mirrors the base layer's dtype onto new LoRA adapter weights
            # (or, with no LoRA, every trainable param is already fp16 from
            # from_pretrained). GradScaler.unscale_() raises "Attempting to
            # unscale FP16 gradients" unless trainable params are fp32, so
            # upcast only the trainable weights; frozen fp16 base weights stay
            # fp16 and autocast reconciles the mixed dtypes during forward/backward.
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        # Gradient checkpointing trades compute for activation memory -- without it,
        # every layer's activations stay resident through the whole backward pass,
        # which is what was blowing past the T4's 16GB. enable_input_require_grads()
        # is required alongside it: the base model is frozen, so nothing upstream of
        # the LoRA adapters would otherwise carry requires_grad=True into the
        # checkpointed segments, breaking recomputation. use_cache is disabled since
        # KV-caching (a generation-time optimization) is incompatible with it and
        # only wastes memory during training.
        print("[*] Enabling gradient checkpointing to reduce activation memory...")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

        full_dataset = PretrainedTextDataset(
            "data/processed/dataset.json", tokenizer, cfg["max_seq_len"]
        )

        # Safely get vocab_size -- PEFT wrapping can break model.config access
        try:
            vocab_size = model.config.vocab_size
        except AttributeError:
            try:
                vocab_size = model.base_model.config.vocab_size
            except AttributeError:
                vocab_size = len(tokenizer)

    else:
        # Custom Mode (Word-level Tokenizer & Scratch Transformer)
        tokenizer = SimpleTokenizer(cfg["vocab_size"])
        tokenizer.build_vocab(texts)
        tokenizer.save(cfg["vocab_path"])

        full_dataset = TextDataset(
            "data/processed/dataset.json", tokenizer, cfg["max_seq_len"]
        )
        vocab_size = cfg["vocab_size"]

        model = SUBaiModel(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            ff_dim=cfg["ff_dim"],
            max_seq_len=cfg["max_seq_len"],
            dropout=cfg["dropout"]
        )
        model = model.to(device)

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

    # For custom mode we still need a manual loss criterion
    if model_mode != "pretrained":
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float("inf")

    # Mixed precision scaler -- use non-deprecated torch.amp.GradScaler
    use_amp = (device.type == "cuda" and model_mode == "pretrained")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[*] Gradient accumulation steps: {grad_accum_steps}")
    print(f"[*] Effective batch size: {cfg['batch_size'] * grad_accum_steps}")

    for epoch in range(1, cfg["epochs"] + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [Train]")
        ):
            # Use non-deprecated torch.amp.autocast; only enable on CUDA
            with torch.amp.autocast("cuda", enabled=use_amp):
                if model_mode == "pretrained":
                    # HF models: pass input_ids, attention_mask and labels, get loss directly.
                    # attention_mask matters for models that default to left-padding
                    # (e.g. Phi-3) -- without it, real tokens would attend into
                    # leading pad-token garbage.
                    x, attn_mask, y = batch
                    x, attn_mask, y = x.to(device), attn_mask.to(device), y.to(device)
                    outputs = model(input_ids=x, attention_mask=attn_mask, labels=y)
                    loss = outputs.loss
                else:
                    # Custom model: returns logits directly
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            # Step optimizer every grad_accum_steps or at the last batch
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

        # -- Validate --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if model_mode == "pretrained":
                        x, attn_mask, y = batch
                        x, attn_mask, y = x.to(device), attn_mask.to(device), y.to(device)
                        outputs = model(input_ids=x, attention_mask=attn_mask, labels=y)
                        loss = outputs.loss
                    else:
                        x, y = batch
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        loss = criterion(
                            logits.reshape(-1, vocab_size), y.reshape(-1)
                        )
                val_loss += loss.item()

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
                json.dump({**cfg, "best_val_loss": best_val_loss}, f, indent=4)

    print(f"\n[v] Training complete. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
