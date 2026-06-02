import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from model.model import SUBaiModel
from model.config import CONFIG
from tqdm import tqdm

# ── Tokenizer ────────────────────────────────────────────────────────────────
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
        print(f"[✓] Vocabulary saved to {path}")

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

# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        with open(path, encoding="utf-8") as f:   # ✔ utf-8 fix
            data = json.load(f)
        self.samples = [tokenizer.encode(d["text"], max_len) for d in data]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:],  dtype=torch.long)
        return x, y

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    os.makedirs("checkpoints", exist_ok=True)

    # GPU setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[✓] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[!] CUDA not available. Using CPU.")
        print("    Fix: pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121")

    with open("data/processed/dataset.json", encoding="utf-8") as f:   # ✔ utf-8 fix
        raw = json.load(f)
    texts = [d["text"] for d in raw]

    tokenizer = SimpleTokenizer(cfg["vocab_size"])
    tokenizer.build_vocab(texts)
    tokenizer.save(cfg["vocab_path"])

    full_dataset = TextDataset("data/processed/dataset.json", tokenizer, cfg["max_seq_len"])
    val_size   = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"])

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
    print(f"[✓] Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [Train]"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1)).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        scheduler.step()

        print(f"  Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), cfg["save_path"])
            config_save_path = os.path.join(os.path.dirname(cfg["save_path"]), "config.json")
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)
            print(f"  [✓] Best model saved (val loss: {best_val_loss:.4f})")

    print(f"\n[✓] Training complete. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
