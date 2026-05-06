import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from model.model import SUBaiModel
from model.config import CONFIG
from tqdm import tqdm

# ── Tokenizer ──────────────────────────────────────────────
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

    def encode(self, text, max_len):
        tokens = [self.word2idx.get(w, 1) for w in text.lower().split()]
        tokens = tokens[:max_len]
        tokens += [0] * (max_len - len(tokens))  # pad
        return tokens

# ── Dataset ────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        with open(path) as f:
            data = json.load(f)
        self.samples = [tokenizer.encode(d["text"], max_len) for d in data]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.samples[idx][1:],  dtype=torch.long)
        return x, y

# ── Main ───────────────────────────────────────────────────
def main():
    cfg = CONFIG
    os.makedirs("checkpoints", exist_ok=True)

    # Load processed data
    with open("data/processed/dataset.json") as f:
        raw = json.load(f)
    texts = [d["text"] for d in raw]

    tokenizer = SimpleTokenizer(cfg["vocab_size"])
    tokenizer.build_vocab(texts)

    dataset = TextDataset("data/processed/dataset.json", tokenizer, cfg["max_seq_len"])
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[✓] Using device: {device}")

    model = SUBaiModel(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)                      # (B, T, vocab)
            loss = criterion(logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), cfg["save_path"])
    print(f"[✓] Model saved to {cfg['save_path']}")

if __name__ == "__main__":
    main()
