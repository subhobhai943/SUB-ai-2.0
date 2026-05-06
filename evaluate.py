import json
import torch
from model.model import SUBaiModel
from model.config import CONFIG
from train import SimpleTokenizer

def generate(model, tokenizer, prompt, max_new=20, device="cpu"):
    model.eval()
    cfg = CONFIG
    tokens = tokenizer.encode(prompt, cfg["max_seq_len"])
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)  # (1, T, vocab)
    preds = logits.argmax(dim=-1).squeeze().tolist()
    words = [tokenizer.idx2word.get(t, "<UNK>") for t in preds[:max_new]]
    return " ".join(words)

if __name__ == "__main__":
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/dataset.json") as f:
        raw = json.load(f)
    texts = [d["text"] for d in raw]

    tokenizer = SimpleTokenizer(cfg["vocab_size"])
    tokenizer.build_vocab(texts)

    model = SUBaiModel(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        max_seq_len=cfg["max_seq_len"]
    ).to(device)
    model.load_state_dict(torch.load(cfg["save_path"], map_location=device))

    prompt = "User: What is AI?"
    response = generate(model, tokenizer, prompt, device=device)
    print(f"Prompt : {prompt}")
    print(f"Response: {response}")
