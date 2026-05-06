import json
import torch
from model.model import SUBaiModel
from model.config import CONFIG
from train import SimpleTokenizer


def generate(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8, top_k=40, device="cpu"):
    model.eval()
    cfg = CONFIG

    tokens = [tokenizer.word2idx.get("<BOS>", 2)]
    tokens += [tokenizer.word2idx.get(w, 1) for w in prompt.lower().split()]

    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.size(1) >= cfg["max_seq_len"]:
                break
            logits = model(x)                  # (1, T, vocab)
            next_logits = logits[0, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[-1]] = -float("inf")

            probs     = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.word2idx.get("<EOS>", 3):
                break

            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    generated = x[0].tolist()[len(tokens):]
    words = [tokenizer.idx2word.get(t, "<UNK>") for t in generated]
    return " ".join(words).strip()


if __name__ == "__main__":
    cfg = CONFIG

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[✓] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[!] Using CPU")

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

    model.load_state_dict(torch.load(cfg["save_path"], map_location=device))
    print("[✓] Model loaded\n")

    prompts = [
        "User: What is AI?\nBot:",
        "User: What is machine learning?\nBot:",
        "User: What is Python?\nBot:",
        "User: Tell me a fact.\nBot:",
        "User: What is 12 plus 8?\nBot:",
    ]

    for prompt in prompts:
        response = generate(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8, top_k=40, device=device)
        print(f"Prompt  : {prompt}")
        print(f"Response: {response}")
        print("-" * 60)
