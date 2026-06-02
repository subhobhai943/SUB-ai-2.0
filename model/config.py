# Model Hyperparameters
CONFIG = {
    "vocab_size": 20000,    # Reduced — actual unique tokens are ~18K
    "embed_dim": 512,       # Doubled for richer representations
    "num_heads": 8,
    "num_layers": 6,        # More layers for deeper understanding
    "ff_dim": 1024,         # Wider feed-forward for capacity
    "max_seq_len": 256,     # Longer context window
    "dropout": 0.1,
    "batch_size": 32,       # Larger batches for stable gradients
    "epochs": 50,           # More epochs for better convergence
    "lr": 1e-4,             # Lower LR for smoother training
    "save_path": "checkpoints/sub_ai.pt",
    "vocab_path": "checkpoints/vocab.json"
}
