# Model Hyperparameters
CONFIG = {
    "vocab_size": 30000,    # Increased from 10000 to reduce <UNK> tokens
    "embed_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "ff_dim": 512,
    "max_seq_len": 128,
    "dropout": 0.1,
    "batch_size": 16,
    "epochs": 30,
    "lr": 3e-4,
    "save_path": "checkpoints/sub_ai.pt",
    "vocab_path": "checkpoints/vocab.json"
}
