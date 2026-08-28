# Model Hyperparameters
CONFIG = {
    # ── Mode Selection ─────────────────────────────────────────────
    # "custom" for scratch-built SUBaiModel, "pretrained" for fine-tuning
    "model_mode": "pretrained", 
    "pretrained_model_name": "Qwen/Qwen2.5-3B-Instruct", # 3B param model, fully compatible with latest transformers
    
    # ── LoRA Hyperparameters (only used if model_mode == "pretrained") ──
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Optimized for Qwen2.5

    # ── Custom Model Hyperparameters ─────────────────────────────────
    "vocab_size": 20000,    # Reduced — actual unique tokens are ~18K
    "embed_dim": 512,       # Doubled for richer representations
    "num_heads": 8,
    "num_layers": 6,        # More layers for deeper understanding
    "ff_dim": 1024,         # Wider feed-forward for capacity
    "dropout": 0.1,

    # ── Shared Training Settings ────────────────────────────────────
    "max_seq_len": 256,     # Longer context window
    "batch_size": 4,        # Lowered from 8 -- 8 was OOMing a T4 (14.56GB) even
                             # with gradient checkpointing enabled; grad accumulation
                             # below keeps the effective batch size at 32.
    "gradient_accumulation_steps": 8,
    "epochs": 3,            # Fine-tuning a pretrained model takes 3-5 epochs
    "lr": 2e-5,             # standard LR for LLM fine-tuning (2e-5)
    "save_path": "checkpoints/sub_ai.pt", # Standard save path
    "vocab_path": "checkpoints/vocab.json"
}
