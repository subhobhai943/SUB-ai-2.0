# SUB-ai 2.0

SUB-ai 2.0 is a personal AI research and engineering project focused on building a language model pipeline from scratch using a custom JSON dataset, a lightweight tokenizer, a simple Transformer-based architecture, and a clean training workflow.

This repository is designed as a learning-first project. The goal is not to compete with large production LLMs, but to understand the full stack of model building: collecting data, formatting a dataset, tokenizing text, defining a model, training it, evaluating it, and improving it over time.

## Project Goals

- Build an AI model from scratch instead of only fine-tuning an existing one.
- Create and maintain a custom dataset in JSON format.
- Keep the codebase understandable for students and independent builders.
- Experiment with model architecture, tokenization, and training strategies.
- Grow from a small prototype into a more capable conversational model.

## Current Features

- Custom JSON dataset with input/output pairs.
- Dataset builder that converts raw pairs into training-ready text format.
- Lightweight word-level tokenizer.
- Transformer encoder based model in PyTorch.
- Training script with batching, loss calculation, and checkpoint saving.
- Evaluation script for simple response generation.

## Repository Structure

```text
SUB-ai-2.0/
├── data/
│   ├── raw/
│   │   └── data.json
│   └── processed/
│       └── dataset.json
├── dataset/
│   └── build_dataset.py
├── model/
│   ├── config.py
│   └── model.py
├── docs/
│   ├── ROADMAP.md
│   ├── DATASET_GUIDE.md
│   ├── TRAINING_GUIDE.md
│   └── PROJECT_VISION.md
├── checkpoints/
├── train.py
├── evaluate.py
├── requirements.txt
├── LICENSE
└── README.md
```

## How the Dataset Works

The raw dataset uses a simple JSON array of objects:

```json
[
  {
    "input": "What is AI?",
    "output": "AI stands for Artificial Intelligence."
  },
  {
    "input": "What is Python?",
    "output": "Python is a popular programming language."
  }
]
```

The dataset builder converts each pair into a conversational training string such as:

```text
User: What is AI?
Bot: AI stands for Artificial Intelligence.
```

This makes the project simple to edit manually while still creating a consistent format for training.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/subhobhai943/SUB-ai-2.0
cd SUB-ai-2.0
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the processed dataset

```bash
python dataset/build_dataset.py
```

### 4. Train the model

```bash
python train.py
```

### 5. Evaluate the model

```bash
python evaluate.py
```

## Training Workflow

1. Add or edit examples in `data/raw/data.json`.
2. Run the dataset builder to prepare training text.
3. Train the model using the processed dataset.
4. Save model weights into the `checkpoints/` directory.
5. Evaluate outputs and improve the dataset or model configuration.

## Recommended Next Improvements

- Replace the word-level tokenizer with subword tokenization.
- Add train/validation dataset splitting.
- Add proper autoregressive decoding.
- Store tokenizer vocabulary to disk.
- Add experiment tracking and metrics visualization.
- Expand the dataset to hundreds or thousands of high-quality examples.
- Support importing public data from Hugging Face datasets.

## Important Notes

- This is an educational and experimental repository.
- The current model is intentionally small and simple.
- Output quality depends heavily on dataset quality, size, and training duration.
- A small custom model will not behave like GPT-scale systems without major scaling.

## Documentation

Detailed documentation is available in the `docs/` directory:

- `docs/ROADMAP.md` — full long-term development plan
- `docs/DATASET_GUIDE.md` — how to design and improve the dataset
- `docs/TRAINING_GUIDE.md` — training workflow, tuning, and troubleshooting
- `docs/PROJECT_VISION.md` — long-term purpose and direction

## License

This repository uses a custom personal project license. See the `LICENSE` file for usage terms.

## Author

Created by Subhobhai as a personal AI building project for learning, experimentation, and long-term improvement.
