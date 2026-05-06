# ROADMAP

This roadmap describes the intended direction for SUB-ai 2.0 from its current prototype stage to a more capable personal language model project.

## Phase 1: Stabilize the Foundation

### Goals

- Keep the repository organized and understandable.
- Ensure the current training pipeline runs end to end without errors.
- Make dataset creation and editing simple.
- Establish a repeatable workflow for experimentation.

### Tasks

- Finalize project directory structure.
- Improve comments, naming, and readability in core files.
- Confirm that `build_dataset.py`, `train.py`, and `evaluate.py` work in sequence.
- Save checkpoints reliably after training.
- Add better error handling for missing files and empty datasets.
- Document setup steps clearly.

### Success Criteria

- A new user can clone the repo and run it successfully.
- The dataset can be edited without touching training code.
- Training produces a saved checkpoint.
- Evaluation runs from a saved model.

## Phase 2: Improve Dataset Quality

### Goals

- Expand the dataset beyond tiny examples.
- Improve consistency in answer style.
- Reduce noisy, repetitive, and low-value samples.
- Create a reusable format for future dataset growth.

### Tasks

- Expand to 500+ curated examples.
- Group examples by category such as AI, science, coding, math, and conversation.
- Standardize punctuation, tone, and response length.
- Remove contradictory or low-quality responses.
- Add a validation script for dataset sanity checks.
- Consider adding metadata fields like `category`, `difficulty`, or `source_type`.

### Success Criteria

- Dataset becomes large enough to produce better responses.
- Samples follow a consistent structure.
- Data quality improves model stability during training.

## Phase 3: Upgrade Tokenization

### Goals

- Move beyond the current basic word-level tokenizer.
- Reduce unknown tokens.
- Improve handling of punctuation and unseen words.

### Tasks

- Save tokenizer vocabulary as a file.
- Add support for punctuation-aware tokenization.
- Experiment with subword tokenization such as BPE or WordPiece.
- Compare vocabulary size and coverage across tokenizer methods.

### Success Criteria

- Fewer unknown tokens during training and evaluation.
- Better generalization to unseen prompts.
- Cleaner generated text.

## Phase 4: Improve Model Architecture

### Goals

- Make the model more suitable for language generation.
- Improve output quality and training efficiency.
- Keep architecture understandable enough for learning.

### Tasks

- Replace or extend the current encoder-only approach with a causal language modeling design.
- Add masking appropriate for next-token prediction.
- Add layer normalization and other stability improvements if needed.
- Experiment with embedding size, number of heads, number of layers, and feedforward size.
- Add configuration presets for small, medium, and experimental models.

### Success Criteria

- Model generates more coherent responses.
- Training becomes more aligned with language modeling objectives.
- Architecture remains maintainable and well documented.

## Phase 5: Training Improvements

### Goals

- Make training more informative, stable, and resumable.
- Track progress better across experiments.

### Tasks

- Add train/validation split.
- Log validation loss after each epoch.
- Save the best checkpoint, not only the last one.
- Add early stopping.
- Add resume-from-checkpoint support.
- Add gradient clipping.
- Add learning rate scheduling.
- Record experiment settings automatically.

### Success Criteria

- Training is more stable.
- Results are easier to compare across runs.
- Failed runs can be resumed without starting over.

## Phase 6: Evaluation and Testing

### Goals

- Move beyond simple manual prompt testing.
- Build a more reliable way to judge model progress.

### Tasks

- Create a fixed evaluation prompt set.
- Add qualitative review categories such as correctness, fluency, and helpfulness.
- Add quantitative metrics where appropriate.
- Compare outputs across checkpoints.
- Write example-based regression tests for dataset and model behavior.

### Success Criteria

- Model quality can be measured more systematically.
- Regressions are easier to detect.
- Improvements can be justified with evidence.

## Phase 7: Tooling and Automation

### Goals

- Reduce manual effort.
- Make the project easier to scale.

### Tasks

- Add scripts for dataset import and conversion.
- Add support for selected Hugging Face datasets.
- Create CLI utilities for common actions.
- Add linting and formatting tools.
- Add GitHub Actions for checks and validation.

### Success Criteria

- Common tasks become faster and less error-prone.
- Project becomes easier to maintain as complexity grows.

## Phase 8: Long-Term Vision

### Goals

- Turn the repo into a serious personal AI lab project.
- Build a stronger model and stronger understanding over time.

### Possibilities

- Domain-specific assistant versions.
- Better conversational memory design.
- Retrieval-augmented answering.
- Distillation from stronger open models.
- Fine-tuning on personal datasets.
- Deployment through API or web interface.

## Priorities Right Now

The most important immediate priorities are:

1. Improve dataset quality and size.
2. Upgrade tokenization.
3. Improve the model architecture for language generation.
4. Add validation and better training metrics.
5. Keep documentation and structure clean as the project grows.
