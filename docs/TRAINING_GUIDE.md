# TRAINING GUIDE

This guide explains how training currently works and how to improve it.

## Current Pipeline

1. Edit `data/raw/data.json`.
2. Run `python dataset/build_dataset.py`.
3. Run `python train.py`.
4. Run `python evaluate.py`.

## What the Training Script Does

- Loads processed training text.
- Builds a simple vocabulary from the dataset.
- Encodes text into token IDs.
- Trains a Transformer-based model.
- Saves learned weights to a checkpoint.

## Known Limitations

- The tokenizer is basic.
- There is no train/validation split yet.
- Generation is still simplistic.
- Small datasets will limit output quality.

## Recommended Improvements

- Add validation loss.
- Save tokenizer state.
- Improve decoding logic.
- Add checkpoint resume support.
- Add better metrics and logs.
