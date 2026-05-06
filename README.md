# SUB-ai 2.0

A custom AI model built from scratch with a custom JSON dataset.

## Setup
```bash
pip install -r requirements.txt
```

## Steps
1. Add your data to `data/raw/data.json`
2. Run `python dataset/build_dataset.py` to process it
3. Run `python train.py` to train the model
4. Run `python evaluate.py` to test it
