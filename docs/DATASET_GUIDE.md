# DATASET GUIDE

This document explains how to create, maintain, and improve the custom dataset used in SUB-ai 2.0.

## Current Format

The current raw dataset is a JSON array of input/output objects.

```json
[
  {
    "input": "What is machine learning?",
    "output": "Machine learning is a field of AI where systems learn patterns from data."
  }
]
```

## Good Dataset Practices

- Keep inputs clear and natural.
- Keep outputs correct, readable, and consistent in style.
- Avoid duplicate samples unless the wording meaningfully differs.
- Avoid contradictory answers.
- Avoid extremely long answers in a small model dataset.
- Keep formatting consistent.

## Suggested Categories

- AI and machine learning
- Programming and software
- Math and science
- General knowledge
- Daily conversation
- Personal assistant style help

## Future Improvements

- Add metadata like category and difficulty.
- Create separate train and validation files.
- Build an automatic data cleaning script.
- Import and adapt public datasets carefully.
