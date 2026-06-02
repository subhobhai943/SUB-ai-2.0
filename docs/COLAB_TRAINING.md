# Training SUB-ai 2.0 on Google Colab (Tesla T4 GPU)

This guide explains how to fine-tune your **4B parameter class model** (e.g. `microsoft/Phi-3-mini-4k-instruct` or `Qwen/Qwen2.5-3B-Instruct`) on Google Colab using a free or paid **Tesla T4 GPU** (16GB VRAM), and transfer the fine-tuned adapter weights back to your local machine.

---

## Step 1: Open Google Colab and Setup Runtime
1. Go to [Google Colab](https://colab.research.google.com).
2. Create a new notebook.
3. Click on **Runtime** -> **Change runtime type**.
4. Select **T4 GPU** as the hardware accelerator and click **Save**.

---

## Step 2: Run Setup Cells in Colab

Create code cells in your Colab notebook and run them in sequence:

### Cell 1: Clone your Repository & Change Directory
If your repository is public, you can clone it directly. Otherwise, you can upload your project files as a ZIP using the Colab file browser and unzip them.

```python
# Option A: Clone from Github
!git clone https://github.com/YOUR_GITHUB_USERNAME/SUB-ai-2.0.git
%cd SUB-ai-2.0

# Option B: If you uploaded a ZIP file named SUB-ai-2.0.zip
# !unzip SUB-ai-2.0.zip
# %cd SUB-ai-2.0
```

### Cell 2: Install PyTorch with CUDA & Dependencies
Install the required libraries (including HuggingFace `transformers`, `peft` for LoRA, and `accelerate` for optimized device loading).

```python
!pip install -r requirements.txt
```

### Cell 3: Download HuggingFace Datasets
Run the importer script to download SQuAD, OpenAssistant conversations, TriviaQA, OpenBookQA, Alpaca, CommonsenseQA, ELI5, GSM8K, and Python code datasets.

```python
!python dataset/import_hf_datasets.py
```

### Cell 4: Assemble the Merged Dataset
Compile all downloaded datasets into the unified training format:

```python
!python dataset/build_dataset.py
```

### Cell 5: Train / Fine-tune the Model
Start the fine-tuning process. Since `model_mode` is set to `"pretrained"` in `model/config.py`, the training script will:
1. Load your chosen pre-trained 4B model (e.g., `microsoft/Phi-3-mini-4k-instruct`).
2. Wrap it with a low-rank adapter (LoRA) config.
3. Fine-tune it using half-precision (`fp16`) mixed training, which fits comfortably within the T4's 16GB VRAM.
4. Save the trained LoRA adapter weights under `checkpoints/pretrained_model/`.

```python
!python train.py
```

### Cell 6: Zip and Download Fine-Tuned Weights
Once training completes, zip the saved adapter weights and download them to your computer.

```python
import os
from google.colab import files

# Zip the fine-tuned adapter directory
!zip -r checkpoints_pretrained.zip checkpoints/pretrained_model checkpoints/config.json

# Download the zip file to your local computer
files.download('checkpoints_pretrained.zip')
```

---

## Step 3: Run Locally

1. Extract the downloaded `checkpoints_pretrained.zip` directly into your local `SUB-ai-2.0/` folder.
2. Verify that your directory structure looks like this:
   ```text
   SUB-ai-2.0/
   └── checkpoints/
       ├── config.json
       └── pretrained_model/
           ├── adapter_config.json
           ├── adapter_model.safetensors
           ├── tokenizer_config.json
           ├── tokenizer.json
           └── ...
   ```
3. Update `model/config.py` locally to match the `config.json` settings:
   - `"model_mode": "pretrained"`
   - `"pretrained_model_name": "microsoft/Phi-3-mini-4k-instruct"`
4. Run evaluation:
   ```bash
   python evaluate.py
   ```
5. Launch the local interactive chat and talk with your fine-tuned model:
   ```bash
   python chat.py
   ```
