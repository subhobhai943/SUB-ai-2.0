"""release_gguf.py
Post-training pipeline: merge the fine-tuned LoRA adapter into the base model,
convert to GGUF, quantize with llama.cpp, and publish the result as a new
tagged GitHub Release.

Run this on the same Colab session right after `python train.py`:

    python scripts/release_gguf.py

Requires a GitHub Personal Access Token with "Contents: write" on this repo
(classic PAT needs the "repo" scope). Supply it as either:
  - a Colab secret named GITHUB_TOKEN (Secrets panel in the left sidebar,
    then grant this notebook access) -- preferred, never touches shell history
  - the GITHUB_TOKEN environment variable, e.g.:
        import os; os.environ["GITHUB_TOKEN"] = "ghp_..."

Optional environment variables:
  GGUF_QUANT     -- llama.cpp quant type (default: Q4_K_M)
  RELEASE_TAG    -- release tag (default: train-<UTC timestamp>)
  GITHUB_REPOSITORY -- "owner/repo" override (default: parsed from `git remote origin`)
  KEEP_INTERMEDIATES -- set to "1" to keep the merged HF model and f16 gguf
                        (default: deleted after quantization to save disk)
"""

import os
import subprocess
import sys
import shutil
import json
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from model.config import CONFIG

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
ADAPTER_DIR = os.path.join(CHECKPOINTS_DIR, "pretrained_model")
MERGED_DIR = os.path.join(CHECKPOINTS_DIR, "merged_model")
GGUF_DIR = os.path.join(CHECKPOINTS_DIR, "gguf")
LLAMA_CPP_DIR = os.path.join(REPO_ROOT, "llama.cpp")

QUANT_TYPE = os.environ.get("GGUF_QUANT", "Q4_K_M")
KEEP_INTERMEDIATES = os.environ.get("KEEP_INTERMEDIATES", "0") == "1"


def run(cmd, cwd=None):
    print(f"[$] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def load_config():
    config_path = os.path.join(CHECKPOINTS_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return CONFIG


def get_token():
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
        if token:
            return token
    except Exception:
        pass
    raise RuntimeError(
        "No GitHub token found. Set the GITHUB_TOKEN environment variable, "
        "or add a Colab secret named GITHUB_TOKEN (Secrets panel, left sidebar) "
        "and grant this notebook access to it."
    )


def get_repo_slug():
    env_slug = os.environ.get("GITHUB_REPOSITORY")
    if env_slug:
        return env_slug
    url = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True
    ).stdout.strip()
    # Handles both https://github.com/owner/repo.git and git@github.com:owner/repo.git
    slug = url.split("github.com")[-1].lstrip(":/").removesuffix(".git")
    return slug


def merge_adapter(cfg, device):
    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError(
            f"{ADAPTER_DIR} not found -- run `python train.py` first."
        )

    base_model_name = cfg["pretrained_model_name"]
    is_adapter = os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json"))

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if is_adapter:
        print(f"[*] Loading base model {base_model_name} and merging LoRA adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=dtype, trust_remote_code=True, attn_implementation="eager"
        ).to(device)
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        model = model.merge_and_unload()
    else:
        print("[*] checkpoints/pretrained_model is a full fine-tune (no adapter) -- using it directly.")
        model = AutoModelForCausalLM.from_pretrained(
            ADAPTER_DIR, dtype=dtype, trust_remote_code=True, attn_implementation="eager"
        ).to(device)

    os.makedirs(MERGED_DIR, exist_ok=True)
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"[v] Merged model saved to {MERGED_DIR}")


def ensure_llama_cpp():
    if not os.path.isdir(LLAMA_CPP_DIR):
        print("[*] Cloning llama.cpp...")
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", LLAMA_CPP_DIR])

    convert_req = os.path.join(LLAMA_CPP_DIR, "requirements", "requirements-convert_hf_to_gguf.txt")
    fallback_req = os.path.join(LLAMA_CPP_DIR, "requirements.txt")
    req_file = convert_req if os.path.exists(convert_req) else fallback_req
    print("[*] Installing llama.cpp conversion dependencies...")
    run([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])


def convert_to_gguf():
    os.makedirs(GGUF_DIR, exist_ok=True)
    f16_path = os.path.join(GGUF_DIR, "model-f16.gguf")
    print("[*] Converting merged model to GGUF (f16)...")
    run([
        sys.executable, os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py"),
        MERGED_DIR, "--outfile", f16_path, "--outtype", "f16",
    ])
    return f16_path


def build_quantize_binary():
    build_dir = os.path.join(LLAMA_CPP_DIR, "build")
    quantize_bin = os.path.join(build_dir, "bin", "llama-quantize")
    if os.path.exists(quantize_bin):
        return quantize_bin

    print("[*] Building llama-quantize (this takes a few minutes)...")
    run(["cmake", "-B", build_dir, "-S", LLAMA_CPP_DIR, "-DCMAKE_BUILD_TYPE=Release"])
    nproc = str(os.cpu_count() or 2)
    run(["cmake", "--build", build_dir, "--config", "Release", "-j", nproc, "--target", "llama-quantize"])

    if not os.path.exists(quantize_bin):
        raise FileNotFoundError(f"Expected {quantize_bin} after build but it wasn't produced.")
    return quantize_bin


def quantize(f16_path, quant_type):
    quantize_bin = build_quantize_binary()
    out_path = os.path.join(GGUF_DIR, f"model-{quant_type}.gguf")
    print(f"[*] Quantizing to {quant_type}...")
    run([quantize_bin, f16_path, out_path, quant_type])
    return out_path


def create_release(token, repo_slug, tag, name, body):
    import requests

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.post(
        f"https://api.github.com/repos/{repo_slug}/releases",
        headers=headers,
        json={"tag_name": tag, "name": name, "body": body, "draft": False, "prerelease": False},
        timeout=30,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to create release ({resp.status_code}): {resp.text}")
    return resp.json()


def upload_asset(token, upload_url_template, file_path):
    import requests

    filename = os.path.basename(file_path)
    upload_url = upload_url_template.split("{")[0] + f"?name={filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/octet-stream",
    }
    print(f"[*] Uploading {filename} ({os.path.getsize(file_path) / 1e9:.2f} GB) to the release...")
    with open(file_path, "rb") as f:
        resp = requests.post(upload_url, headers=headers, data=f, timeout=None)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to upload asset ({resp.status_code}): {resp.text}")
    return resp.json()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()

    token = get_token()
    repo_slug = get_repo_slug()

    merge_adapter(cfg, device)
    ensure_llama_cpp()
    f16_path = convert_to_gguf()
    quant_path = quantize(f16_path, QUANT_TYPE)

    if not KEEP_INTERMEDIATES:
        print("[*] Cleaning up intermediates to save disk (set KEEP_INTERMEDIATES=1 to keep)...")
        shutil.rmtree(MERGED_DIR, ignore_errors=True)
        if os.path.exists(f16_path):
            os.remove(f16_path)

    tag = os.environ.get("RELEASE_TAG", f"train-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    name = f"SUB-ai 2.0 -- {tag}"
    body_lines = [
        f"Base model: `{cfg.get('pretrained_model_name', 'unknown')}`",
        f"Quantization: `{QUANT_TYPE}`",
        f"Epochs: {cfg.get('epochs', 'unknown')}",
        f"LoRA: r={cfg.get('lora_r')}, alpha={cfg.get('lora_alpha')}" if cfg.get("use_lora") else "LoRA: disabled (full fine-tune)",
    ]
    if "best_val_loss" in cfg:
        body_lines.append(f"Best validation loss: {cfg['best_val_loss']:.4f}")
    body = "\n".join(body_lines)

    print(f"[*] Creating release {tag} on {repo_slug}...")
    release = create_release(token, repo_slug, tag, name, body)
    upload_asset(token, release["upload_url"], quant_path)

    print(f"\n[v] Release published: {release['html_url']}")


if __name__ == "__main__":
    main()
