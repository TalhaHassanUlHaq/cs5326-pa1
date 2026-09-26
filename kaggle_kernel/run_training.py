"""Kaggle GPU Automated Training Script for CS 5326 PA1."""

import math
import os
import subprocess
import sys


def run_command(command: str, cwd: str | None = None) -> None:
    print(f"\n=======================================================")
    print(f"Executing: {command}")
    print(f"Directory: {cwd or os.getcwd()}")
    print(f"=======================================================\n", flush=True)
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {command}")


def main() -> None:
    print(">>> 1. Checking GPU Accelerator:")
    run_command("nvidia-smi")

    repo_dir = "/kaggle/working/cs5326-pa1"
    if not os.path.exists(repo_dir):
        print("\n>>> 2. Cloning Repository:")
        run_command(f"git clone https://github.com/TalhaHassanUlHaq/cs5326-pa1.git {repo_dir}")
    else:
        print("\n>>> 2. Pulling Latest Code:")
        run_command("git pull", cwd=repo_dir)

    print("\n>>> 3. Setting up uv and Dependencies:")
    run_command("curl -LsSf https://astral.sh/uv/install.sh | sh")
    os.environ["PATH"] = f"/root/.local/bin:{os.environ.get('PATH', '')}"

    run_command("uv sync --frozen", cwd=repo_dir)

    print("\n>>> 4. Verifying Public Test Suite:")
    run_command("uv run pytest", cwd=repo_dir)

    print("\n>>> 5. Downloading TinyStories Dataset from Hugging Face:")
    run_command(
        "uv run hf download alooboii/pa1-tinystories "
        "metadata.json tokenizer/tokenizer.json "
        "data/train.bin data/validation.bin "
        "--repo-type dataset "
        "--local-dir data/tinystories",
        cwd=repo_dir,
    )

    print("\n>>> 6. Launching Transformer LM Training (10,000 steps on GPU):")
    train_cmd = (
        "uv run python src/train.py "
        "--train_data data/tinystories/data/train.bin "
        "--val_data data/tinystories/data/validation.bin "
        "--batch_size 16 "
        "--sequence_length 256 "
        "--gradient_accumulation_steps 16 "
        "--num_steps 10000 "
        "--learning_rate_max 3e-4 "
        "--learning_rate_min 3e-5 "
        "--warmup_steps 200 "
        "--cosine_steps 9999 "
        "--weight_decay 0.1 "
        "--max_grad_norm 1.0 "
        "--eval_interval 500 "
        "--log_interval 100 "
        "--checkpoint_interval 1000 "
        "--num_validation_batches 100 "
        "--checkpoint_dir /kaggle/working/checkpoints "
        "--export_model_path /kaggle/working/final_model.pt "
        "--device cuda"
    )
    run_command(train_cmd, cwd=repo_dir)

    print("\n>>> 7. Running Inference & Decoding Analysis (for REPORT.md):")
    sample_script = """
import torch
from tokenizers import Tokenizer
from src.model import TransformerLM
from src.generate import generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerLM(
    vocab_size=8192, context_length=256, d_model=512,
    num_layers=4, n_q_heads=16, n_kv_heads=4, d_ff=1344,
    rope_theta=10000.0, norm_eps=1e-5, device=device
)
state = torch.load("/kaggle/working/final_model.pt", weights_only=True)
model.load_state_dict({k: v.to(device).float() for k, v in state.items()})
model.eval()

tokenizer = Tokenizer.from_file("data/tinystories/tokenizer/tokenizer.json")
prompts = [
    "Once upon a time, there was a little girl named Lily.",
    "One day, Tom and Mia found a big box in the garden.",
]

configs = [
    {"temp": 0.3, "top_p": 1.0, "label": "Low Temp (0.3)"},
    {"temp": 0.7, "top_p": 0.9, "label": "Balanced (0.7, top-p 0.9)"},
    {"temp": 1.0, "top_p": 0.95, "label": "Creative (1.0, top-p 0.95)"},
]

with open("/kaggle/working/generated_samples.txt", "w", encoding="utf-8") as f:
    for prompt in prompts:
        prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False).ids, device=device)
        for cfg in configs:
            out = generate(
                model, prompt_ids, max_new_tokens=120, context_length=256,
                temperature=cfg["temp"], top_p=cfg["top_p"], eot_token_id=0
            )
            text = tokenizer.decode(out.tolist(), skip_special_tokens=False)
            header = f"=== Prompt: '{prompt}' | Setting: {cfg['label']} ==="
            print(header)
            print(text)
            print("-" * 50)
            f.write(header + "\\n" + text + "\\n\\n")

print("Generated samples written to /kaggle/working/generated_samples.txt")
"""
    with open(os.path.join(repo_dir, "run_samples.py"), "w", encoding="utf-8") as f:
        f.write(sample_script)

    run_command("uv run python run_samples.py", cwd=repo_dir)

    print("\n>>> Training and evaluation complete! Final artifacts ready in /kaggle/working/")


if __name__ == "__main__":
    main()
