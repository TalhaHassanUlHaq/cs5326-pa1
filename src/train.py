"""End-to-End Training Loop for the Modern Transformer LM on TinyStories."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path when invoked directly as python src/train.py
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.checkpoint import load_checkpoint, save_checkpoint
from src.data import get_batch, load_token_array
from src.model import TransformerLM
from src.optim import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Modern Transformer LM on TinyStories")
    parser.add_argument("--train_data", type=str, default="data/tinystories/data/train.bin")
    parser.add_argument("--val_data", type=str, default="data/tinystories/data/validation.bin")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--learning_rate_max", type=float, default=3e-4)
    parser.add_argument("--learning_rate_min", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--cosine_steps", type=int, default=9999)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--num_validation_batches", type=int, default=100)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--export_model_path", type=str, default="final_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate_validation(
    model: TransformerLM,
    val_tokens,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    val_generator: torch.Generator,
    num_batches: int,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0

    with torch.inference_mode():
        for _ in range(num_batches):
            x, y = get_batch(val_tokens, batch_size, sequence_length, device, val_generator)
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()

    model.train(was_training)
    return total_loss / num_batches


def main() -> None:
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)

    # 1. Initialize fixed assignment architecture (19.27M parameters)
    model = TransformerLM(
        vocab_size=8192,
        context_length=args.sequence_length,
        d_model=512,
        num_layers=4,
        n_q_heads=16,
        n_kv_heads=4,
        d_ff=1344,
        rope_theta=10000.0,
        norm_eps=1e-5,
        device=device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized Transformer LM with {total_params:,} parameters (target: 19,272,192)")

    # 2. Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate_max,
        betas=tuple(args.betas),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # 3. Independent RNG generators for reproducibility
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    val_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    next_step = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 4. Resume from checkpoint if provided
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        next_step = load_checkpoint(
            args.resume_checkpoint, model, optimizer, train_generator, val_generator
        )
        print(f"Resumed at next_step={next_step}")

    # 5. Load token data memory maps
    print(f"Loading training data from {args.train_data}...")
    train_tokens = load_token_array(args.train_data)
    print(f"Training tokens: {len(train_tokens):,}")

    print(f"Loading validation data from {args.val_data}...")
    val_tokens = load_token_array(args.val_data)
    print(f"Validation tokens: {len(val_tokens):,}")

    # 6. Training Loop (Section 5.3)
    print(f"Starting training from step {next_step} to {args.num_steps}...")
    for step in range(next_step, args.num_steps):
        model.train()
        lr = get_lr_cosine_schedule(
            step,
            learning_rate_max=args.learning_rate_max,
            learning_rate_min=args.learning_rate_min,
            warmup_steps=args.warmup_steps,
            cosine_steps=args.cosine_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        train_loss = 0.0

        for _ in range(args.gradient_accumulation_steps):
            x, y = get_batch(
                train_tokens,
                args.batch_size,
                args.sequence_length,
                device,
                generator=train_generator,
            )
            logits = model(x)
            microbatch_loss = cross_entropy(logits, y)
            (microbatch_loss / args.gradient_accumulation_steps).backward()
            train_loss += microbatch_loss.detach().item()

        train_loss /= args.gradient_accumulation_steps
        grad_norm = gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        completed_steps = step + 1
        final_step = (completed_steps == args.num_steps)
        should_validate = final_step or (completed_steps % args.eval_interval == 0)
        should_log = should_validate or (completed_steps % args.log_interval == 0)
        should_checkpoint = final_step or (completed_steps % args.checkpoint_interval == 0)

        val_loss = None
        if should_validate:
            val_loss = evaluate_validation(
                model,
                val_tokens,
                args.batch_size,
                args.sequence_length,
                device,
                val_generator,
                args.num_validation_batches,
            )

        if should_log:
            val_str = f" | Val Loss: {val_loss:.4f} | Val PPL: {math.exp(val_loss):.2f}" if val_loss is not None else ""
            print(
                f"Step {completed_steps:5d}/{args.num_steps} | "
                f"Train Loss: {train_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Grad Norm: {grad_norm:.3f}"
                f"{val_str}"
            )

        if should_checkpoint:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{completed_steps}.pt")
            save_checkpoint(
                model,
                optimizer,
                completed_steps,
                train_generator,
                val_generator,
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    # 7. Standardized Final Validation Evaluation (Section 7.3)
    print("\nRunning standardized final evaluation (seed=42, 100 validation batches)...")
    final_val_generator = torch.Generator(device="cpu").manual_seed(42)
    final_val_loss = evaluate_validation(
        model,
        val_tokens,
        batch_size=16,
        sequence_length=256,
        device=device,
        val_generator=final_val_generator,
        num_batches=100,
    )
    final_ppl = math.exp(final_val_loss)
    print(f"Final Validation Loss: {final_val_loss:.4f} nats/token")
    print(f"Final Validation Perplexity (PPL): {final_ppl:.2f}")

    # 8. Export Final Model Artifact (Section 7.4)
    print(f"\nExporting FP16 CPU final model artifact to {args.export_model_path}...")
    export_state = {
        name: tensor.detach().cpu().to(torch.float16)
        if tensor.is_floating_point()
        else tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(export_state, args.export_model_path)
    total_exported = sum(t.numel() for t in export_state.values())
    print(f"Successfully saved {args.export_model_path} with {total_exported:,} values.")


if __name__ == "__main__":
    main()
