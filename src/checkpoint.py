"""Training Checkpointing: Model, Optimizer, and Generator RNG States.

===============================================================================
WHAT NEEDS TO BE IMPLEMENTED:
===============================================================================
1. save_checkpoint(
       model: torch.nn.Module,
       optimizer: torch.optim.Optimizer,
       next_step: int,
       train_generator: torch.Generator,
       val_generator: torch.Generator,
       out: str | Path | BinaryIO,
   ) -> None:
   - Assemble state dictionary containing:
     * "model": model.state_dict()
     * "optimizer": optimizer.state_dict()
     * "next_step": next_step (integer)
     * "train_generator": train_generator.get_state()
     * "val_generator": val_generator.get_state()
   - Serialize dictionary to `out` via `torch.save(checkpoint, out)`.

2. load_checkpoint(
       src: str | Path | BinaryIO,
       model: torch.nn.Module,
       optimizer: torch.optim.Optimizer,
       train_generator: torch.Generator,
       val_generator: torch.Generator,
   ) -> int:
   - Load checkpoint dictionary via `torch.load(src, weights_only=False)`.
   - Restore states:
     * model.load_state_dict(checkpoint["model"])
     * optimizer.load_state_dict(checkpoint["optimizer"])
     * train_generator.set_state(checkpoint["train_generator"])
     * val_generator.set_state(checkpoint["val_generator"])
   - Return saved next_step (integer).

===============================================================================
PSEUDOCODE:
===============================================================================

def save_checkpoint(model, optimizer, next_step, train_generator, val_generator, out):
    # payload = {
    #     "model": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "next_step": next_step,
    #     "train_generator": train_generator.get_state(),
    #     "val_generator": val_generator.get_state(),
    # }
    # torch.save(payload, out)

def load_checkpoint(src, model, optimizer, train_generator, val_generator):
    # payload = torch.load(src, weights_only=False)
    # model.load_state_dict(payload["model"])
    # optimizer.load_state_dict(payload["optimizer"])
    # train_generator.set_state(payload["train_generator"])
    # val_generator.set_state(payload["val_generator"])
    # return int(payload["next_step"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, IO

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    next_step: int,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    # raise NotImplementedError("TODO: implement save_checkpoint")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_step": next_step,
        "train_generator": train_generator.get_state(),
        "val_generator": val_generator.get_state(),
    }
    torch.save(payload, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: Optimizer,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
) -> int:
    # raise NotImplementedError("TODO: implement load_checkpoint")
    payload = torch.load(src, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    train_generator.set_state(payload["train_generator"])
    val_generator.set_state(payload["val_generator"])
    return int(payload["next_step"])
