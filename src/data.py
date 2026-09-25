"""Data Loading and Batching for Pretokenized Streams.

===============================================================================
WHAT NEEDS TO BE IMPLEMENTED:
===============================================================================
1. load_token_array(path: str | Path) -> np.memmap:
   - Accept filesystem path to a raw little-endian uint16 binary file.
   - Validation:
     * Reject missing file (raise FileNotFoundError).
     * Reject file with odd byte length (raise ValueError: odd byte length cannot form uint16).
   - Return one-dimensional, read-only memory map:
       np.memmap(path, mode="r", dtype=np.dtype("<u2"))
   - Do NOT load or convert the complete stream into an in-memory array.

2. get_batch(
       dataset: np.ndarray,
       batch_size: int,
       sequence_length: int,
       device: str | torch.device,
       generator: torch.Generator,
   ) -> tuple[torch.Tensor, torch.Tensor]:
   - Validation:
     * Require batch_size > 0 and sequence_length > 0.
     * Require len(dataset) >= sequence_length + 1.
   - Sampling:
     * Sample batch_size starting indices uniformly from [0, len(dataset) - sequence_length)
       using the supplied torch.Generator as the sole source of randomness:
         starts = torch.randint(0, len(dataset) - sequence_length, (batch_size,), generator=generator)
   - Slices:
     * For each start offset o in starts:
       x_slice = dataset[o : o + sequence_length]
       y_slice = dataset[o + 1 : o + sequence_length + 1]
     * Stack sampled slices, convert to torch.long, and move to device.
     * Return (x, y) both of shape (batch_size, sequence_length) as torch.long on device.

===============================================================================
PSEUDOCODE:
===============================================================================

def load_token_array(path):
    # p = Path(path)
    # if not p.exists() or not p.is_file():
    #     raise FileNotFoundError(f"File not found: {path}")
    # if p.stat().st_size % 2 != 0:
    #     raise ValueError(f"File size {p.stat().st_size} is not a multiple of 2 (uint16)")
    # return np.memmap(p, mode="r", dtype=np.dtype("<u2"))

def get_batch(dataset, batch_size, sequence_length, device, generator):
    # if batch_size <= 0 or sequence_length <= 0:
    #     raise ValueError("Batch size and sequence length must be positive")
    # n_tokens = len(dataset)
    # if n_tokens < sequence_length + 1:
    #     raise ValueError(f"Dataset length {n_tokens} too short for sequence_length {sequence_length}")
    # max_start = n_tokens - sequence_length
    # starts = torch.randint(0, max_start, (batch_size,), generator=generator).tolist()
    # x_batch = [dataset[s : s + sequence_length] for s in starts]
    # y_batch = [dataset[s + 1 : s + sequence_length + 1] for s in starts]
    # x_tensor = torch.tensor(np.array(x_batch), dtype=torch.long, device=device)
    # y_tensor = torch.tensor(np.array(y_batch), dtype=torch.long, device=device)
    # return x_tensor, y_tensor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_token_array(path: str | Path) -> np.memmap:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if p.stat().st_size % 2 != 0:
        raise ValueError(f"File size {p.stat().st_size} is not a multiple of 2 (uint16)")
    return np.memmap(p, mode="r", dtype=np.dtype("<u2"))


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    sequence_length: int,
    device: str | torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("Batch size and sequence length must be positive")
    n_tokens = len(dataset)
    if n_tokens < sequence_length + 1:
        raise ValueError(f"Dataset length {n_tokens} too short for sequence_length {sequence_length}")

    max_start = n_tokens - sequence_length
    starts = torch.randint(0, max_start, (batch_size,), generator=generator).tolist()
    x_batch = np.stack([dataset[s : s + sequence_length] for s in starts]).astype(np.int64)
    y_batch = np.stack([dataset[s + 1 : s + sequence_length + 1] for s in starts]).astype(np.int64)

    device_obj = torch.device(device) if isinstance(device, str) else device
    x_tensor = torch.from_numpy(x_batch).to(device_obj)
    y_tensor = torch.from_numpy(y_batch).to(device_obj)
    return x_tensor, y_tensor