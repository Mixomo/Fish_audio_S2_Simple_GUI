"""Native s2.cpp LoRA adapter export helpers.

The .s2lora format is intentionally tiny and dependency-light for C++ loading:
  magic[8] = b"S2LORA1\\0"
  uint32 version, uint32 rank, float32 alpha, uint32 tensor_count
  repeated: uint32 name_len, name bytes, uint32 rows, uint32 cols, float32 data

Only Fish S2 Fast-AR LoRA targets are accepted. This matches the recommended
training preset and prevents silently applying a partial adapter in C++.
"""
from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

MAGIC = b"S2LORA1\0"
VERSION = 1
FAST_ROOTS = ("fast_layers.", "fast_embeddings.", "fast_output.")


def _state_dict_from_checkpoint(checkpoint_path: str | os.PathLike) -> Dict[str, torch.Tensor]:
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint does not contain a state_dict.")
    return obj


def _normalize_lora_name(name: str) -> str | None:
    if ".lora_A" not in name and ".lora_B" not in name:
        return None
    # Lightning checkpoints commonly prepend model.model. or model.; anchor at
    # an actual Fish Fast-AR module instead of relying on a particular prefix.
    for root in FAST_ROOTS:
        idx = name.find(root)
        if idx >= 0:
            return name[idx:]
    return None


def inspect_fast_lora(checkpoint_path: str | os.PathLike) -> Tuple[Dict[str, torch.Tensor], list[str]]:
    state = _state_dict_from_checkpoint(checkpoint_path)
    tensors: Dict[str, torch.Tensor] = {}
    unsupported: list[str] = []
    for raw_name, value in state.items():
        if "lora_" not in raw_name or not torch.is_tensor(value):
            continue
        normalized = _normalize_lora_name(raw_name)
        if normalized is None:
            unsupported.append(raw_name)
            continue
        tensors[normalized] = value.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return tensors, unsupported


def export_s2lora(
    checkpoint_path: str | os.PathLike,
    output_path: str | os.PathLike,
    rank: int,
    alpha: float,
    metadata_path: str | os.PathLike | None = None,
) -> str:
    rank = int(rank)
    alpha = float(alpha)
    if rank <= 0:
        raise ValueError("LoRA rank must be > 0.")

    tensors, unsupported = inspect_fast_lora(checkpoint_path)
    if unsupported:
        preview = "\n".join(f"  - {x}" for x in unsupported[:8])
        raise ValueError(
            "This checkpoint contains non-Fast-AR LoRA tensors. Native C++ "
            "adapter inference intentionally refuses partial adapters. Export a "
            "Fast-AR-only checkpoint or use the merged PyTorch model instead.\n" + preview
        )
    if not tensors:
        raise ValueError("No Fish S2 Fast-AR LoRA tensors were found in the checkpoint.")

    # Validate A/B pairs and rank. loralib Embedding stores A as [r, vocab].
    # Transpose only that A tensor so C++ can use ggml_get_rows directly.
    bases = sorted({name.rsplit(".lora_", 1)[0] for name in tensors})
    normalized: Dict[str, np.ndarray] = {}
    for base in bases:
        a_name = base + ".lora_A"
        b_name = base + ".lora_B"
        if a_name not in tensors or b_name not in tensors:
            raise ValueError(f"Incomplete LoRA pair: {base}")
        a = tensors[a_name]
        b = tensors[b_name]
        if base == "fast_embeddings":
            # [r, vocab] -> [vocab, r]
            a = a.transpose(0, 1).contiguous()
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Only 2D LoRA tensors are supported: {base}")
        normalized[a_name] = a.numpy().astype(np.float32, copy=False)
        normalized[b_name] = b.numpy().astype(np.float32, copy=False)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIfI", VERSION, rank, alpha, len(normalized)))
        for name in sorted(normalized):
            arr = np.ascontiguousarray(normalized[name], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Adapter tensor is not 2D: {name}")
            name_b = name.encode("utf-8")
            rows, cols = map(int, arr.shape)
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<II", rows, cols))
            f.write(arr.tobytes(order="C"))

    if metadata_path is not None:
        meta = {
            "format": "s2lora",
            "version": VERSION,
            "source_checkpoint": os.path.abspath(os.fspath(checkpoint_path)),
            "rank": rank,
            "alpha": alpha,
            "target_scope": "fast_ar_only",
            "tensor_pairs": bases,
            "adapter": str(output.name),
        }
        mp = Path(metadata_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return str(output)
