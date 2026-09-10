"""Checkpoint compatibility reports without exposing local paths."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import torch


EXPECTED_NEW_PREFIXES = (
    "head.okeydrive.",
    "head.motion_plan_head.okeydrive_refiner.",
)


def file_sha256(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file by content without embedding its name or location."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes without path metadata."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def categorize_checkpoint_keys(
    missing_keys: Iterable[str], unexpected_keys: Iterable[str]
) -> dict[str, list[str]]:
    """Separate expected new-module gaps from structural incompatibilities."""

    expected_missing = []
    incompatible_missing = []
    for key in missing_keys:
        if key.startswith(EXPECTED_NEW_PREFIXES):
            expected_missing.append(key)
        else:
            incompatible_missing.append(key)
    return {
        "expected_new_module_missing": sorted(expected_missing),
        "incompatible_missing": sorted(incompatible_missing),
        "unexpected": sorted(unexpected_keys),
    }
