"""External keypoint annotation mappings used by OkeyDrive pretraining."""

from __future__ import annotations

from typing import Tuple

import torch


CANONICAL_PART_NAMES = {
    0: ("front-left", "front-right", "rear-left", "rear-right"),
    1: ("left arm", "right arm", "left leg", "right leg"),
}

# This is the 12-point order in the locally inspected COCO-format CarFusion
# conversion. The raw CarFusion identifiers 9 and 14 are intentionally absent;
# the conversion retains only wheel, light, and roof landmarks.
CARFUSION_KEYPOINT_NAMES = (
    "rear_left_wheel",
    "rear_right_wheel",
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_light",
    "rear_right_light",
    "front_left_light",
    "front_right_light",
    "rear_left_roof",
    "rear_right_roof",
    "front_left_roof",
    "front_right_roof",
)

# Canonical order groups three landmarks per semantic part:
# front-left, front-right, rear-left, rear-right.
CARFUSION_CANONICAL_SOURCE_INDICES = (2, 6, 10, 3, 7, 11, 0, 4, 8, 1, 5, 9)

# Zero-based COCO person keypoint indices. Face landmarks are excluded.
COCO_PERSON_SOURCE_INDICES = (5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16)

KEYPOINT_TO_PART = torch.tensor(
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long
)


def _canonicalize(
    keypoints: torch.Tensor, source_indices: Tuple[int, ...]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return canonical coordinates and COCO-style visibility states.

    Args:
        keypoints: Tensor shaped ``[..., K, 3]`` with ``(x, y, v)`` entries.
        source_indices: Source indices for the 12 canonical output points.

    Returns:
        Coordinates ``[..., 12, 2]`` and integer states ``[..., 12]``.
        State 0 is unlabeled, 1 is labeled but occluded, and 2 is visible.
    """

    if keypoints.shape[-1] != 3:
        raise ValueError("keypoints must end in (x, y, visibility)")
    index = torch.as_tensor(source_indices, device=keypoints.device)
    if keypoints.shape[-2] <= int(index.max()):
        raise ValueError("annotation does not contain the required keypoint indices")
    selected = keypoints.index_select(-2, index)
    return selected[..., :2], selected[..., 2].to(torch.long)


def canonicalize_carfusion(keypoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map the inspected 12-point CarFusion conversion to canonical regions."""

    return _canonicalize(keypoints, CARFUSION_CANONICAL_SOURCE_INDICES)


def canonicalize_coco_person(keypoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map COCO shoulder-to-wrist and hip-to-ankle points to four limbs."""

    return _canonicalize(keypoints, COCO_PERSON_SOURCE_INDICES)


def build_part_visibility_targets(
    states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build independent part visibility targets without conflating state 0 and 1.

    A part target is the fraction of its labeled landmarks whose COCO state is
    2. A part is supervised when at least one of its three landmarks is labeled.
    The four outputs are independent probabilities and are never normalized by
    a softmax.
    """

    if states.shape[-1] != 12:
        raise ValueError("states must contain 12 canonical keypoints")
    part_index = KEYPOINT_TO_PART.to(states.device)
    targets = []
    masks = []
    for part in range(4):
        part_states = states[..., part_index == part]
        labeled = part_states > 0
        count = labeled.sum(dim=-1)
        visible = (part_states == 2).sum(dim=-1)
        targets.append(visible.to(torch.float32) / count.clamp_min(1))
        masks.append(count > 0)
    return torch.stack(targets, dim=-1), torch.stack(masks, dim=-1)
