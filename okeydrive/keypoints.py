"""Class-conditioned keypoint localization, reconstruction, and hard gating."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .annotations import KEYPOINT_TO_PART


class KeypointLocalizer(nn.Module):
    """Predict 12 crop-normalized keypoints from CLIP object embeddings."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 24),
        )

    def forward(self, object_features: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        if object_features.shape[:-1] != class_ids.shape:
            raise ValueError("object feature and class index leading shapes must match")
        class_features = self.class_embedding(class_ids.clamp(0, 1))
        keypoints = self.net(torch.cat([object_features, class_features], dim=-1))
        return keypoints.sigmoid().reshape(*class_ids.shape, 12, 2)


class ClassConditionedKeypointAutoencoder(nn.Module):
    """Reconstruct clean normalized geometry from corrupted 12-point inputs."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, 16)
        self.encoder = nn.Sequential(
            nn.Linear(24 + 12 + 16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 24),
        )

    def forward(
        self,
        keypoints: torch.Tensor,
        class_ids: torch.Tensor,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if keypoints.shape[-2:] != (12, 2):
            raise ValueError("keypoints must have shape [..., 12, 2]")
        if labeled_mask is None:
            labeled_mask = torch.ones_like(keypoints[..., :1], dtype=torch.bool)
        if labeled_mask.shape == keypoints.shape[:-1]:
            labeled_mask = labeled_mask.unsqueeze(-1)
        if labeled_mask.shape != keypoints.shape[:-1] + (1,):
            raise ValueError("labeled_mask must have shape [..., 12, 1]")
        class_features = self.class_embedding(class_ids.clamp(0, 1))
        masked = torch.where(labeled_mask, keypoints, torch.zeros_like(keypoints))
        encoded = torch.cat(
            [masked.flatten(-2), labeled_mask.to(keypoints.dtype).squeeze(-1), class_features],
            dim=-1,
        )
        latent = self.encoder(encoded)
        reconstructed = self.decoder(torch.cat([latent, class_features], dim=-1))
        return reconstructed.sigmoid().reshape_as(keypoints)


def visibility_to_keypoint_mask(
    visibility: torch.Tensor,
    threshold: float = 0.5,
    keypoint_to_part: torch.Tensor = KEYPOINT_TO_PART,
) -> torch.Tensor:
    """Expand four independent part probabilities to a hard 12-point mask."""

    if visibility.shape[-1] != 4:
        raise ValueError("visibility must have four part channels")
    part_index = keypoint_to_part.to(visibility.device)
    return visibility.index_select(-1, part_index).unsqueeze(-1) >= threshold


def recover_keypoints(
    initial: torch.Tensor,
    reconstructed: torch.Tensor,
    visibility: torch.Tensor,
    threshold: float = 0.5,
    labeled_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep reliable keypoints exactly and replace only unreliable keypoints.

    The returned boolean mask is true for visible/reliable points. Recovered
    coordinates do not alter the observed visibility evidence.
    """

    if initial.shape != reconstructed.shape or initial.shape[-2:] != (12, 2):
        raise ValueError("initial and reconstructed keypoints must match [..., 12, 2]")
    reliable = visibility_to_keypoint_mask(visibility, threshold)
    if labeled_mask is not None:
        if labeled_mask.shape == initial.shape[:-1]:
            labeled_mask = labeled_mask.unsqueeze(-1)
        reliable = reliable & labeled_mask.to(torch.bool)
    recovered = torch.where(reliable, initial, reconstructed)
    return recovered, reliable


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labeled_mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 over labeled coordinates only; unlabeled zeros have no effect."""

    if labeled_mask.shape == prediction.shape[:-1]:
        labeled_mask = labeled_mask.unsqueeze(-1)
    coordinate_mask = labeled_mask.expand_as(prediction).to(prediction.dtype)
    raw = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return (raw * coordinate_mask).sum() / coordinate_mask.sum().clamp_min(1.0)


def normalize_keypoints_to_box(
    points_xy: torch.Tensor, boxes_xyxy: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize image coordinates to an object crop and return valid boxes."""

    size = boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]
    valid = (size[..., 0] > 1.0) & (size[..., 1] > 1.0)
    normalized = (points_xy - boxes_xyxy[..., None, :2]) / size.clamp_min(1.0)[..., None, :]
    return normalized, valid


def denormalize_keypoints_from_box(points: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Map crop-normalized coordinates back to augmented image pixels."""

    size = boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]
    return boxes_xyxy[..., None, :2] + points * size[..., None, :]
