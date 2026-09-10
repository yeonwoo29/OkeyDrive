"""Trainable external supervision model for CLIP visibility and keypoint recovery."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .keypoints import (
    ClassConditionedKeypointAutoencoder,
    KeypointLocalizer,
    masked_smooth_l1_loss,
)
from .visibility import FineGrainedCLIPVisibility


class OkeyDrivePretrainingModel(nn.Module):
    """Jointly fine-tune CLIP calibration/localization and denoising geometry."""

    def __init__(
        self,
        finetune_image: bool = True,
        finetune_text: bool = True,
        local_files_only: bool = False,
        visibility_module: nn.Module | None = None,
    ):
        super().__init__()
        self.visibility = visibility_module or FineGrainedCLIPVisibility(
            finetune_image=finetune_image,
            finetune_text=finetune_text,
            local_files_only=local_files_only,
        )
        self.localizer = KeypointLocalizer(self.visibility.output_dim)
        self.autoencoder = ClassConditionedKeypointAutoencoder()

    def forward(self, batch: dict, corruption_probability: float = 0.35) -> dict:
        crops = batch["crop"]
        class_ids = batch["class_id"]
        clip_output = self.visibility(
            crops,
            class_ids,
            horizontal_flip=batch.get("horizontal_flip", False),
        )
        initial = self.localizer(clip_output.image_embeddings, class_ids)
        labeled = batch["keypoint_mask"].to(torch.bool)
        random_keep = torch.rand_like(labeled.to(torch.float32)) > corruption_probability
        corrupted_mask = labeled & random_keep
        corrupted = torch.where(corrupted_mask[..., None], batch["keypoints"], torch.zeros_like(initial))
        reconstructed = self.autoencoder(corrupted, class_ids, corrupted_mask)
        visibility_loss_raw = F.binary_cross_entropy(
            clip_output.probabilities, batch["part_visibility"], reduction="none"
        )
        part_mask = batch["part_mask"].to(visibility_loss_raw.dtype)
        visibility_loss = (visibility_loss_raw * part_mask).sum() / part_mask.sum().clamp_min(1.0)
        localization_loss = masked_smooth_l1_loss(initial, batch["keypoints"], labeled)
        reconstruction_loss = masked_smooth_l1_loss(reconstructed, batch["keypoints"], labeled)
        return {
            "loss_visibility": visibility_loss,
            "loss_keypoint_localization": localization_loss,
            "loss_keypoint_reconstruction": reconstruction_loss,
            "initial": initial,
            "reconstructed": reconstructed,
        }
