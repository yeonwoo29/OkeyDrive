"""CLIP ViT-B/16 fine-grained visibility grounding."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


PART_PROMPTS: Dict[int, Tuple[str, ...]] = {
    0: (
        "the visible front-left side of the vehicle",
        "the visible front-right side of the vehicle",
        "the visible rear-left side of the vehicle",
        "the visible rear-right side of the vehicle",
    ),
    1: (
        "the visible left arm of the pedestrian",
        "the visible right arm of the pedestrian",
        "the visible left leg of the pedestrian",
        "the visible right leg of the pedestrian",
    ),
}

FLIP_PART_PERMUTATION = torch.tensor([1, 0, 3, 2], dtype=torch.long)


@dataclass
class VisibilityOutput:
    probabilities: torch.Tensor
    image_embeddings: torch.Tensor
    valid_mask: torch.Tensor


class IndependentSigmoidCalibration(nn.Module):
    """Convert cosine scores to four independent calibrated probabilities."""

    def __init__(self, num_classes: int = 2, num_parts: int = 4):
        super().__init__()
        self.log_scale = nn.Parameter(torch.full((num_classes, num_parts), 2.0))
        self.bias = nn.Parameter(torch.zeros(num_classes, num_parts))

    def forward(self, cosine: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale[class_ids.clamp(0, 1)]) + 1e-4
        bias = self.bias[class_ids.clamp(0, 1)]
        return torch.sigmoid(scale * cosine + bias)


def remap_semantic_parts_for_flip(
    values: torch.Tensor, horizontal_flip: torch.Tensor | bool
) -> torch.Tensor:
    """Swap semantic left/right part slots for horizontally flipped crops."""

    if values.shape[-1] != 4:
        raise ValueError("part values must have four channels")
    permutation = FLIP_PART_PERMUTATION.to(values.device)
    swapped = values.index_select(-1, permutation)
    flip = torch.as_tensor(horizontal_flip, device=values.device, dtype=torch.bool)
    while flip.ndim < values.ndim:
        flip = flip.unsqueeze(-1)
    return torch.where(flip, swapped, values)


class FineGrainedCLIPVisibility(nn.Module):
    """Use both pretrained CLIP ViT-B/16 encoders on aligned RGB crops.

    Production construction loads ``openai/clip-vit-base-patch16`` through
    Transformers. Tests may pass a compatible pretrained encoder and tokenizer
    explicitly, but the production path never substitutes a random image MLP.
    """

    model_id = "openai/clip-vit-base-patch16"
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        finetune_image: bool = True,
        finetune_text: bool = True,
        local_files_only: bool = False,
        image_batch_size: int = 16,
        model: Optional[nn.Module] = None,
        tokenizer=None,
    ):
        super().__init__()
        if model is None:
            try:
                from transformers import AutoTokenizer, CLIPModel
            except ImportError as exc:
                raise RuntimeError(
                    "Transformers with CLIPModel is required for the production visibility path"
                ) from exc
            cache_dir = os.getenv("CLIP_CACHE_ROOT") or None
            model = CLIPModel.from_pretrained(
                self.model_id,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
        if tokenizer is None:
            raise ValueError("a tokenizer is required with an injected CLIP model")
        self.model = model
        self.tokenizer = tokenizer
        self.finetune_image = finetune_image
        self.finetune_text = finetune_text
        if image_batch_size < 1:
            raise ValueError("image_batch_size must be positive")
        self.image_batch_size = image_batch_size
        for parameter in self.model.vision_model.parameters():
            parameter.requires_grad = finetune_image
        for parameter in self.model.visual_projection.parameters():
            parameter.requires_grad = finetune_image
        for parameter in self.model.text_model.parameters():
            parameter.requires_grad = finetune_text
        for parameter in self.model.text_projection.parameters():
            parameter.requires_grad = finetune_text
        self.calibration = IndependentSigmoidCalibration()
        projection_dim = int(self.model.config.projection_dim)
        self.output_dim = projection_dim
        self.register_buffer("_cached_text", torch.empty(0), persistent=False)

    def _preprocess(self, crops: torch.Tensor) -> torch.Tensor:
        if crops.shape[-3] != 3:
            raise ValueError("CLIP crops must be RGB tensors with three channels")
        crops = F.interpolate(crops, size=(224, 224), mode="bicubic", align_corners=False)
        mean = crops.new_tensor(self.clip_mean).view(1, 3, 1, 1)
        std = crops.new_tensor(self.clip_std).view(1, 3, 1, 1)
        return (crops.clamp(0.0, 1.0) - mean) / std

    def _encode_text(self, device: torch.device) -> torch.Tensor:
        if not self.finetune_text and self._cached_text.numel() > 0:
            return self._cached_text.to(device)
        prompts = [prompt for class_prompts in PART_PROMPTS.values() for prompt in class_prompts]
        tokens = self.tokenizer(prompts, padding=True, return_tensors="pt")
        tokens = {name: value.to(device) for name, value in tokens.items()}
        text = self.model.get_text_features(**tokens)
        text = F.normalize(text, dim=-1).reshape(2, 4, -1)
        if not self.finetune_text:
            self._cached_text = text.detach()
        return text

    def forward(
        self,
        crops: torch.Tensor,
        class_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        horizontal_flip: torch.Tensor | bool = False,
    ) -> VisibilityOutput:
        leading_shape = class_ids.shape
        if crops.shape[:-3] != leading_shape:
            raise ValueError("crop and class index leading shapes must match")
        flat_crops = crops.reshape(-1, 3, crops.shape[-2], crops.shape[-1])
        flat_classes = class_ids.reshape(-1).clamp(0, 1)
        if valid_mask is None:
            valid_mask = torch.ones(leading_shape, device=crops.device, dtype=torch.bool)
        flat_valid = valid_mask.reshape(-1)
        embeddings = crops.new_zeros((flat_crops.shape[0], self.output_dim))
        probabilities = crops.new_zeros((flat_crops.shape[0], 4))
        if flat_valid.any():
            text_features = self._encode_text(crops.device)
            valid_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
            for start in range(0, valid_indices.numel(), self.image_batch_size):
                indices = valid_indices[start : start + self.image_batch_size]
                pixel_values = self._preprocess(flat_crops.index_select(0, indices))
                image_features = F.normalize(
                    self.model.get_image_features(pixel_values=pixel_values), dim=-1
                )
                selected_classes = flat_classes.index_select(0, indices)
                selected_text = text_features[selected_classes]
                cosine = torch.einsum("nd,npd->np", image_features, selected_text)
                calibrated = self.calibration(cosine, selected_classes)
                embeddings[indices] = image_features
                probabilities[indices] = calibrated
        probabilities = probabilities.reshape(*leading_shape, 4)
        probabilities = remap_semantic_parts_for_flip(probabilities, horizontal_flip)
        return VisibilityOutput(
            probabilities=probabilities,
            image_embeddings=embeddings.reshape(*leading_shape, self.output_dim),
            valid_mask=valid_mask,
        )
