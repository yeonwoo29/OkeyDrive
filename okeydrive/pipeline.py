"""End-to-end OkeyDrive proposal, visibility, keypoint, and BEV enhancement."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from .geometry import (
    BEVToSparseAdapter,
    GeometryAwareBEV,
    KeypointForegroundFusion,
    crop_tensor_rois,
    decoded_bev_boxes,
    project_sparse_boxes_to_views,
    scatter_roi_residuals,
    select_sparse_proposals,
)
from .keypoints import ClassConditionedKeypointAutoencoder, KeypointLocalizer, recover_keypoints
from .visibility import FineGrainedCLIPVisibility


class OkeyDriveEnhancer(nn.Module):
    """Enhance image features and sparse proposals before map/planning heads.

    The module consumes initial image-only SparseDrive proposals. It never uses
    ground-truth boxes, visibility, keypoints, trajectories, or collision labels
    during inference.
    """

    def __init__(
        self,
        channels: int = 256,
        max_objects: int = 20,
        proposal_score_threshold: float = 0.2,
        roi_size: int = 7,
        visibility_threshold: float = 0.5,
        keypoint_source: str = "recovered",
        use_keypoint_fusion: bool = True,
        use_bev_fusion: bool = True,
        use_perception_refinement: bool = True,
        x_bound: Sequence[float] = (-15.0, 15.0, 0.5),
        y_bound: Sequence[float] = (-10.0, 50.0, 0.5),
        clip_finetune_image: bool = True,
        clip_finetune_text: bool = True,
        clip_local_files_only: bool = False,
        clip_batch_size: int = 16,
        pretrained_keypoint_checkpoint: Optional[str] = None,
        visibility_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        if keypoint_source not in {"initial", "reconstructed", "recovered"}:
            raise ValueError("keypoint_source must be initial, reconstructed, or recovered")
        self.channels = channels
        self.max_objects = max_objects
        self.proposal_score_threshold = proposal_score_threshold
        self.roi_size = roi_size
        self.visibility_threshold = visibility_threshold
        self.keypoint_source = keypoint_source
        self.use_keypoint_fusion = use_keypoint_fusion
        self.use_bev_fusion = use_bev_fusion
        self.use_perception_refinement = use_perception_refinement
        self.x_bound = tuple(x_bound)
        self.y_bound = tuple(y_bound)
        if use_keypoint_fusion:
            self.visibility = visibility_module or FineGrainedCLIPVisibility(
                finetune_image=clip_finetune_image,
                finetune_text=clip_finetune_text,
                local_files_only=clip_local_files_only,
                image_batch_size=clip_batch_size,
            )
            self.localizer = KeypointLocalizer(self.visibility.output_dim, hidden_dim=channels)
            self.autoencoder = ClassConditionedKeypointAutoencoder(hidden_dim=channels)
            self.fusion = KeypointForegroundFusion(channels)
            if pretrained_keypoint_checkpoint is not None:
                self.load_pretrained_keypoint_modules(pretrained_keypoint_checkpoint)
        else:
            self.visibility = None
            self.localizer = None
            self.autoencoder = None
            self.fusion = None
        if use_bev_fusion:
            self.bev_encoder = GeometryAwareBEV(channels, x_bound=x_bound, y_bound=y_bound)
        else:
            self.bev_encoder = None
        if use_perception_refinement and use_bev_fusion:
            self.sparse_adapter = BEVToSparseAdapter(
                channels, x_bound=x_bound[:2], y_bound=y_bound[:2]
            )
        else:
            self.sparse_adapter = None

    def load_pretrained_keypoint_modules(self, checkpoint_path: str) -> None:
        """Load the three externally pretrained modules with strict key checks.

        Optimizer state and metadata are intentionally ignored. The checkpoint
        path is never included in an exception or report generated here.
        """

        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict):
            raise RuntimeError("the external pretraining checkpoint has no model state")
        modules = {
            "visibility": self.visibility,
            "localizer": self.localizer,
            "autoencoder": self.autoencoder,
        }
        consumed = set()
        for prefix, module in modules.items():
            prefix_with_dot = prefix + "."
            selected = {
                name[len(prefix_with_dot) :]: value
                for name, value in state_dict.items()
                if name.startswith(prefix_with_dot)
            }
            result = module.load_state_dict(selected, strict=False)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(
                    f"the external pretraining state is incompatible with {prefix}"
                )
            consumed.update(name for name in state_dict if name.startswith(prefix_with_dot))
        if set(state_dict) != consumed:
            raise RuntimeError("the external pretraining state contains unsupported modules")

    @staticmethod
    def upstream_normalized_to_rgb(images: torch.Tensor) -> torch.Tensor:
        """Invert the upstream ImageNet normalization to aligned RGB in [0,1]."""

        mean = images.new_tensor((123.675, 116.28, 103.53)).view(1, 1, 3, 1, 1)
        std = images.new_tensor((58.395, 57.12, 57.375)).view(1, 1, 3, 1, 1)
        return ((images * std + mean) / 255.0).clamp(0.0, 1.0)

    def _empty_context(self, det_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | None]:
        encoded = det_output["prediction"][-1]
        batch = encoded.shape[0]
        return {
            "bev": None,
            "bev_valid": None,
            "object_boxes": encoded.new_zeros((batch, 0, 5)),
            "object_visibility": encoded.new_zeros((batch, 0, 4)),
            "object_class": torch.zeros((batch, 0), device=encoded.device, dtype=torch.long),
            "object_valid": torch.zeros((batch, 0), device=encoded.device, dtype=torch.bool),
            "proposal_indices": torch.zeros((batch, 0), device=encoded.device, dtype=torch.long),
        }

    def forward(
        self,
        dense_features: Sequence[torch.Tensor],
        normalized_images: torch.Tensor,
        predicted_depths: Optional[Sequence[torch.Tensor]],
        det_output: Dict[str, torch.Tensor],
        projection_mat: torch.Tensor,
        image_wh: torch.Tensor,
        horizontal_flip: torch.Tensor | bool = False,
    ) -> Tuple[list[torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor | None]]:
        if not (self.use_keypoint_fusion or self.use_bev_fusion):
            return list(dense_features), det_output, self._empty_context(det_output)
        classification = det_output["classification"][-1]
        encoded_boxes = det_output["prediction"][-1]
        proposals, class_ids, proposal_valid, proposal_indices = select_sparse_proposals(
            classification,
            encoded_boxes,
            self.max_objects,
            self.proposal_score_threshold,
        )
        pixel_boxes, crop_valid = project_sparse_boxes_to_views(
            proposals, projection_mat, image_wh, proposal_valid
        )
        enhanced_features = list(dense_features)
        object_visibility = proposals.new_zeros((*crop_valid.shape, 4))
        if self.use_keypoint_fusion:
            rgb = self.upstream_normalized_to_rgb(normalized_images)
            rgb_crops, crop_valid = crop_tensor_rois(
                rgb, pixel_boxes, image_wh, 224, crop_valid
            )
            feature_crops, feature_valid = crop_tensor_rois(
                enhanced_features[0], pixel_boxes, image_wh, self.roi_size, crop_valid
            )
            crop_valid = crop_valid & feature_valid
            expanded_class = class_ids[:, None].expand(-1, normalized_images.shape[1], -1)
            visibility_output = self.visibility(
                rgb_crops,
                expanded_class,
                crop_valid,
                horizontal_flip=horizontal_flip,
            )
            object_visibility = visibility_output.probabilities
            initial = self.localizer(visibility_output.image_embeddings, expanded_class)
            reconstructed = self.autoencoder(initial, expanded_class)
            recovered, reliable = recover_keypoints(
                initial,
                reconstructed,
                object_visibility,
                threshold=self.visibility_threshold,
            )
            if self.keypoint_source == "initial":
                selected_keypoints = initial
                reliable = torch.ones_like(reliable)
            elif self.keypoint_source == "reconstructed":
                selected_keypoints = reconstructed
                reliable = torch.zeros_like(reliable)
            else:
                selected_keypoints = recovered
            fused_crops = self.fusion(
                feature_crops,
                selected_keypoints,
                object_visibility,
                reliable,
                expanded_class,
                crop_valid,
            )
            enhanced_features[0] = scatter_roi_residuals(
                enhanced_features[0],
                pixel_boxes,
                image_wh,
                fused_crops - feature_crops,
                crop_valid,
            )
        view_count = crop_valid.sum(dim=1)
        pooled_visibility = (
            object_visibility * crop_valid[..., None].to(object_visibility.dtype)
        ).sum(dim=1) / view_count.clamp_min(1).unsqueeze(-1)
        object_valid = proposal_valid & (view_count > 0)
        if self.use_bev_fusion:
            if not predicted_depths:
                raise RuntimeError(
                    "predicted depth is required for calibrated BEV lifting; GT depth is not an inference substitute"
                )
            bev, bev_valid = self.bev_encoder(
                enhanced_features[0], predicted_depths[0], projection_mat, image_wh
            )
            if self.sparse_adapter is not None:
                det_output["instance_feature"] = self.sparse_adapter(
                    det_output["instance_feature"], encoded_boxes, bev
                )
        else:
            bev, bev_valid = None, None
        context = {
            "bev": bev,
            "bev_valid": bev_valid,
            "object_boxes": decoded_bev_boxes(proposals),
            "object_visibility": pooled_visibility,
            "object_class": class_ids,
            "object_valid": object_valid,
            "proposal_indices": proposal_indices,
            "crop_valid": crop_valid,
        }
        det_output["okeydrive_context"] = context
        return enhanced_features, det_output, context
