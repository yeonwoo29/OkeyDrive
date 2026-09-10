"""ROI alignment, calibrated lifting, BEV splatting, and sparse adaptation."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def select_sparse_proposals(
    classification: torch.Tensor,
    encoded_boxes: torch.Tensor,
    max_objects: int,
    score_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select current predicted proposals without using ground truth.

    NuScenes detection classes 0--4 are mapped to vehicle and class 8 to
    pedestrian. Other classes remain invalid for the keypoint path.
    """

    scores, labels = classification.sigmoid().max(dim=-1)
    count = min(max_objects, scores.shape[1])
    top_scores, indices = scores.topk(count, dim=1, sorted=True)
    gather_box = indices[..., None].expand(-1, -1, encoded_boxes.shape[-1])
    boxes = torch.gather(encoded_boxes, 1, gather_box)
    labels = torch.gather(labels, 1, indices)
    class_ids = torch.where(labels == 8, torch.ones_like(labels), torch.zeros_like(labels))
    supported = ((labels >= 0) & (labels <= 4)) | (labels == 8)
    valid = supported & (top_scores >= score_threshold)
    return boxes, class_ids, valid, indices


def _encoded_box_corners(encoded_boxes: torch.Tensor) -> torch.Tensor:
    """Decode sparse boxes to their eight lidar-frame corners."""

    centers = encoded_boxes[..., :3]
    sizes = encoded_boxes[..., 3:6].exp().clamp(max=100.0)
    sin_yaw = encoded_boxes[..., 6]
    cos_yaw = encoded_boxes[..., 7]
    norm = torch.sqrt(sin_yaw.square() + cos_yaw.square()).clamp_min(1e-6)
    sin_yaw, cos_yaw = sin_yaw / norm, cos_yaw / norm
    signs = encoded_boxes.new_tensor(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )
    # SparseDrive stores (width, length, height), while local +x follows yaw.
    oriented_sizes = sizes[..., [1, 0, 2]]
    local = signs * oriented_sizes[..., None, :] * 0.5
    x = cos_yaw[..., None] * local[..., 0] - sin_yaw[..., None] * local[..., 1]
    y = sin_yaw[..., None] * local[..., 0] + cos_yaw[..., None] * local[..., 1]
    rotated = torch.stack([x, y, local[..., 2]], dim=-1)
    return rotated + centers[..., None, :]


def project_sparse_boxes_to_views(
    encoded_boxes: torch.Tensor,
    projection_mat: torch.Tensor,
    image_wh: torch.Tensor,
    object_valid: torch.Tensor | None = None,
    min_size: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project lidar-frame 3D boxes to augmented camera pixels.

    Args:
        encoded_boxes: ``[B, O, >=8]`` SparseDrive box representation.
        projection_mat: ``[B, V, 4, 4]`` augmented lidar-to-image matrices.
        image_wh: ``[B, V, 2]`` augmented image width and height.

    Returns:
        Pixel boxes ``[B, V, O, 4]`` and validity ``[B, V, O]``.
    """

    corners = _encoded_box_corners(encoded_boxes)
    homogeneous = torch.cat([corners, torch.ones_like(corners[..., :1])], dim=-1)
    projected = torch.einsum("bvij,boqj->bvoqi", projection_mat, homogeneous)
    depth = projected[..., 2]
    front = depth > 0.1
    uv = projected[..., :2] / depth.clamp_min(0.1).unsqueeze(-1)
    width = image_wh[..., 0, None, None]
    height = image_wh[..., 1, None, None]
    u = uv[..., 0].clamp(min=0.0)
    v = uv[..., 1].clamp(min=0.0)
    u = torch.minimum(u, (width - 1.0).clamp_min(0.0))
    v = torch.minimum(v, (height - 1.0).clamp_min(0.0))
    large = torch.finfo(uv.dtype).max / 16
    u_min = torch.where(front, u, large).amin(dim=-1)
    v_min = torch.where(front, v, large).amin(dim=-1)
    u_max = torch.where(front, u, -large).amax(dim=-1)
    v_max = torch.where(front, v, -large).amax(dim=-1)
    boxes = torch.stack([u_min, v_min, u_max, v_max], dim=-1)
    valid = front.sum(dim=-1) >= 4
    valid = valid & ((u_max - u_min) >= min_size) & ((v_max - v_min) >= min_size)
    if object_valid is not None:
        valid = valid & object_valid[:, None]
    boxes = torch.where(valid[..., None], boxes, torch.zeros_like(boxes))
    return boxes, valid


def crop_tensor_rois(
    images: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    image_wh: torch.Tensor,
    output_size: int,
    valid_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiably crop aligned ``[B,V,O]`` regions using grid sampling."""

    if images.ndim != 5 or boxes_xyxy.ndim != 4:
        raise ValueError("images and boxes must be [B,V,C,H,W] and [B,V,O,4]")
    batch, views, channels = images.shape[:3]
    objects = boxes_xyxy.shape[2]
    if boxes_xyxy.shape[:2] != (batch, views):
        raise ValueError("box batch/view dimensions do not match images")
    size = boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]
    valid = (size[..., 0] > 1.0) & (size[..., 1] > 1.0)
    if valid_mask is not None:
        valid = valid & valid_mask
    lin = torch.linspace(0.0, 1.0, output_size, device=images.device, dtype=images.dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    x = boxes_xyxy[..., 0, None, None] + xx * size[..., 0, None, None]
    y = boxes_xyxy[..., 1, None, None] + yy * size[..., 1, None, None]
    norm_x = 2.0 * x / (image_wh[..., 0, None, None, None] - 1.0).clamp_min(1.0) - 1.0
    norm_y = 2.0 * y / (image_wh[..., 1, None, None, None] - 1.0).clamp_min(1.0) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).reshape(-1, output_size, output_size, 2)
    expanded = images[:, :, None].expand(-1, -1, objects, -1, -1, -1)
    expanded = expanded.reshape(-1, channels, images.shape[-2], images.shape[-1])
    crops = F.grid_sample(expanded, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    crops = crops.reshape(batch, views, objects, channels, output_size, output_size)
    crops = crops * valid[..., None, None, None].to(crops.dtype)
    return crops, valid


def scatter_roi_residuals(
    feature_map: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    image_wh: torch.Tensor,
    residual_crops: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter ROI residuals with overlap normalization and no outside overwrite."""

    batch, views, channels, height, width = feature_map.shape
    accumulation = torch.zeros_like(feature_map)
    counts = feature_map.new_zeros((batch, views, 1, height, width))
    scale_x = width / image_wh[..., 0].clamp_min(1.0)
    scale_y = height / image_wh[..., 1].clamp_min(1.0)
    for batch_index in range(batch):
        for view_index in range(views):
            for object_index in range(boxes_xyxy.shape[2]):
                if not bool(valid_mask[batch_index, view_index, object_index]):
                    continue
                box = boxes_xyxy[batch_index, view_index, object_index]
                x0 = max(0, int(torch.floor(box[0] * scale_x[batch_index, view_index]).item()))
                y0 = max(0, int(torch.floor(box[1] * scale_y[batch_index, view_index]).item()))
                x1 = min(width, int(torch.ceil(box[2] * scale_x[batch_index, view_index]).item()))
                y1 = min(height, int(torch.ceil(box[3] * scale_y[batch_index, view_index]).item()))
                if x1 <= x0 or y1 <= y0:
                    continue
                resized = F.interpolate(
                    residual_crops[batch_index, view_index, object_index][None],
                    size=(y1 - y0, x1 - x0),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                padded = F.pad(resized, (x0, width - x1, y0, height - y1))
                mask = F.pad(
                    resized.new_ones((1, y1 - y0, x1 - x0)),
                    (x0, width - x1, y0, height - y1),
                )
                view_selector = feature_map.new_zeros((batch, views, 1, 1, 1))
                view_selector[batch_index, view_index] = 1.0
                accumulation = accumulation + view_selector * padded[None, None]
                counts = counts + view_selector * mask[None, None]
    return feature_map + accumulation / counts.clamp_min(1.0)


class KeypointForegroundFusion(nn.Module):
    """Use foreground tokens as queries and geometric keypoints as keys/values."""

    def __init__(self, channels: int, token_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.query = nn.Linear(channels, token_dim)
        self.keypoint = nn.Sequential(
            nn.Linear(10, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.attention = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.output = nn.Linear(token_dim, channels)

    def forward(
        self,
        roi_features: torch.Tensor,
        keypoints: torch.Tensor,
        part_visibility: torch.Tensor,
        reliable: torch.Tensor,
        class_ids: torch.Tensor,
        roi_valid: torch.Tensor,
    ) -> torch.Tensor:
        leading = roi_features.shape[:3]
        channels, roi_h, roi_w = roi_features.shape[-3:]
        query = roi_features.flatten(-2).transpose(-1, -2).reshape(-1, roi_h * roi_w, channels)
        part_ids = torch.arange(4, device=keypoints.device).repeat_interleave(3)
        part_one_hot = F.one_hot(part_ids, 4).to(keypoints.dtype)
        part_one_hot = part_one_hot.view(*([1] * (keypoints.ndim - 2)), 12, 4).expand(*keypoints.shape[:-2], -1, -1)
        point_visibility = part_visibility.index_select(-1, part_ids).unsqueeze(-1)
        reconstructed = (~reliable).to(keypoints.dtype)
        class_one_hot = F.one_hot(class_ids.clamp(0, 1), 2).to(keypoints.dtype)
        class_one_hot = class_one_hot[..., None, :].expand(*keypoints.shape[:-2], 12, 2)
        geometry = torch.cat(
            [keypoints, point_visibility, reconstructed, part_one_hot, class_one_hot], dim=-1
        ).reshape(-1, 12, 10)
        q = self.query(query)
        kv = self.keypoint(geometry)
        fused, _ = self.attention(q, kv, kv, need_weights=False)
        residual = self.output(fused).transpose(1, 2).reshape(*leading, channels, roi_h, roi_w)
        return roi_features + residual * roi_valid[..., None, None, None].to(residual.dtype)


class GeometryAwareBEV(nn.Module):
    """Lift predicted image depth through augmented calibration and splat to BEV."""

    def __init__(
        self,
        channels: int,
        x_bound: Sequence[float] = (-15.0, 15.0, 0.5),
        y_bound: Sequence[float] = (-10.0, 50.0, 0.5),
        min_depth: float = 0.1,
        max_depth: float = 60.0,
    ):
        super().__init__()
        self.channels = channels
        self.x_bound = tuple(float(value) for value in x_bound)
        self.y_bound = tuple(float(value) for value in y_bound)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.height = int(round((self.y_bound[1] - self.y_bound[0]) / self.y_bound[2]))
        self.width = int(round((self.x_bound[1] - self.x_bound[0]) / self.x_bound[2]))
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        predicted_depth: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image_features.ndim != 5:
            raise ValueError("image_features must be [B,V,C,H,W]")
        batch, views, channels, height, width = image_features.shape
        if predicted_depth.ndim == 4 and predicted_depth.shape[0] == batch * views:
            predicted_depth = predicted_depth.reshape(batch, views, 1, height, width)
        if predicted_depth.shape != (batch, views, 1, height, width):
            predicted_depth = F.interpolate(
                predicted_depth.reshape(batch * views, 1, *predicted_depth.shape[-2:]),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, views, 1, height, width)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=image_features.device, dtype=image_features.dtype),
            torch.arange(width, device=image_features.device, dtype=image_features.dtype),
            indexing="ij",
        )
        pixel_x = (x_grid + 0.5) * image_wh[..., 0, None, None] / width
        pixel_y = (y_grid + 0.5) * image_wh[..., 1, None, None] / height
        depth = predicted_depth[:, :, 0].clamp(self.min_depth, self.max_depth)
        homogeneous = torch.stack(
            [pixel_x * depth, pixel_y * depth, depth, torch.ones_like(depth)], dim=-1
        )
        inverse_projection = torch.linalg.inv(projection_mat.to(image_features.dtype))
        lidar = torch.einsum("bvij,bvhwj->bvhwi", inverse_projection, homogeneous)
        lidar_xyz = lidar[..., :3] / lidar[..., 3:4].clamp_min(1e-6)
        cell_x = torch.floor((lidar_xyz[..., 0] - self.x_bound[0]) / self.x_bound[2]).long()
        cell_y = torch.floor((lidar_xyz[..., 1] - self.y_bound[0]) / self.y_bound[2]).long()
        valid = (
            (cell_x >= 0)
            & (cell_x < self.width)
            & (cell_y >= 0)
            & (cell_y < self.height)
            & torch.isfinite(depth)
        )
        linear = (cell_y * self.width + cell_x).clamp(0, self.height * self.width - 1)
        bev = image_features.new_zeros((batch, channels, self.height * self.width))
        counts = image_features.new_zeros((batch, 1, self.height * self.width))
        source = image_features.permute(0, 2, 1, 3, 4).reshape(batch, channels, -1)
        indices = linear.reshape(batch, 1, -1).expand(-1, channels, -1)
        weight = valid.reshape(batch, 1, -1).to(image_features.dtype)
        bev.scatter_add_(2, indices, source * weight)
        counts.scatter_add_(2, linear.reshape(batch, 1, -1), weight)
        bev = (bev / counts.clamp_min(1.0)).reshape(batch, channels, self.height, self.width)
        return bev + self.enhance(bev), counts.reshape(batch, 1, self.height, self.width) > 0

    def normalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        x = 2.0 * (xy[..., 0] - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0]) - 1.0
        y = 2.0 * (xy[..., 1] - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0]) - 1.0
        return torch.stack([x, y], dim=-1)


class BEVToSparseAdapter(nn.Module):
    """Sample calibrated BEV at sparse anchor centers and refine token features."""

    def __init__(self, channels: int, x_bound=(-15.0, 15.0), y_bound=(-10.0, 50.0)):
        super().__init__()
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.projection = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, sparse_features: torch.Tensor, encoded_boxes: torch.Tensor, bev: torch.Tensor) -> torch.Tensor:
        xy = encoded_boxes[..., :2]
        x = 2.0 * (xy[..., 0] - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0]) - 1.0
        y = 2.0 * (xy[..., 1] - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0]) - 1.0
        grid = torch.stack([x, y], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(bev, grid, align_corners=True, padding_mode="zeros")
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return sparse_features + self.projection(sampled)


def decoded_bev_boxes(encoded_boxes: torch.Tensor) -> torch.Tensor:
    """Return ``[x, y, width, length, yaw]`` predicted BEV boxes."""

    yaw = torch.atan2(encoded_boxes[..., 6], encoded_boxes[..., 7])
    return torch.cat([encoded_boxes[..., :2], encoded_boxes[..., 3:5].exp(), yaw[..., None]], dim=-1)


def rasterized_oriented_iou(
    candidate_xy: torch.Tensor,
    object_boxes: torch.Tensor,
    object_valid: torch.Tensor,
    x_bound=(-15.0, 15.0, 0.5),
    y_bound=(-10.0, 50.0, 0.5),
    ego_width: float = 1.85,
    ego_length: float = 4.084,
) -> torch.Tensor:
    """Compute BEV-cell IoU for candidate footprints and predicted objects.

    Candidate trajectories are ``[B,Z,T,2]`` in the DiffusionDrive planning
    frame (+y forward). Object boxes are ``[B,O,5]`` in the same lidar frame.
    The output is ``[B,Z,T,O]`` and uses no camera 2D boxes.
    """

    batch, candidates, steps = candidate_xy.shape[:3]
    objects = object_boxes.shape[1]
    xs = torch.arange(x_bound[0] + x_bound[2] * 0.5, x_bound[1], x_bound[2], device=candidate_xy.device)
    ys = torch.arange(y_bound[0] + y_bound[2] * 0.5, y_bound[1], y_bound[2], device=candidate_xy.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).to(candidate_xy.dtype)
    grid = grid.reshape(1, 1, 1, 1, -1, 2)
    start = torch.cat([torch.zeros_like(candidate_xy[..., :1, :]), candidate_xy[..., :-1, :]], dim=-2)
    direction = candidate_xy - start
    yaw = torch.atan2(direction[..., 1], direction[..., 0])
    relative = grid - candidate_xy[:, :, :, None, None, :]
    cos_yaw = torch.cos(yaw)[..., None, None]
    sin_yaw = torch.sin(yaw)[..., None, None]
    local_x = cos_yaw * relative[..., 0] + sin_yaw * relative[..., 1]
    local_y = -sin_yaw * relative[..., 0] + cos_yaw * relative[..., 1]
    ego_mask = (local_x.abs() <= ego_length * 0.5) & (local_y.abs() <= ego_width * 0.5)
    object_relative = grid - object_boxes[:, None, None, :, None, :2]
    object_yaw = object_boxes[:, None, None, :, None, 4]
    object_cos = torch.cos(object_yaw)
    object_sin = torch.sin(object_yaw)
    object_x = object_cos * object_relative[..., 0] + object_sin * object_relative[..., 1]
    object_y = -object_sin * object_relative[..., 0] + object_cos * object_relative[..., 1]
    object_mask = (
        (object_x.abs() <= object_boxes[:, None, None, :, None, 3] * 0.5)
        & (object_y.abs() <= object_boxes[:, None, None, :, None, 2] * 0.5)
        & object_valid[:, None, None, :, None]
    )
    intersection = (ego_mask & object_mask).sum(dim=-1).to(candidate_xy.dtype)
    union = (ego_mask | object_mask).sum(dim=-1).to(candidate_xy.dtype)
    return intersection / union.clamp_min(1.0)
