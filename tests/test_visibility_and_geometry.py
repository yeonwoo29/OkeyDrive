from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from okeydrive.geometry import (
    BEVToSparseAdapter,
    GeometryAwareBEV,
    KeypointForegroundFusion,
    crop_tensor_rois,
    project_sparse_boxes_to_views,
    rasterized_oriented_iou,
    scatter_roi_residuals,
)
from okeydrive.visibility import FineGrainedCLIPVisibility


class FakeTokenizer:
    def __call__(self, prompts, padding, return_tensors):
        del padding, return_tensors
        return {"input_ids": torch.arange(1, len(prompts) + 1).view(-1, 1)}


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(1, 1)
        self.visual_projection = nn.Linear(1, 1)
        self.text_model = nn.Linear(1, 1)
        self.text_projection = nn.Linear(1, 1)
        self.config = types.SimpleNamespace(projection_dim=4)
        self.image_calls = 0
        self.text_calls = 0

    def get_image_features(self, pixel_values):
        self.image_calls += 1
        mean = pixel_values.mean(dim=(1, 2, 3))
        return torch.stack([mean + 1.0, mean + 2.0, mean + 3.0, mean + 4.0], dim=-1)

    def get_text_features(self, input_ids):
        self.text_calls += 1
        value = input_ids[:, 0].to(torch.float32)
        return torch.stack([value + 1.0, value + 2.0, value + 3.0, value + 4.0], dim=-1)


class VisibilityAndGeometryTests(unittest.TestCase):
    def test_clip_uses_both_encoders_and_caches_only_frozen_text(self):
        clip = FakeCLIP()
        module = FineGrainedCLIPVisibility(
            finetune_image=False,
            finetune_text=False,
            model=clip,
            tokenizer=FakeTokenizer(),
        )
        crops = torch.rand(2, 3, 32, 32)
        classes = torch.tensor([0, 1])
        normal = module(crops, classes).probabilities
        flipped = module(crops, classes, horizontal_flip=torch.tensor([True, False])).probabilities
        self.assertEqual(clip.image_calls, 2)
        self.assertEqual(clip.text_calls, 1)
        self.assertTrue(torch.allclose(flipped[0], normal[0, [1, 0, 3, 2]]))
        self.assertTrue(torch.all((normal >= 0.0) & (normal <= 1.0)))
        self.assertFalse(torch.allclose(normal.sum(-1), torch.ones(2)))

    def test_box_projection_and_roi_scatter(self):
        boxes = torch.zeros(1, 1, 8)
        boxes[..., 2] = 10.0
        boxes[..., 3:6] = torch.log(torch.tensor([2.0, 2.0, 2.0]))
        boxes[..., 7] = 1.0
        projection = torch.tensor(
            [[[[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 50.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]]
        )
        image_wh = torch.tensor([[[100.0, 100.0]]])
        projected, valid = project_sparse_boxes_to_views(
            boxes, projection, image_wh, torch.ones(1, 1, dtype=torch.bool)
        )
        self.assertTrue(valid.item())
        image = torch.arange(100, dtype=torch.float32).reshape(1, 1, 1, 10, 10)
        crops, crop_valid = crop_tensor_rois(image, projected, image_wh, 3, valid)
        self.assertEqual(crops.shape, (1, 1, 1, 1, 3, 3))
        residual = torch.ones_like(crops)
        scattered = scatter_roi_residuals(image, projected, image_wh, residual, crop_valid)
        self.assertEqual(scattered[0, 0, 0, 0, 0], image[0, 0, 0, 0, 0])
        self.assertGreater(float((scattered - image).abs().sum()), 0.0)

    def test_invalid_crop_is_masked_to_zero(self):
        image = torch.ones(1, 1, 2, 8, 8)
        boxes = torch.zeros(1, 1, 1, 4)
        crops, valid = crop_tensor_rois(
            image, boxes, torch.tensor([[[8.0, 8.0]]]), 3
        )
        self.assertFalse(valid.item())
        self.assertEqual(float(crops.abs().sum()), 0.0)

    def test_calibrated_depth_lifts_to_nonempty_bev(self):
        encoder = GeometryAwareBEV(
            channels=8, x_bound=(-3.0, 3.0, 1.0), y_bound=(-3.0, 3.0, 1.0)
        )
        features = torch.ones(1, 1, 8, 2, 2)
        depths = torch.full((1, 1, 1, 2, 2), 10.0)
        projection = torch.tensor(
            [[[[10.0, 0.0, 2.0, 0.0], [0.0, 10.0, 2.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]]
        )
        bev, valid = encoder(features, depths, projection, torch.tensor([[[4.0, 4.0]]]))
        self.assertEqual(bev.shape, (1, 8, 6, 6))
        self.assertTrue(valid.any())
        self.assertTrue(torch.isfinite(bev).all())

    def test_rasterized_oriented_iou_is_geometric(self):
        trajectories = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])
        objects = torch.tensor([[[0.0, 0.0, 2.0, 4.0, 0.0]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        iou = rasterized_oriented_iou(
            trajectories, objects, valid, x_bound=(-3.0, 3.0, 0.5), y_bound=(-3.0, 3.0, 0.5)
        )
        self.assertEqual(iou.shape, (1, 1, 2, 1))
        self.assertGreater(float(iou[0, 0, 0, 0]), 0.5)

    def test_fusion_and_sparse_adapter_are_trainable_residual_paths(self):
        torch.manual_seed(3)
        fusion = KeypointForegroundFusion(channels=8, token_dim=8, num_heads=2)
        roi = torch.randn(1, 1, 1, 8, 2, 2, requires_grad=True)
        keypoints = torch.rand(1, 1, 1, 12, 2, requires_grad=True)
        fused = fusion(
            roi,
            keypoints,
            torch.rand(1, 1, 1, 4),
            torch.ones(1, 1, 1, 12, 1, dtype=torch.bool),
            torch.zeros(1, 1, 1, dtype=torch.long),
            torch.ones(1, 1, 1, dtype=torch.bool),
        )
        self.assertFalse(torch.equal(fused, roi))
        fused.sum().backward()
        self.assertGreater(float(keypoints.grad.abs().sum()), 0.0)

        adapter = BEVToSparseAdapter(8, x_bound=(-2.0, 2.0), y_bound=(-2.0, 2.0))
        with torch.no_grad():
            adapter.projection[-1].weight.fill_(0.1)
        sparse = torch.zeros(1, 2, 8)
        encoded = torch.zeros(1, 2, 8)
        bev = torch.ones(1, 8, 4, 4)
        self.assertGreater(float(adapter(sparse, encoded, bev).abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
