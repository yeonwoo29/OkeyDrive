from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn

from okeydrive.pipeline import OkeyDriveEnhancer
from okeydrive.mamba import ContextMotionMambaRefiner
from okeydrive.visibility import VisibilityOutput


REPOSITORY = Path(__file__).resolve().parents[1]


class ConstantVisibility(nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, self.output_dim)

    def forward(self, crops, class_ids, valid_mask, horizontal_flip=False):
        del class_ids, horizontal_flip
        embedding = self.encoder(crops.mean(dim=(-1, -2)))
        probability = embedding.new_full((*embedding.shape[:-1], 4), 0.8)
        return VisibilityOutput(
            probability * valid_mask[..., None],
            embedding * valid_mask[..., None],
            valid_mask,
        )


def detection_output():
    classification = torch.full((1, 2, 10), -10.0)
    classification[0, 0, 0] = 10.0
    classification[0, 1, 5] = 10.0
    boxes = torch.zeros(1, 2, 10)
    boxes[..., 2] = 10.0
    boxes[..., 3:6] = torch.log(torch.tensor([2.0, 2.0, 2.0]))
    boxes[..., 7] = 1.0
    return {
        "classification": [classification],
        "prediction": [boxes],
        "quality": [None],
        "instance_feature": torch.zeros(1, 2, 8, requires_grad=True),
    }


class PipelineTests(unittest.TestCase):
    def test_disabled_feature_flags_reproduce_input_path(self):
        enhancer = OkeyDriveEnhancer(
            channels=8,
            use_keypoint_fusion=False,
            use_bev_fusion=False,
            use_perception_refinement=False,
        )
        feature = torch.randn(1, 1, 8, 4, 4)
        detector = detection_output()
        output_features, output_detector, context = enhancer(
            [feature],
            torch.zeros(1, 1, 3, 16, 16),
            None,
            detector,
            torch.eye(4).reshape(1, 1, 4, 4),
            torch.tensor([[[16.0, 16.0]]]),
        )
        self.assertIs(output_detector, detector)
        self.assertTrue(torch.equal(output_features[0], feature))
        self.assertIsNone(context["bev"])
        self.assertEqual(context["object_boxes"].shape[1], 0)

    def test_full_enhancer_changes_dense_features_and_backpropagates_to_keypoints(self):
        torch.manual_seed(11)
        enhancer = OkeyDriveEnhancer(
            channels=8,
            max_objects=2,
            roi_size=3,
            x_bound=(-5.0, 5.0, 1.0),
            y_bound=(-5.0, 15.0, 1.0),
            visibility_module=ConstantVisibility(),
        )
        feature = torch.randn(1, 1, 8, 4, 4, requires_grad=True)
        projection = torch.tensor(
            [[[[10.0, 0.0, 8.0, 0.0], [0.0, 10.0, 8.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]]
        )
        output_features, output_detector, context = enhancer(
            [feature],
            torch.zeros(1, 1, 3, 16, 16),
            [torch.full((1, 1, 4, 4), 10.0)],
            detection_output(),
            projection,
            torch.tensor([[[16.0, 16.0]]]),
        )
        self.assertFalse(torch.equal(output_features[0], feature))
        self.assertEqual(context["bev"].shape, (1, 8, 20, 10))
        self.assertEqual(context["object_valid"].tolist(), [[True, False]])
        self.assertIn("okeydrive_context", output_detector)
        planner = ContextMotionMambaRefiner(
            model_dim=8,
            state_dim=4,
            scan_backend="reference",
            x_bound=(-5.0, 5.0, 1.0),
            y_bound=(-5.0, 15.0, 1.0),
        )
        planning = planner(
            torch.randn(1, 3, 6, 6, 2).cumsum(dim=-2) * 0.05,
            torch.randn(1, 3, 6, 8),
            output_detector["instance_feature"].mean(dim=1),
            context["bev"],
            context["object_boxes"],
            context["object_visibility"],
            context["object_class"],
            context["object_valid"],
        )
        planning["score"].sum().backward()
        gradient = enhancer.localizer.net[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_integrated_head_passes_enhanced_features_to_consumers(self):
        source = (REPOSITORY / "projects/mmdet3d_plugin/models/okeydrive_integration.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.map_head(enhanced_features, metas)", source)
        self.assertIn('planner_metas["okeydrive_context"] = context', source)
        self.assertIn("enhanced_features,\n                planner_metas", source)
        self.assertIn("self.det_head.instance_bank.cache(", source)


if __name__ == "__main__":
    unittest.main()
