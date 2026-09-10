from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from okeydrive.data import ExternalKeypointDataset
from okeydrive.pretraining import OkeyDrivePretrainingModel
from okeydrive.visibility import VisibilityOutput


class TinyVisibility(nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.image_projection = nn.Linear(3, self.output_dim)
        self.part_projection = nn.Linear(self.output_dim, 4)

    def forward(self, crops, class_ids, valid_mask=None, horizontal_flip=False):
        del horizontal_flip
        pooled = crops.mean(dim=(-1, -2))
        embeddings = self.image_projection(pooled)
        probabilities = torch.sigmoid(self.part_projection(embeddings))
        if valid_mask is None:
            valid_mask = torch.ones_like(class_ids, dtype=torch.bool)
        probabilities = probabilities * valid_mask[..., None]
        embeddings = embeddings * valid_mask[..., None]
        return VisibilityOutput(probabilities, embeddings, valid_mask)


class DataAndTrainingTests(unittest.TestCase):
    def test_external_dataset_reads_real_coco_format_and_keeps_state_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.full((20, 20, 3), 127, dtype=np.uint8)
            Image.fromarray(image).save(root / "sample.png")
            keypoints = []
            for index in range(12):
                keypoints.extend([5 + index * 0.2, 6 + index * 0.2, index % 3])
            payload = {
                "images": [{"id": 1, "file_name": "sample.png"}],
                "annotations": [{"image_id": 1, "bbox": [2, 3, 15, 14], "keypoints": keypoints}],
            }
            annotation = root / "annotations.json"
            annotation.write_text(json.dumps(payload), encoding="utf-8")
            dataset = ExternalKeypointDataset(annotation, root, "vehicle", training=True)
            with patch("okeydrive.data.random.random", return_value=1.0):
                sample = dataset[0]
            self.assertEqual(sample["crop"].shape, (3, 224, 224))
            self.assertEqual(sample["keypoints"].shape, (12, 2))
            self.assertTrue(torch.equal(sample["keypoint_mask"], sample["keypoint_states"] > 0))
            self.assertTrue((sample["keypoints"][~sample["keypoint_mask"]] == 0).all())
            self.assertFalse(sample["horizontal_flip"].item())

    def test_train_eval_backward_and_checkpoint_round_trip(self):
        torch.manual_seed(10)
        model = OkeyDrivePretrainingModel(visibility_module=TinyVisibility())
        batch = {
            "crop": torch.rand(2, 3, 32, 32),
            "class_id": torch.tensor([0, 1]),
            "keypoints": torch.rand(2, 12, 2),
            "keypoint_mask": torch.tensor([[True] * 10 + [False] * 2, [True] * 12]),
            "part_visibility": torch.rand(2, 4),
            "part_mask": torch.tensor([[True, True, False, True], [True] * 4]),
        }
        model.train()
        output = model(batch)
        loss = sum(output[name] for name in output if name.startswith("loss_"))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.visibility.image_projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.localizer.net[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.autoencoder.encoder[0].weight.grad.abs().sum()), 0.0)

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = OkeyDrivePretrainingModel(visibility_module=TinyVisibility())
        incompatible = restored.load_state_dict(torch.load(buffer, map_location="cpu"), strict=False)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        model.eval()
        restored.eval()
        with torch.no_grad():
            expected = model(batch, corruption_probability=0.0)
            actual = restored(batch, corruption_probability=0.0)
        self.assertTrue(torch.equal(expected["initial"], actual["initial"]))
        self.assertTrue(torch.equal(expected["reconstructed"], actual["reconstructed"]))


if __name__ == "__main__":
    unittest.main()
