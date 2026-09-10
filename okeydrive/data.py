"""COCO-format CarFusion and COCO person datasets for external pretraining."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .annotations import (
    build_part_visibility_targets,
    canonicalize_carfusion,
    canonicalize_coco_person,
)


class ExternalKeypointDataset(Dataset):
    """Read real COCO-style keypoint annotations and object RGB crops."""

    def __init__(
        self,
        annotation_file: str | Path,
        image_root: str | Path,
        class_name: str,
        training: bool = True,
        output_size: int = 224,
    ):
        if class_name not in {"vehicle", "pedestrian"}:
            raise ValueError("class_name must be vehicle or pedestrian")
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.class_name = class_name
        self.class_id = 0 if class_name == "vehicle" else 1
        self.training = training
        self.output_size = output_size
        with self.annotation_file.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        images: Dict[int, dict] = {int(item["id"]): item for item in payload["images"]}
        self.records: List[dict] = []
        for annotation in payload["annotations"]:
            if "keypoints" not in annotation or "bbox" not in annotation:
                continue
            image = images.get(int(annotation["image_id"]))
            if image is not None:
                self.records.append({"image": image, "annotation": annotation})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        annotation = record["annotation"]
        image_path = self.image_root / record["image"]["file_name"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            x0 = max(0.0, x)
            y0 = max(0.0, y)
            x1 = min(float(image.width), x + width)
            y1 = min(float(image.height), y + height)
            if x1 - x0 <= 1 or y1 - y0 <= 1:
                raise ValueError("annotation contains an invalid object crop")
            crop = image.crop((x0, y0, x1, y1)).resize(
                (self.output_size, self.output_size), Image.Resampling.BICUBIC
            )
        keypoints = torch.tensor(annotation["keypoints"], dtype=torch.float32).reshape(-1, 3)
        if self.class_name == "vehicle":
            coordinates, states = canonicalize_carfusion(keypoints)
        else:
            coordinates, states = canonicalize_coco_person(keypoints)
        coordinates = coordinates.clone()
        coordinates[..., 0] = (coordinates[..., 0] - x0) / max(x1 - x0, 1.0)
        coordinates[..., 1] = (coordinates[..., 1] - y0) / max(y1 - y0, 1.0)
        labeled = states > 0
        coordinates = torch.where(labeled[:, None], coordinates.clamp(0.0, 1.0), torch.zeros_like(coordinates))
        horizontal_flip = self.training and random.random() < 0.5
        if horizontal_flip:
            coordinates[..., 0] = torch.where(labeled, 1.0 - coordinates[..., 0], coordinates[..., 0])
            permutation = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
            coordinates = coordinates[permutation]
            states = states[permutation]
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        part_target, part_mask = build_part_visibility_targets(states)
        crop_array = np.asarray(crop, dtype=np.float32) / 255.0
        crop_tensor = torch.from_numpy(crop_array).permute(2, 0, 1).contiguous()
        return {
            "crop": crop_tensor,
            "class_id": torch.tensor(self.class_id, dtype=torch.long),
            "keypoints": coordinates,
            "keypoint_states": states,
            "keypoint_mask": states > 0,
            "part_visibility": part_target,
            "part_mask": part_mask,
            "horizontal_flip": torch.tensor(horizontal_flip, dtype=torch.bool),
        }
