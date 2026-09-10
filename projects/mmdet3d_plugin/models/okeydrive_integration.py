"""MMDetection integration for the OkeyDrive feature enhancement path."""

from __future__ import annotations

from typing import List, Union

import torch

from mmdet.models import DETECTORS, HEADS

from okeydrive.pipeline import OkeyDriveEnhancer
from projects.mmdet3d_plugin.ops import feature_maps_format

from .sparsedrive_head_v1 import V1SparseDriveHead
from .sparsedrive_v1 import V1SparseDrive


def _horizontal_flip_from_metas(metas: dict, batch: int, device: torch.device) -> torch.Tensor:
    records = metas.get("img_metas", [])
    values = []
    for index in range(batch):
        record = records[index] if index < len(records) else {}
        values.append(bool(record.get("horizontal_flip", False)))
    return torch.tensor(values, device=device, dtype=torch.bool)


@HEADS.register_module()
class OkeyDriveSparseHead(V1SparseDriveHead):
    """Run initial proposals, enhance image/BEV features, then map and planning."""

    def __init__(self, okeydrive: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.okeydrive = OkeyDriveEnhancer(**okeydrive)

    @staticmethod
    def _to_dense(feature_maps):
        is_formatted = (
            isinstance(feature_maps, (list, tuple))
            and len(feature_maps) == 3
            and torch.is_tensor(feature_maps[0])
            and feature_maps[0].ndim == 3
        )
        if not is_formatted:
            return list(feature_maps), False
        camera_groups = feature_maps_format(feature_maps, inverse=True)
        if len(camera_groups) != 1:
            raise RuntimeError("OkeyDrive currently requires equal FPN shapes across camera views")
        return list(camera_groups[0]), True

    def _refine_detection(self, det_output):
        if not self.okeydrive.use_perception_refinement:
            return det_output
        refine_indices = [
            index for index, operation in enumerate(self.det_head.operation_order) if operation == "refine"
        ]
        if not refine_indices:
            raise RuntimeError("the detector has no refinement layer for the OkeyDrive adapter")
        anchor = det_output["prediction"][-1]
        anchor_embed = self.det_head.anchor_encoder(anchor)
        prediction, classification, quality = self.det_head.layers[refine_indices[-1]](
            det_output["instance_feature"],
            anchor,
            anchor_embed,
            time_interval=det_output.get("time_interval", 1.0),
            return_cls=True,
        )
        det_output["prediction"].append(prediction)
        det_output["classification"].append(classification)
        det_output["quality"].append(quality)
        det_output["anchor_embed"] = self.det_head.anchor_encoder(prediction)
        return det_output

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        images: torch.Tensor | None = None,
        depths=None,
    ):
        if images is None:
            raise RuntimeError("OkeyDriveSparseHead requires aligned normalized RGB images")
        det_output = self.det_head(feature_maps, metas) if self.task_config["with_det"] else None
        if det_output is None:
            raise RuntimeError("OkeyDrive requires the upstream initial detector")
        dense_features, was_formatted = self._to_dense(feature_maps)
        horizontal_flip = _horizontal_flip_from_metas(
            metas, images.shape[0], images.device
        )
        enhanced_dense, det_output, context = self.okeydrive(
            dense_features,
            images,
            depths,
            det_output,
            metas["projection_mat"],
            metas["image_wh"],
            horizontal_flip,
        )
        det_output = self._refine_detection(det_output)
        enhanced_features = feature_maps_format(enhanced_dense) if was_formatted else enhanced_dense
        self.det_head.instance_bank.cache(
            det_output["instance_feature"],
            det_output["prediction"][-1],
            det_output["classification"][-1],
            metas,
            enhanced_features,
        )
        map_output = (
            self.map_head(enhanced_features, metas) if self.task_config["with_map"] else None
        )
        if self.task_config["with_motion_plan"]:
            planner_metas = dict(metas)
            planner_metas["okeydrive_context"] = context
            motion_output, planning_output = self.motion_plan_head(
                det_output,
                map_output,
                enhanced_features,
                planner_metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
            )
        else:
            motion_output, planning_output = None, None
        return det_output, map_output, motion_output, planning_output


@DETECTORS.register_module()
class OkeyDriveSparseDrive(V1SparseDrive):
    """Pass predicted depth and aligned RGB to the integrated OkeyDrive head."""

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data, images=img, depths=depths)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(depths, data["gt_depth"])
        return output

    def simple_test(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data, images=img, depths=depths)
        results = self.head.post_process(model_outs, data)
        return [{"img_bbox": result} for result in results]
