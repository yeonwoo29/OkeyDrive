"""Core OkeyDrive modules with no MMDetection import-time dependency."""

from .annotations import (
    CANONICAL_PART_NAMES,
    CARFUSION_KEYPOINT_NAMES,
    COCO_PERSON_SOURCE_INDICES,
    build_part_visibility_targets,
    canonicalize_carfusion,
    canonicalize_coco_person,
)
from .geometry import GeometryAwareBEV, project_sparse_boxes_to_views
from .keypoints import (
    ClassConditionedKeypointAutoencoder,
    KeypointLocalizer,
    recover_keypoints,
)
from .mamba import ContextMotionMambaRefiner, VisibilityConditionedMambaBlock
from .pipeline import OkeyDriveEnhancer
from .visibility import FineGrainedCLIPVisibility, remap_semantic_parts_for_flip

__all__ = [
    "CANONICAL_PART_NAMES",
    "CARFUSION_KEYPOINT_NAMES",
    "COCO_PERSON_SOURCE_INDICES",
    "ClassConditionedKeypointAutoencoder",
    "ContextMotionMambaRefiner",
    "FineGrainedCLIPVisibility",
    "GeometryAwareBEV",
    "KeypointLocalizer",
    "OkeyDriveEnhancer",
    "VisibilityConditionedMambaBlock",
    "build_part_visibility_targets",
    "canonicalize_carfusion",
    "canonicalize_coco_person",
    "project_sparse_boxes_to_views",
    "recover_keypoints",
    "remap_semantic_parts_for_flip",
]
