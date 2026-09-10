"""Ablation 6: visibility-gated keypoints and BEV without Mamba modulation."""

_base_ = ["../okeydrive_small_stage2.py"]

ablation_name = "recovered_bev_mamba"

model = dict(
    head=dict(
        okeydrive=dict(keypoint_source="recovered"),
        motion_plan_head=dict(okeydrive_refiner=dict(use_modulation=False)),
    )
)
