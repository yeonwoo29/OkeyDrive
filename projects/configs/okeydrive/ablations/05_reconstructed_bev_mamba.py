"""Ablation 5: AE-only keypoints, BEV, and Mamba without modulation."""

_base_ = ["../okeydrive_small_stage2.py"]

ablation_name = "reconstructed_bev_mamba"

model = dict(
    head=dict(
        okeydrive=dict(keypoint_source="reconstructed"),
        motion_plan_head=dict(okeydrive_refiner=dict(use_modulation=False)),
    )
)
