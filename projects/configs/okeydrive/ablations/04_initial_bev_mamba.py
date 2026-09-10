"""Ablation 4: initial keypoints, BEV, and Mamba without modulation."""

_base_ = ["../okeydrive_small_stage2.py"]

ablation_name = "initial_bev_mamba"

model = dict(
    head=dict(
        okeydrive=dict(keypoint_source="initial"),
        motion_plan_head=dict(okeydrive_refiner=dict(use_modulation=False)),
    )
)
