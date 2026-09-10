"""Ablation 3: Mamba candidate refiner without keypoint/BEV or modulation."""

_base_ = ["../okeydrive_small_stage2.py"]

ablation_name = "mamba_only"

model = dict(
    head=dict(
        okeydrive=dict(
            use_keypoint_fusion=False,
            use_bev_fusion=False,
            use_perception_refinement=False,
        ),
        motion_plan_head=dict(
            okeydrive_refiner=dict(use_modulation=False),
        ),
    )
)
