"""Ablation 2: recovered keypoints and BEV with the upstream diffusion refiner."""

_base_ = ["../okeydrive_small_stage2.py"]

ablation_name = "recovery_bev_diffusion"

model = dict(
    head=dict(
        motion_plan_head=dict(
            okeydrive_refiner=None,
            planning_sampler=dict(ego_fut_mode=6),
            planning_decoder=dict(ego_fut_mode=6),
        ),
    )
)
