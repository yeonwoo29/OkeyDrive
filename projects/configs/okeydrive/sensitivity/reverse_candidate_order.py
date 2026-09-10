"""Sensitivity run with the canonical candidate sequence reversed."""

_base_ = ["../okeydrive_small_stage2.py"]

ablation_name = "sensitivity_reverse_candidate_order"

model = dict(
    head=dict(
        motion_plan_head=dict(
            okeydrive_refiner=dict(order_mode="reversed")
        )
    )
)
