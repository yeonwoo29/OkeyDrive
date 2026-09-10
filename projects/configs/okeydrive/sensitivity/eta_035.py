"""Sensitivity run with keypoint visibility threshold eta=0.35."""

_base_ = ["../okeydrive_small_stage2.py"]

ablation_name = "sensitivity_eta_035"

model = dict(head=dict(okeydrive=dict(visibility_threshold=0.35)))
