"""Sensitivity run with keypoint visibility threshold eta=0.65."""

_base_ = ["../okeydrive_small_stage2.py"]

ablation_name = "sensitivity_eta_065"

model = dict(head=dict(okeydrive=dict(visibility_threshold=0.65)))
