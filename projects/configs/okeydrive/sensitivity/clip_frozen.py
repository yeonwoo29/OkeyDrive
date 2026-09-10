"""Sensitivity run with both pretrained CLIP encoders frozen."""

_base_ = ["../okeydrive_small_stage2.py"]

ablation_name = "sensitivity_clip_frozen"

model = dict(
    head=dict(
        okeydrive=dict(
            clip_finetune_image=False,
            clip_finetune_text=False,
        )
    )
)
