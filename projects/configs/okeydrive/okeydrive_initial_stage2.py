"""Stage C: initialize OkeyDrive from a compatible DiffusionDrive checkpoint."""

_base_ = ["./okeydrive_small_stage2.py"]

ablation_name = "okeydrive_initial_training"
load_from = "checkpoints/diffusiondrive_nusc_stage2.pth"

model = dict(
    head=dict(
        okeydrive=dict(
            pretrained_keypoint_checkpoint="outputs/pretrain/last_training_checkpoint.pt"
        )
    )
)

optimizer = dict(
    type="AdamW",
    lr=3e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.0),
            "head.det_head": dict(lr_mult=0.1),
            "head.map_head": dict(lr_mult=0.1),
            "head.motion_plan_head.interact_layers": dict(lr_mult=0.1),
        }
    ),
)
