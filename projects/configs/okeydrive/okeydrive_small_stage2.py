"""Full OkeyDrive nuScenes stage-2 configuration.

The inherited upstream head keeps six command-conditioned DiffusionDrive
anchors. A learned interpolation over all six modes creates ten stable OkeyDrive
candidates per command; no anchor prefix is truncated and no validation data is
used to construct anchors.
"""

_base_ = ["../diffusiondrive_configs/diffusiondrive_small_stage2.py"]

ablation_name = "full_okeydrive"
experiment_seeds = [0, 1, 2]

model = dict(
    type="OkeyDriveSparseDrive",
    head=dict(
        type="OkeyDriveSparseHead",
        okeydrive=dict(
            channels=256,
            max_objects=20,
            proposal_score_threshold=0.2,
            roi_size=7,
            visibility_threshold=0.5,
            keypoint_source="recovered",
            use_keypoint_fusion=True,
            use_bev_fusion=True,
            use_perception_refinement=True,
            x_bound=(-15.0, 15.0, 0.5),
            y_bound=(-10.0, 50.0, 0.5),
            clip_finetune_image=True,
            clip_finetune_text=True,
            clip_local_files_only=False,
            clip_batch_size=16,
        ),
        motion_plan_head=dict(
            okeydrive_refiner=dict(
                model_dim=256,
                future_steps=6,
                base_modes=6,
                num_candidates=10,
                state_dim=16,
                scan_backend="cuda",
                use_modulation=True,
                order_mode="canonical",
                x_bound=(-15.0, 15.0, 0.5),
                y_bound=(-10.0, 50.0, 0.5),
            ),
            planning_sampler=dict(
                type="V1PlanningTarget", ego_fut_ts=6, ego_fut_mode=10
            ),
            planning_decoder=dict(
                type="HierarchicalPlanningDecoder",
                ego_fut_ts=6,
                ego_fut_mode=10,
                use_rescore=True,
            ),
        ),
    ),
)

# Tracking remains local. These hooks perform no external upload.
log_config = dict(interval=51, hooks=[dict(type="TextLoggerHook", by_epoch=False)])
