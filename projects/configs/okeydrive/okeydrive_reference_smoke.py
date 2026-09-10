"""CPU/reference-scan smoke configuration; not a performance configuration."""

_base_ = ["./okeydrive_small_stage2.py"]

ablation_name = "full_okeydrive_reference_smoke"

model = dict(
    head=dict(
        okeydrive=dict(clip_local_files_only=True),
        motion_plan_head=dict(
            okeydrive_refiner=dict(scan_backend="reference"),
        ),
    )
)
