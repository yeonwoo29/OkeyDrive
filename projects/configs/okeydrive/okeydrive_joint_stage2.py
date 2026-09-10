"""Stage D: jointly fine-tune perception, BEV fusion, and Mamba planning."""

_base_ = ["./okeydrive_small_stage2.py"]

ablation_name = "okeydrive_joint_finetuning"
load_from = "checkpoints/okeydrive_initial_stage2.pth"
