"""Pretrain visibility, keypoint localization, and reconstruction on real annotations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from okeydrive.runtime_privacy import install_redacting_excepthook

install_redacting_excepthook()

import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

from okeydrive.checkpoint import state_dict_sha256
from okeydrive.data import ExternalKeypointDataset
from okeydrive.pretraining import OkeyDrivePretrainingModel


def seed_worker(worker_id):
    """Seed Python and NumPy from the DataLoader-assigned worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carfusion-annotations", type=Path)
    parser.add_argument("--carfusion-images", type=Path)
    parser.add_argument("--coco-annotations", type=Path)
    parser.add_argument("--coco-images", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("OUTPUT_ROOT", "outputs")) / "pretrain")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--visibility-weight", type=float, default=1.0)
    parser.add_argument("--localization-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--freeze-image", action="store_true")
    parser.add_argument("--freeze-text", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    datasets = []
    if args.carfusion_annotations and args.carfusion_images:
        datasets.append(ExternalKeypointDataset(args.carfusion_annotations, args.carfusion_images, "vehicle"))
    if args.coco_annotations and args.coco_images:
        datasets.append(ExternalKeypointDataset(args.coco_annotations, args.coco_images, "pedestrian"))
    if not datasets:
        raise ValueError("at least one real CarFusion or COCO dataset pair is required")
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OkeyDrivePretrainingModel(
        finetune_image=not args.freeze_image,
        finetune_text=not args.freeze_text,
        local_files_only=args.local_files_only,
    ).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        result = model.load_state_dict(checkpoint["model"], strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("resume checkpoint is structurally incompatible")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(batch)
                loss = (
                    args.visibility_weight * output["loss_visibility"]
                    + args.localization_weight * output["loss_keypoint_localization"]
                    + args.reconstruction_weight * output["loss_keypoint_reconstruction"]
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach())
        scheduler.step()
        state = model.state_dict()
        checkpoint = {
            "model": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metadata": {
                "format": "okeydrive_pretraining_v1",
                "seed": args.seed,
                "deterministic": args.deterministic,
                "model_sha256": state_dict_sha256(state),
                "mean_training_loss": epoch_loss / max(len(loader), 1),
                "loss_weights": {
                    "visibility": args.visibility_weight,
                    "keypoint_localization": args.localization_weight,
                    "keypoint_reconstruction": args.reconstruction_weight,
                },
            },
        }
        torch.save(checkpoint, args.output_dir / "last_training_checkpoint.pt")
        with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(checkpoint["metadata"], stream, indent=2)


if __name__ == "__main__":
    main()
