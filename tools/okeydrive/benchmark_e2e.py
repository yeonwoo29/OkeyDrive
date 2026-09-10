"""Measure full OkeyDrive model-forward latency on real nuScenes samples."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from okeydrive.runtime_privacy import install_redacting_excepthook

install_redacting_excepthook()

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

import projects.mmdet3d_plugin  # noqa: F401
from okeydrive.checkpoint import file_sha256


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def image_shape(batch) -> list[int]:
    value = batch.get("img")
    if isinstance(value, (list, tuple)):
        value = value[0]
    if hasattr(value, "data"):
        value = value.data
        if isinstance(value, (list, tuple)):
            value = value[0]
    if not torch.is_tensor(value) or value.ndim < 4:
        raise RuntimeError("unable to determine the measured input tensor shape")
    return list(value.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the integrated benchmark requires a CUDA device")
    if args.warmup < 1 or args.samples < 1:
        raise ValueError("warmup and samples must be positive")

    cfg = Config.fromfile(str(args.config))
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    samples_per_gpu = int(cfg.data.test.pop("samples_per_gpu", 1))
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])
    iterator = iter(loader)
    shape = None
    latencies_ms = []
    with torch.inference_mode():
        for index in range(args.warmup + args.samples):
            batch, iterator = next_batch(iterator, loader)
            if shape is None:
                shape = image_shape(batch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            model(return_loss=False, rescale=True, **batch)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if index >= args.warmup:
                latencies_ms.append(elapsed_ms)

    ordered = sorted(latencies_ms)
    percentile_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    report = {
        "status": "measured",
        "scope": "complete_model_forward_excluding_data_loading",
        "includes": [
            "backbone_and_fpn",
            "initial_proposals",
            "clip_vit_b16",
            "keypoint_recovery_and_roi_fusion",
            "calibrated_bev_lifting_and_sparse_adapter",
            "detection_and_map_heads",
            "candidate_aggregation_and_mamba1",
            "postprocessing",
        ],
        "synchronization": "before_and_after_each_forward",
        "warmup_forwards": args.warmup,
        "measured_forwards": args.samples,
        "batch_size": samples_per_gpu,
        "input_tensor_shape": shape,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95": ordered[percentile_index],
            "minimum": min(latencies_ms),
            "maximum": max(latencies_ms),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mmcv": mmcv.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "accelerator": {
            "name": torch.cuda.get_device_name(0),
            "count": 1,
        },
        "checkpoint_sha256": file_sha256(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("benchmark_status=measured")


if __name__ == "__main__":
    main()
