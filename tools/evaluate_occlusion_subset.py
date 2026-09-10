"""Evaluate planning results on the 30 m visibility 1/2/3 subset.

Inference is run on the complete nuScenes validation sequence because the
dataset needs following frames to construct future annotations.  This script
filters the completed results back to the requested target tokens before
computing the official DiffusionDrive planning metrics.
"""

import argparse
import json
import sys

import mmcv
import numpy as np
import torch
import numpy.core
import numpy.core.numeric
import numpy.core.multiarray

# Some subset pickles were written with NumPy 2.x module names while the
# official evaluation environment uses NumPy 1.x. Register compatible aliases
# before mmcv/pickle loads an annotation file.
sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)

from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset

from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import (
    PlanningMetric,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("results")
    parser.add_argument("ann_file")
    parser.add_argument("target_ann_file")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--target-tokens-file",
        default=None,
        help="Optional newline-delimited sample tokens. If set, use these tokens instead of loading target_ann_file.",
    )
    parser.add_argument("--subset-name", default=None)
    parser.add_argument("--radius-m", type=float, default=30.0)
    parser.add_argument("--visibility-tokens", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    # The evaluation pipeline, unlike the inference pipeline, collects the
    # future boxes needed by the official collision metric.
    test_cfg = cfg.eval_config.copy()
    test_cfg.ann_file = args.ann_file
    if args.data_root is not None:
        test_cfg.data_root = args.data_root
    dataset = build_dataset(test_cfg)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        shuffle=False,
        dist=False,
    )

    full_infos = mmcv.load(args.ann_file)["infos"]
    if args.target_tokens_file is not None:
        with open(args.target_tokens_file, "r", encoding="utf-8") as file:
            target_tokens = {line.strip() for line in file if line.strip()}
        target_source = args.target_tokens_file
    else:
        target_infos = mmcv.load(args.target_ann_file)["infos"]
        target_tokens = {info["token"] for info in target_infos}
        target_source = args.target_ann_file
    selected_indices = {
        i for i, info in enumerate(full_infos) if info["token"] in target_tokens
    }
    results = mmcv.load(args.results)
    if len(results) != len(full_infos):
        raise RuntimeError(
            "result count ({}) does not match annotation count ({})".format(
                len(results), len(full_infos)
            )
        )

    metric = PlanningMetric()
    selected = 0
    valid = 0
    for i, data in enumerate(dataloader):
        if i not in selected_indices:
            continue
        selected += 1
        gt = data["gt_ego_fut_trajs"].cumsum(dim=-2).unsqueeze(1)
        mask = data["gt_ego_fut_masks"].unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)
        if not bool(mask.all()):
            continue
        pred_result = results[i]
        if "img_bbox" in pred_result:
            pred = pred_result["img_bbox"]["final_planning"]
        elif "pts_bbox" in pred_result:
            pred = pred_result["pts_bbox"]["ego_fut_preds"]
            # ResWorld stores one trajectory per command rather than a
            # preselected final trajectory. Select the original NuScenes
            # command from the current sample.
            if pred.ndim == 3:
                command = data["gt_ego_fut_cmd"].argmax(dim=-1).item()
                pred = pred[command]
                # ResWorld exports per-step displacements; the official
                # planning metric consumes cumulative positions.
                pred = pred.cumsum(dim=-2)
        else:
            pred = pred_result["final_planning"]
        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred)
        pred = pred.unsqueeze(0) if pred.ndim == 2 else pred
        metric.update(
            pred[:, :6, :2].float(),
            gt[0, :, :6, :2].float(),
            mask[0, :, :6, :2].float(),
            data["fut_boxes"],
        )
        valid += 1

    raw = metric.compute()
    raw = {key: value.detach().cpu().numpy().tolist() for key, value in raw.items()}

    def cumulative(values):
        values = np.asarray(values, dtype=float)
        return [float(values[: i + 1].mean()) for i in range(len(values))]

    metrics = {}
    for key, values in raw.items():
        cum = cumulative(values)
        metrics[key + "_1s"] = cum[1]
        metrics[key + "_2s"] = cum[3]
        metrics[key + "_3s"] = cum[5]
        metrics[key + "_avg"] = float(np.mean([cum[1], cum[3], cum[5]]))

    output = {
        "criteria": {
            "target_ann_file": target_source,
            "radius_m": args.radius_m,
            "visibility_tokens": args.visibility_tokens,
            "subset": args.subset_name
            or ("full_nuscenes_val" if len(target_tokens) == len(full_infos) else "target_token_subset"),
        },
        "counts": {
            "target_tokens": len(target_tokens),
            "selected": selected,
            "valid": valid,
        },
        "metrics": metrics,
        "raw_per_step": raw,
    }
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
