"""Validate an upstream checkpoint against OkeyDrive without leaking paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from okeydrive.runtime_privacy import install_redacting_excepthook

install_redacting_excepthook()

import torch
from mmcv import Config
from mmdet.models import build_detector

from okeydrive.checkpoint import categorize_checkpoint_keys, state_dict_sha256
import projects.mmdet3d_plugin  # noqa: F401


def _state_dict(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not all(
        isinstance(name, str) and torch.is_tensor(value) for name, value in payload.items()
    ):
        raise RuntimeError("checkpoint has no supported model state")
    if payload and all(name.startswith("module.") for name in payload):
        payload = {name[7:]: value for name, value in payload.items()}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = Config.fromfile(str(args.config))
    cfg.model.pretrained = None
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    state_dict = _state_dict(torch.load(args.checkpoint, map_location="cpu"))
    incompatible = model.load_state_dict(state_dict, strict=False)
    categories = categorize_checkpoint_keys(
        incompatible.missing_keys, incompatible.unexpected_keys
    )
    report = {
        "passed": not categories["incompatible_missing"] and not categories["unexpected"],
        "config": args.config.name,
        "checkpoint_sha256": state_dict_sha256(state_dict),
        **categories,
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
