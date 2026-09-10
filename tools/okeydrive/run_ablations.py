"""Evaluate all seven OkeyDrive ablations under one fixed protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from okeydrive.runtime_privacy import install_redacting_excepthook

install_redacting_excepthook()

from okeydrive.checkpoint import file_sha256


CONDITIONS = (
    ("01_original_diffusiondrive", "diffusiondrive_nusc_stage2"),
    ("02_recovery_bev_diffusion", "02_recovery_bev_diffusion"),
    ("03_mamba_only", "03_mamba_only"),
    ("04_initial_bev_mamba", "04_initial_bev_mamba"),
    ("05_reconstructed_bev_mamba", "05_reconstructed_bev_mamba"),
    ("06_recovered_bev_mamba", "06_recovered_bev_mamba"),
    ("07_full_okeydrive", "07_full_okeydrive"),
)


def choose_checkpoint(root: Path, stem: str, seed: int) -> Path:
    seeded = root / f"{stem}_seed{seed}.pth"
    shared = root / f"{stem}.pth"
    if seeded.is_file():
        return seeded
    if shared.is_file():
        return shared
    raise FileNotFoundError(f"missing checkpoint for condition={stem} seed={seed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "ablations")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    checkpoint_hashes = {}
    for condition, checkpoint_stem in CONDITIONS:
        config = Path("projects/configs/okeydrive/ablations") / f"{condition}.py"
        for seed in args.seeds:
            checkpoint = choose_checkpoint(args.checkpoint_root, checkpoint_stem, seed)
            metrics = args.output_root / f"{condition}_seed{seed}.json"
            command = [
                sys.executable,
                "tools/test.py",
                str(config),
                str(checkpoint),
                "--eval",
                "bbox",
                "--metrics-out",
                str(metrics),
                "--deterministic",
                "--seed",
                str(seed),
            ]
            subprocess.run(command, check=True)
            measured = json.loads(metrics.read_text(encoding="utf-8"))
            checkpoint_key = str(checkpoint)
            if checkpoint_key not in checkpoint_hashes:
                checkpoint_hashes[checkpoint_key] = file_sha256(checkpoint)
            records.append(
                {
                    "condition": condition,
                    "config": config.name,
                    "seed": seed,
                    "checkpoint_sha256": checkpoint_hashes[checkpoint_key],
                    "metrics": measured,
                }
            )
    aggregate = {}
    for condition, _ in CONDITIONS:
        selected = [record["metrics"] for record in records if record["condition"] == condition]
        common_keys = set.intersection(*(set(item) for item in selected)) if selected else set()
        aggregate[condition] = {}
        for key in sorted(common_keys):
            values = [item[key] for item in selected]
            if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                aggregate[condition][key] = {
                    "mean": statistics.fmean(values),
                    "std_population": statistics.pstdev(values),
                    "count": len(values),
                }
    summary = {
        "protocol": "official_full_validation",
        "statistics": aggregate,
        "runs": records,
    }
    (args.output_root / "all_measured_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"ablation_runs_completed={len(records)}")


if __name__ == "__main__":
    main()
