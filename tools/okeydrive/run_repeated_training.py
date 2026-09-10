"""Launch deterministic repeated-seed training runs with isolated outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from okeydrive.runtime_privacy import install_redacting_excepthook

install_redacting_excepthook()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "repeated")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for seed in args.seeds:
        work_dir = args.output_root / args.config.stem / f"seed{seed}"
        command = [
            sys.executable,
            "tools/train.py",
            str(args.config),
            "--work-dir",
            str(work_dir),
            "--seed",
            str(seed),
            "--deterministic",
        ]
        latest = work_dir / "latest.pth"
        if args.resume and latest.is_file():
            command.extend(["--resume-from", str(latest)])
        subprocess.run(command, check=True)
    print(f"training_runs_completed={len(args.seeds)}")


if __name__ == "__main__":
    main()
