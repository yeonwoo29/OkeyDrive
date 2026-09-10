"""Create a privacy-checked, reproducible OkeyDrive source release archive."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from okeydrive.runtime_privacy import install_redacting_excepthook

install_redacting_excepthook()

import torch

from okeydrive.checkpoint import state_dict_sha256
from tools.okeydrive.privacy import scan_archive, scan_tree, summarized_report


EXCLUDED_NAMES = {
    ".git", ".env", ".idea", ".vscode", "__pycache__", "data", "work_dirs", "outputs", "dist", ".dist_test"
}
EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".log", ".tmp", ".pth", ".pt", ".ipynb"
}


def include_source(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part.lower() in EXCLUDED_NAMES for part in relative.parts):
        return False
    return path.suffix.lower() not in EXCLUDED_SUFFIXES


def export_checkpoint(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise ValueError("checkpoint does not contain a recognizable state_dict")
    if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state_dict.items()):
        raise ValueError("state_dict contains unsupported entries")
    payload = {
        "state_dict": {name: value.detach().cpu() for name, value in state_dict.items()},
        "metadata": {
            "format": "okeydrive_release_state_dict_v1",
            "sha256": state_dict_sha256(state_dict),
        },
    }
    torch.save(payload, destination)


def write_reproducible_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2025, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    source = args.source
    with tempfile.TemporaryDirectory(prefix="okeydrive_release_") as temporary:
        staging = Path(temporary) / "OkeyDrive"
        staging.mkdir()
        for path in sorted(source.rglob("*")):
            if not path.is_file() or not include_source(path, source):
                continue
            destination = staging / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        if args.checkpoint:
            checkpoint_destination = staging / "checkpoints" / "okeydrive_state_dict.pt"
            checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
            export_checkpoint(args.checkpoint, checkpoint_destination)
        findings = scan_tree(staging)
        if findings:
            print(json.dumps(summarized_report(findings), indent=2))
            raise SystemExit(1)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_reproducible_zip(staging, args.output)
    archive_findings = scan_archive(args.output)
    if archive_findings:
        args.output.unlink(missing_ok=True)
        print(json.dumps(summarized_report(archive_findings), indent=2))
        raise SystemExit(1)
    print("release_export_passed=True")


if __name__ == "__main__":
    main()
