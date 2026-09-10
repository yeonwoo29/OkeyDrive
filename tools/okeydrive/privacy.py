"""Privacy checks for first-party sources, metadata, and release archives."""

from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh", ".example"
}
TEXT_FILENAMES = {".gitignore"}
EXCLUDED_PARTS = {
    ".git", ".env", ".idea", ".vscode", "__pycache__", "data", "work_dirs", "outputs"
}


@dataclass(frozen=True)
class Finding:
    file: str
    rule: str


RULES = {
    "hangul": re.compile(r"[\uac00-\ud7a3]"),
    "windows_user_path": re.compile(r"(?i)[a-z]:[\\/]+users[\\/]+[^\\/\s]+"),
    "posix_home_path": re.compile(r"/(?:home|users)/[^/\s]+"),
    "ssh_private_key": re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
    "credential_assignment": re.compile(
        r"(?i)(?:access[_-]?token|api[_-]?key|client[_-]?secret|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
    "email_address": re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
    "host_identity_collection": re.compile(
        "(?i)(?:" + "|".join(
            (
                r"platform\." + "node",
                r"socket\." + "gethostname",
                r"os\." + "getlogin",
                r"getpass\." + "getuser",
                "COMPUTER" + "NAME",
            )
        ) + ")"
    ),
}

UPSTREAM_ATTRIBUTION_EMAIL_FILES = {"README.md", "LICENSE"}


def _eligible(path: Path) -> bool:
    is_text = path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES
    return is_text and not any(part.lower() in EXCLUDED_PARTS for part in path.parts)


def iter_source_text(root: Path) -> Iterator[tuple[str, str]]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and _eligible(path):
            try:
                yield path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue


def scan_text_items(items: Iterable[tuple[str, str]]) -> list[Finding]:
    findings = []
    for name, content in items:
        for rule, pattern in RULES.items():
            if rule == "email_address" and name in UPSTREAM_ATTRIBUTION_EMAIL_FILES:
                continue
            if pattern.search(content):
                findings.append(Finding(name, rule))
    return findings


def scan_tree(root: Path) -> list[Finding]:
    return scan_text_items(iter_source_text(root))


def scan_archive(path: Path) -> list[Finding]:
    def items():
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if Path(name).suffix.lower() not in TEXT_SUFFIXES and Path(name).name not in TEXT_FILENAMES:
                    continue
                try:
                    yield name, archive.read(name).decode("utf-8")
                except UnicodeDecodeError:
                    continue

    return scan_text_items(items())


def summarized_report(findings: list[Finding]) -> dict:
    """Return rule/file locations without reproducing matched sensitive text."""

    return {"passed": not findings, "finding_count": len(findings), "findings": [asdict(item) for item in findings]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    findings = scan_archive(args.path) if args.path.suffix.lower() == ".zip" else scan_tree(args.path)
    report = summarized_report(findings)
    print(f"privacy_check_passed={report['passed']} finding_count={report['finding_count']}")
    for finding in findings:
        print(f"rule={finding.rule} file={finding.file}")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
