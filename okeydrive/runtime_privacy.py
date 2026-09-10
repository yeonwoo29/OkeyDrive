"""Redact user-specific filesystem roots from persisted runtime text."""

from __future__ import annotations

import re
import sys
import traceback


PATH_PATTERNS = (
    re.compile(r"(?i)[a-z]:[\\/](?:[^\s'\";,\]\)]+[\\/]?)+"),
    re.compile(r"/(?:home|users)/[^/\s'\";,\]\)]+(?:/[^\s'\";,\]\)]*)?"),
)


def redact_local_paths(text: object) -> str:
    """Replace absolute local paths without recording their original values."""

    sanitized = str(text)
    for pattern in PATH_PATTERNS:
        sanitized = pattern.sub("<LOCAL_PATH>", sanitized)
    return sanitized


def install_redacting_excepthook() -> None:
    """Ensure uncaught first-party CLI tracebacks do not reveal local roots."""

    def redacted_exception(exception_type, exception, trace) -> None:
        rendered = "".join(traceback.format_exception(exception_type, exception, trace))
        sys.stderr.write(redact_local_paths(rendered))

    sys.excepthook = redacted_exception
