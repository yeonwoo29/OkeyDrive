# Repository Instructions

- Write all first-party code, comments, docstrings, identifiers, filenames, configuration descriptions, logs, tests, and documentation in English.
- Never add personal names, affiliations, account names, email addresses, hostnames, SSH details, credentials, user-home paths, or private absolute paths.
- Use project-relative paths and the `DATA_ROOT`, `CHECKPOINT_ROOT`, `OUTPUT_ROOT`, and `CLIP_CACHE_ROOT` environment variables. Do not persist resolved environment paths in configs, logs, checkpoints, metadata, or release archives.
- Keep experiment tracking local and offline. Do not add automatic uploads.
- Preserve upstream license and attribution. Do not present external code as first-party work.
- Publish only evaluator-produced measurements, and never substitute research targets for results.
- Never use validation data to train, tune, or construct trajectory anchors.
- In inference, never use ground-truth boxes, part visibility, keypoints, future trajectories, or collision labels.
- Run the privacy checker before release export. A release must fail when the checker reports a finding.
- Do not commit, push, publish, or upload without an explicit user request.
