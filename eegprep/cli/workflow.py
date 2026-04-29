"""Workflow helpers for CLI execution."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mne_bids import BIDSPath, get_entity_vals

from eegprep.config import RunConfig


def collect_eeg_data(cfg: RunConfig, subject: str) -> list[BIDSPath]:
    """Collect EEG runs for a subject using EEG-aware BIDS entities."""
    sessions = [cfg.session_label] if cfg.session_label else [None]
    tasks = [cfg.task] if cfg.task else get_entity_vals(cfg.bids_root, "task") or [None]
    runs = get_entity_vals(cfg.bids_root, "run") or [None]

    found: list[BIDSPath] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for session in sessions:
        for task in tasks:
            for run in runs:
                key = (session, task, run)
                if key in seen:
                    continue
                seen.add(key)
                bp = BIDSPath(
                    root=cfg.bids_root,
                    subject=subject,
                    session=session,
                    task=task,
                    run=run,
                    datatype="eeg",
                    suffix="eeg",
                    extension=".edf",
                )
                if bp.fpath.exists():
                    found.append(bp)
    return found


def write_run_config(cfg: RunConfig, subject: str, run_uuid: str) -> Path:
    """Write serialized run configuration under subject log directory."""
    log_dir = cfg.derivatives_root / f"sub-{subject}" / "log" / run_uuid
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / "eegprep.toml"

    payload = asdict(cfg)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        if isinstance(value, list):
            payload[key] = [str(v) for v in value]

    lines = ["[run]", *[f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}" for k, v in payload.items()]]
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_file
