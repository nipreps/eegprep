"""Helpers for participant selection in the CLI."""

from pathlib import Path


def discover_participants(bids_root: Path) -> list[str]:
    """Discover participant labels from sub-* directories in ``bids_root``."""
    labels: list[str] = []
    for child in sorted(bids_root.iterdir()):
        if child.is_dir() and child.name.startswith("sub-"):
            label = child.name.removeprefix("sub-")
            if label:
                labels.append(label)
    return labels
