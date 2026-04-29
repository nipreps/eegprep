"""BIDS-aware EDF loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import csv


TYPE_MAP = {
    "eeg": "eeg",
    "eog": "eog",
    "ecg": "ecg",
    "emg": "emg",
}


def load_edf_with_bids_metadata(
    edf_path: Path,
    json_path: Path,
    channels_path: Path,
    *,
    montage_name: str = "standard_1020",
) -> tuple[mne.io.BaseRaw, dict, list[dict[str, str]]]:
    """Load EDF and harmonize channel metadata with BIDS sidecars."""
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(channels_path, "r", encoding="utf-8", newline="") as f:
        ch_tsv = list(csv.DictReader(f, delimiter="\t"))
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=True)

    bids_names = [row["name"] for row in ch_tsv]
    if len(raw.ch_names) != len(bids_names):
        raise ValueError(
            f"Channel count mismatch: EDF={len(raw.ch_names)} vs TSV={len(bids_names)}"
        )

    rename_map = {old: new for old, new in zip(raw.ch_names, bids_names)}
    raw.rename_channels(rename_map)

    type_map: dict[str, str] = {}
    for row in ch_tsv:
        channel_name = row["name"]
        channel_type = str(row["type"]).strip().lower()
        if channel_type in TYPE_MAP:
            type_map[channel_name] = TYPE_MAP[channel_type]
    if type_map:
        raw.set_channel_types(type_map)

    sfreq_raw = float(raw.info["sfreq"])
    sfreq_json = float(meta["SamplingFrequency"])
    if abs(sfreq_raw - sfreq_json) > 1e-6:
        print(f"WARNING: sfreq mismatch raw={sfreq_raw}, json={sfreq_json}")

    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing="warn")

    print("EEGReference from JSON:", meta.get("EEGReference", "n/a"))
    return raw, meta, ch_tsv
