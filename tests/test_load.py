import json
from pathlib import Path

import csv
import pytest

mne = pytest.importorskip("mne")
import numpy as np

from eegprep.workflows.eeg.load import load_edf_with_bids_metadata


def _write_sidecars(tmp_path: Path, sfreq: float = 200.0) -> tuple[Path, Path]:
    json_path = tmp_path / "sub-001_ses-t1_task-rest_eeg.json"
    channels_path = tmp_path / "sub-001_ses-t1_task-rest_channels.tsv"

    json_path.write_text(
        json.dumps({"SamplingFrequency": sfreq, "EEGReference": "average"}),
        encoding="utf-8",
    )
    with open(channels_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "type"], delimiter="\t")
        writer.writeheader()
        writer.writerows([
            {"name": "Fz", "type": "EEG"},
            {"name": "Cz", "type": "EOG"},
            {"name": "Oz", "type": "EEG"},
        ])
    return json_path, channels_path


def test_load_edf_applies_bids_names_and_types(tmp_path, monkeypatch):
    info = mne.create_info(["A", "B", "C"], sfreq=200.0, ch_types=["eeg", "eeg", "eeg"])
    raw_in = mne.io.RawArray(np.zeros((3, 200)), info, verbose=False)

    json_path, channels_path = _write_sidecars(tmp_path)

    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: raw_in.copy())

    raw, meta, channels = load_edf_with_bids_metadata(
        tmp_path / "dummy.edf",
        json_path,
        channels_path,
    )

    assert raw.ch_names == ["Fz", "Cz", "Oz"]
    assert raw.get_channel_types() == ["eeg", "eog", "eeg"]
    assert meta["EEGReference"] == "average"
    assert [row["name"] for row in channels] == ["Fz", "Cz", "Oz"]


def test_load_edf_raises_on_channel_count_mismatch(tmp_path, monkeypatch):
    info = mne.create_info(["A", "B"], sfreq=200.0, ch_types=["eeg", "eeg"])
    raw_in = mne.io.RawArray(np.zeros((2, 200)), info, verbose=False)
    json_path, channels_path = _write_sidecars(tmp_path)

    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *args, **kwargs: raw_in.copy())

    with pytest.raises(ValueError, match="Channel count mismatch"):
        load_edf_with_bids_metadata(tmp_path / "dummy.edf", json_path, channels_path)
