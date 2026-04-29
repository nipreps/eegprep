"""Sensor/source feature extraction translated from MATLAB analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mne

FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (2.0, 4.0),
    "theta": (5.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta": (15.0, 29.0),
    "gamma1": (30.0, 59.0),
    "gamma2": (60.0, 90.0),
}


@dataclass(slots=True)
class FeatureResult:
    sensor_absolute_bandpower: dict[str, list[float]]
    sensor_relative_bandpower: dict[str, list[float]]


def _bandpower_from_psd(freqs: np.ndarray, psd: np.ndarray, bands: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        out[name] = np.trapz(psd[:, mask], freqs[mask], axis=1)
    return out


def run_features(raw: mne.io.BaseRaw, win_length: float = 4.0, win_overlap_percent: float = 50.0) -> FeatureResult:
    sfreq = raw.info["sfreq"]
    n_per_seg = int(round(win_length * sfreq))
    n_overlap = int(round(n_per_seg * (win_overlap_percent / 100.0)))
    psd = raw.compute_psd(method="welch", n_per_seg=n_per_seg, n_overlap=n_overlap, picks="eeg")
    freqs = psd.freqs
    data = psd.get_data()  # channels x freqs

    abs_bp = _bandpower_from_psd(freqs, data, FREQ_BANDS)
    total = np.sum(np.vstack(list(abs_bp.values())), axis=0)
    rel_bp = {k: np.divide(v, total, out=np.zeros_like(v), where=total > 0) for k, v in abs_bp.items()}

    return FeatureResult(
        sensor_absolute_bandpower={k: [float(x) for x in v] for k, v in abs_bp.items()},
        sensor_relative_bandpower={k: [float(x) for x in v] for k, v in rel_bp.items()},
    )
