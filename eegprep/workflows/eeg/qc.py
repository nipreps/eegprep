"""QC workflow implementation translated from the MATLAB QC logic."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np


@dataclass(slots=True)
class QCParams:
    epoch_length: float = 4.0
    ignore_short_epochs: bool = True
    z_threshold: float = 5.0
    psd_window_sec: float = 4.0
    psd_overlap_percent: float = 50.0
    peak_threshold: float = 1e-13


@dataclass(slots=True)
class QCResult:
    bad_channels: list[str]
    noisy_channels: list[str]
    flat_channels: list[str]
    notch_candidates_hz: list[float]
    peak_frequencies_hz: list[float]
    peak_amplitudes: list[float]
    avg_per_channel: list[float]
    std_per_channel: list[float]


def _find_local_peaks(values: np.ndarray, threshold: float) -> np.ndarray:
    """Simple local-max peak finder with minimum absolute threshold."""
    if values.size < 3:
        return np.array([], dtype=int)
    left = values[1:-1] > values[:-2]
    right = values[1:-1] > values[2:]
    above = values[1:-1] >= threshold
    return np.where(left & right & above)[0] + 1


def run_qc(raw: mne.io.BaseRaw, params: QCParams | None = None) -> QCResult:
    """Run MATLAB-equivalent QC on a raw EEG object.

    Steps mirrored from `scripts/1_QualityControl.m`:
    - fixed-length epoching (4s)
    - per-channel mean/std on concatenated epoch data
    - bad channel detection using median(std) ± z * MAD(std)
    - Welch PSD + peak detection
    - notch candidate frequencies = peaks above 20 Hz
    """
    params = params or QCParams()

    # 1) epoch into fixed windows
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=params.epoch_length,
        preload=True,
        reject_by_annotation=params.ignore_short_epochs,
    )
    data = epochs.get_data(copy=True)  # shape: n_epochs, n_channels, n_times
    if data.size == 0:
        raise RuntimeError("No epoch data available for QC.")

    # 2) concatenate epochs over time and compute per-channel stats
    concatenated = np.transpose(data, (1, 0, 2)).reshape(data.shape[1], -1)
    avg = concatenated.mean(axis=1)
    std = concatenated.std(axis=1, ddof=0)

    # 3) MAD-based bad channel detection
    median_std = float(np.median(std))
    mad_std = float(np.median(np.abs(std - median_std)))
    upper = median_std + (params.z_threshold * mad_std)
    lower = median_std - (params.z_threshold * mad_std)

    ch_names = raw.ch_names
    noisy_idx = np.where(std > upper)[0]
    flat_idx = np.where(std < lower)[0]
    bad_idx = np.unique(np.concatenate([noisy_idx, flat_idx]))

    noisy_channels = [ch_names[i] for i in noisy_idx]
    flat_channels = [ch_names[i] for i in flat_idx]
    bad_channels = [ch_names[i] for i in bad_idx]

    # 4) Welch PSD + peaks (mean across channels)
    sfreq = raw.info["sfreq"]
    n_per_seg = max(8, int(round(params.psd_window_sec * sfreq)))
    n_overlap = int(round(n_per_seg * (params.psd_overlap_percent / 100.0)))
    psd = raw.compute_psd(method="welch", n_per_seg=n_per_seg, n_overlap=n_overlap)
    freqs = psd.freqs
    psd_data = psd.get_data()
    mean_psd = psd_data.mean(axis=0)

    peak_indices = _find_local_peaks(mean_psd, params.peak_threshold)
    peak_freqs = freqs[peak_indices]
    peak_amps = mean_psd[peak_indices]
    notch_candidates = [float(f) for f in peak_freqs if f > 20.0]

    return QCResult(
        bad_channels=bad_channels,
        noisy_channels=noisy_channels,
        flat_channels=flat_channels,
        notch_candidates_hz=notch_candidates,
        peak_frequencies_hz=[float(x) for x in peak_freqs],
        peak_amplitudes=[float(x) for x in peak_amps],
        avg_per_channel=[float(x) for x in avg],
        std_per_channel=[float(x) for x in std],
    )
