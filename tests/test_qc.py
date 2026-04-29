import pytest

mne = pytest.importorskip("mne")
import numpy as np

from eegprep.workflows.eeg.qc import QCParams, run_qc


def test_qc_detects_noisy_channel_and_peaks():
    sfreq = 200.0
    times = np.arange(0, 20, 1 / sfreq)
    sin60 = np.sin(2 * np.pi * 60 * times)
    data = np.vstack([
        1e-6 * sin60,
        1e-6 * sin60,
        8e-6 * sin60,
        1e-6 * sin60,
    ])
    info = mne.create_info(["C1", "C2", "C3", "C4"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    result = run_qc(raw, QCParams(peak_threshold=1e-16))

    assert "C3" in result.bad_channels
    assert any(abs(freq - 60.0) < 1.0 for freq in result.peak_frequencies_hz)
    assert any(freq > 20 for freq in result.notch_candidates_hz)


def test_qc_clamps_psd_overlap_to_valid_range():
    sfreq = 128.0
    times = np.arange(0, 10, 1 / sfreq)
    data = np.vstack([
        1e-6 * np.sin(2 * np.pi * 10 * times),
        1e-6 * np.sin(2 * np.pi * 12 * times),
    ])
    info = mne.create_info(["C1", "C2"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    result = run_qc(raw, QCParams(psd_overlap_percent=100.0, peak_threshold=1e-16))

    assert isinstance(result.peak_frequencies_hz, list)
