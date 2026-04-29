import pytest

mne = pytest.importorskip("mne")
import numpy as np

from eegprep.workflows.eeg.features import FREQ_BANDS, run_features


def test_feature_bandpower_outputs_all_bands():
    sfreq = 200.0
    times = np.arange(0, 20, 1 / sfreq)
    sig = np.sin(2 * np.pi * 10 * times)
    data = np.vstack([1e-6 * sig, 1e-6 * sig])
    info = mne.create_info(["C1", "C2"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    feat = run_features(raw)

    assert set(feat.sensor_absolute_bandpower.keys()) == set(FREQ_BANDS.keys())
    assert set(feat.sensor_relative_bandpower.keys()) == set(FREQ_BANDS.keys())
    assert len(feat.sensor_absolute_bandpower["alpha"]) == 2
