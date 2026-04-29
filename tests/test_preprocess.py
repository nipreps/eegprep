import pytest

mne = pytest.importorskip("mne")
import numpy as np

from eegprep.config import RunConfig
from eegprep.workflows.eeg.preprocess import run_preprocess
from eegprep.workflows.eeg.qc import QCParams, run_qc


def test_preprocess_runs_and_emits_metrics():
    sfreq = 200.0
    times = np.arange(0, 20, 1 / sfreq)
    sin60 = np.sin(2 * np.pi * 60 * times)
    data = np.vstack([1e-6 * sin60, 1e-6 * sin60, 2e-6 * sin60])
    info = mne.create_info(["C1", "C2", "C3"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    cfg = RunConfig(
        bids_root=".",
        derivatives_root=".",
        analysis_level="participant",
        participant_labels=["01"],
        notch="auto",
    )
    qc = run_qc(raw, QCParams(peak_threshold=1e-16))
    result = run_preprocess(raw, cfg, qc)

    assert result.reference == "average"
    assert result.high_pass_hz == 0.3
    assert isinstance(result.notch_hz, list)
    assert len(result.avg_per_channel_post) == 3
