"""Derivative output writer placeholders."""

import json
from pathlib import Path

from eegprep.config import RunConfig
from eegprep.workflows.eeg.features import FeatureResult
from eegprep.workflows.eeg.preprocess import PreprocResult
from eegprep.workflows.eeg.qc import QCResult


def write_subject_stub_outputs(
    cfg: RunConfig,
    subject_id: str,
    qc: QCResult,
    pre: PreprocResult,
    features: FeatureResult | None = None,
) -> None:
    subj_dir = Path(cfg.derivatives_root) / f"sub-{subject_id}" / "eeg"
    subj_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "Subject": subject_id,
        "BadChannelDetection": {
            "Method": "median_std_plus_minus_mad",
            "ZThreshold": 5,
            "NoisyChannels": qc.noisy_channels,
            "FlatChannels": qc.flat_channels,
            "BadChannels": qc.bad_channels,
        },
        "PSDPre": {
            "Method": "welch",
            "WindowLength": 4.0,
            "WindowOverlapPercent": 50,
            "PeakThreshold": 1e-13,
            "PeakFrequencies": qc.peak_frequencies_hz,
            "PeakAmplitudes": qc.peak_amplitudes,
        },
        "Preprocessing": {
            "NotchFrequencies": pre.notch_hz,
            "HighPassHz": pre.high_pass_hz,
            "LowPassHz": pre.low_pass_hz,
            "Reference": pre.reference,
            "NumBlinks": pre.num_blinks,
            "NumCardiac": pre.num_cardiac,
            "BadSegments": {
                "Method": "peak_to_peak",
                "WindowLength": 1.0,
                "ThresholdMicrovolts": 150.0,
                "Count": pre.num_bad_segments,
                "OnsetsSec": pre.bad_segment_onsets_sec,
            },
        },
        "PSDPost": {
            "PeakFrequencies": pre.post_qc.peak_frequencies_hz,
            "PeakAmplitudes": pre.post_qc.peak_amplitudes,
        },
        "AvgPerChannelPre": qc.avg_per_channel,
        "StdPerChannelPre": qc.std_per_channel,
        "AvgPerChannelPost": pre.avg_per_channel_post,
        "StdPerChannelPost": pre.std_per_channel_post,
    }
    if features is not None:
        metrics["Features"] = {
            "FrequencyBands": {
                "delta": [2, 4],
                "theta": [5, 7],
                "alpha": [8, 12],
                "beta": [15, 29],
                "gamma1": [30, 59],
                "gamma2": [60, 90],
            },
            "SensorAbsoluteBandpower": features.sensor_absolute_bandpower,
            "SensorRelativeBandpower": features.sensor_relative_bandpower,
        }

    out_file = subj_dir / f"sub-{subject_id}_desc-qc_metrics.json"
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
