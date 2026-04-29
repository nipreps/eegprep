"""Derivative output writer placeholders."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

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
            "NotchPerformanceDb": pre.notch_performance_db,
            "HighPassHz": pre.high_pass_hz,
            "LowPassHz": pre.low_pass_hz,
            "Reference": pre.reference,
            "ReferenceChannels": pre.reference_channels,
            "NumBlinks": pre.num_blinks,
            "NumCardiac": pre.num_cardiac,
            "NumCardiacRemovedNearBlink": pre.num_cardiac_removed_near_blink,
            "NumCardiacAfterBlinkCensor": pre.num_cardiac_after_blink_censor,
            "BlinkOnsetsSec": pre.blink_onsets_sec,
            "CardiacOnsetsSec": pre.cardiac_onsets_sec,
            "CardiacOnsetsAfterBlinkCensorSec": pre.cardiac_onsets_after_blink_censor_sec,
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
        "NoiseCovariancePerCensor": {
            "Estimator": "numpy_cov_trace",
            "TracePerCensor": pre.noise_cov_trace_per_censor,
            "Count": len(pre.noise_cov_trace_per_censor),
        },
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

    preproc_raw_file = subj_dir / f"sub-{subject_id}_desc-preproc_eeg.fif"
    pre.cleaned_raw.save(preproc_raw_file, overwrite=True)

    post_psd = pre.cleaned_raw.compute_psd(method="welch", picks="eeg")
    psd_fig = post_psd.plot(show=False)
    psd_fig.savefig(subj_dir / f"sub-{subject_id}_desc-postproc_psd.png", dpi=150, bbox_inches="tight")

    report = mne.Report(title=f"EEGPrep subject {subject_id}")
    report.add_raw(pre.cleaned_raw, title="Preprocessed raw", psd=False)
    report.add_figure(psd_fig, title="Post-processing PSD")
    if pre.noise_cov_trace_per_censor:
        fig_cov, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pre.noise_cov_trace_per_censor, color="#4C956C")
        ax.set_xlabel("Censor index")
        ax.set_ylabel("Covariance trace")
        ax.set_title("Noise covariance estimate per censor")
        fig_cov.tight_layout()
        report.add_figure(fig_cov, title="Noise covariance per censor")
        plt.close(fig_cov)
    if pre.notch_performance_db:
        freqs = list(pre.notch_performance_db.keys())
        atten = [pre.notch_performance_db[f] for f in freqs]
        fig_notch, ax = plt.subplots(figsize=(8, 4))
        ax.bar(freqs, atten, color="#3A7CA5")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Attenuation (dB)")
        ax.set_title("Notch filter performance")
        fig_notch.tight_layout()
        report.add_figure(fig_notch, title="Notch filter performance")
        plt.close(fig_notch)

    eeg_for_plot = pre.cleaned_raw.copy().pick("eeg")
    has_positions = any(np.isfinite(ch["loc"][:3]).any() and not np.allclose(ch["loc"][:3], 0.0) for ch in eeg_for_plot.info["chs"])
    if has_positions:
        fig_ref = eeg_for_plot.plot_sensors(show=False)
        ref_title = f"Reference montage ({', '.join(pre.reference_channels)})"
        report.add_figure(fig_ref, title=ref_title)
        plt.close(fig_ref)

    if pre.reference == "average":
        ref_weights = np.ones(len(pre.cleaned_raw.ch_names), dtype=float) / max(1, len(pre.cleaned_raw.ch_names))
        fig_avg_ref, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ref_weights)
        ax.set_ylim(0, max(ref_weights) * 1.2)
        ax.set_title("Average reference weights")
        ax.set_xlabel("Channel index")
        ax.set_ylabel("Weight")
        fig_avg_ref.tight_layout()
        report.add_figure(fig_avg_ref, title="Reference montage weights")
        plt.close(fig_avg_ref)
    report.save(subj_dir / f"sub-{subject_id}_desc-qc_report.html", overwrite=True, open_browser=False)
    plt.close(psd_fig)
