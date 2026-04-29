"""Preprocessing workflow logic translated from MATLAB to MNE."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np

from eegprep.config import RunConfig
from eegprep.workflows.eeg.qc import QCResult, QCParams, run_qc


@dataclass(slots=True)
class PreprocParams:
    eog_channel: str | None = None
    ecg_channel: str | None = None
    eog_proj_count: int = 1
    ecg_proj_count: int = 1
    bad_segment_window_sec: float = 1.0
    bad_segment_threshold_uv: float = 150.0
    psd_window_sec: float = 4.0
    psd_overlap_percent: float = 50.0


@dataclass(slots=True)
class PreprocResult:
    reference: str
    high_pass_hz: float
    low_pass_hz: float | None
    notch_hz: list[float]
    num_blinks: int
    num_cardiac: int
    num_bad_segments: int
    bad_segment_onsets_sec: list[float]
    avg_per_channel_post: list[float]
    std_per_channel_post: list[float]
    post_qc: QCResult


def _resolve_notch(cfg: RunConfig, qc_result: QCResult) -> list[float]:
    if cfg.notch == "auto":
        return qc_result.notch_candidates_hz or [50.0]
    return [float(x) for x in str(cfg.notch).split()]


def _annotate_bad_segments(raw: mne.io.BaseRaw, window_sec: float, threshold_uv: float) -> tuple[mne.Annotations, list[float]]:
    data = raw.get_data(picks="eeg") * 1e6  # convert V -> uV
    sfreq = raw.info["sfreq"]
    n_samp = max(1, int(round(window_sec * sfreq)))
    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    for start in range(0, data.shape[1] - n_samp + 1, n_samp):
        window = data[:, start : start + n_samp]
        ptp = np.ptp(window, axis=1)
        if float(np.max(ptp)) >= threshold_uv:
            onsets.append(start / sfreq)
            durations.append(window_sec)
            descriptions.append("BAD_peak_to_peak")

    return mne.Annotations(onset=onsets, duration=durations, description=descriptions), onsets


def run_preprocess(raw: mne.io.BaseRaw, cfg: RunConfig, qc_result: QCResult, params: PreprocParams | None = None) -> PreprocResult:
    params = params or PreprocParams(eog_channel=cfg.eog_channel, ecg_channel=cfg.ecg_channel)
    pre = raw.copy().load_data()

    # mark bad channels from QC
    pre.info["bads"] = list(qc_result.bad_channels)

    # notch + high/low-pass
    notch_hz = _resolve_notch(cfg, qc_result)
    pre.notch_filter(freqs=notch_hz, picks="eeg")
    h_freq = cfg.low_pass if (cfg.low_pass and cfg.low_pass > 0) else None
    pre.filter(l_freq=cfg.high_pass, h_freq=h_freq, picks="eeg")

    # reference behavior modeled after MATLAB: empty/average => average
    ref = cfg.eeg_reference or "average"
    if ref.lower() == "average":
        pre.set_eeg_reference(ref_channels="average")
    else:
        pre.set_eeg_reference(ref_channels=[ref])

    num_blinks = 0
    num_cardiac = 0

    # artifact events and SSP
    if params.eog_channel:
        eog_events = mne.preprocessing.find_eog_events(pre, ch_name=params.eog_channel, verbose=False)
        num_blinks = int(eog_events.shape[0])
        eog_projs, _ = mne.preprocessing.compute_proj_eog(pre, ch_name=params.eog_channel, n_eeg=params.eog_proj_count, verbose=False)
        pre.add_proj(eog_projs)

    if params.ecg_channel:
        ecg_events, _, _ = mne.preprocessing.find_ecg_events(pre, ch_name=params.ecg_channel, verbose=False)
        num_cardiac = int(ecg_events.shape[0])
        ecg_projs, _ = mne.preprocessing.compute_proj_ecg(pre, ch_name=params.ecg_channel, n_eeg=params.ecg_proj_count, verbose=False)
        pre.add_proj(ecg_projs)

    if pre.info.get("projs"):
        pre.apply_proj()

    # bad segment detection (peak-to-peak)
    bad_ann, onsets = _annotate_bad_segments(pre, params.bad_segment_window_sec, params.bad_segment_threshold_uv)
    pre.set_annotations(pre.annotations + bad_ann)

    # post stats equivalent to MATLAB post-epoch concat
    epochs = mne.make_fixed_length_epochs(pre, duration=4.0, preload=True, reject_by_annotation=True)
    ep_data = epochs.get_data(copy=True)
    concat = np.transpose(ep_data, (1, 0, 2)).reshape(ep_data.shape[1], -1)
    avg_post = concat.mean(axis=1)
    std_post = concat.std(axis=1, ddof=0)

    # post QC PSD/peaks
    post_qc = run_qc(pre, QCParams())

    return PreprocResult(
        reference=ref,
        high_pass_hz=cfg.high_pass,
        low_pass_hz=h_freq,
        notch_hz=notch_hz,
        num_blinks=num_blinks,
        num_cardiac=num_cardiac,
        num_bad_segments=len(onsets),
        bad_segment_onsets_sec=[float(x) for x in onsets],
        avg_per_channel_post=[float(x) for x in avg_post],
        std_per_channel_post=[float(x) for x in std_post],
        post_qc=post_qc,
    )
