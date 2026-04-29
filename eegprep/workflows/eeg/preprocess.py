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
    remove_cardiac_near_blink_sec: float = 0.25


@dataclass(slots=True)
class PreprocResult:
    reference: str
    high_pass_hz: float
    low_pass_hz: float | None
    notch_hz: list[float]
    num_blinks: int
    num_cardiac: int
    num_cardiac_removed_near_blink: int
    num_cardiac_after_blink_censor: int
    num_bad_segments: int
    bad_segment_onsets_sec: list[float]
    blink_onsets_sec: list[float]
    cardiac_onsets_sec: list[float]
    cardiac_onsets_after_blink_censor_sec: list[float]
    avg_per_channel_post: list[float]
    std_per_channel_post: list[float]
    noise_cov_trace_per_censor: list[float]
    post_qc: QCResult
    notch_performance_db: dict[str, float]
    reference_channels: list[str]
    cleaned_raw: mne.io.BaseRaw


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
    pre_psd = pre.compute_psd(method="welch", picks="eeg")
    pre_freqs = pre_psd.freqs
    pre_mean_psd = pre_psd.get_data().mean(axis=0)
    pre.notch_filter(freqs=notch_hz, picks="eeg")
    h_freq = cfg.low_pass if (cfg.low_pass and cfg.low_pass > 0) else None
    pre.filter(l_freq=cfg.high_pass, h_freq=h_freq, picks="eeg")

    # reference behavior modeled after MATLAB: empty/average => average
    ref = cfg.eeg_reference or "average"
    reference_channels: list[str]
    if ref.lower() == "average":
        pre.set_eeg_reference(ref_channels="average")
        reference_channels = ["average"]
    else:
        pre.set_eeg_reference(ref_channels=[ref])
        reference_channels = [ref]

    num_blinks = 0
    num_cardiac = 0
    num_cardiac_removed_near_blink = 0
    blink_onsets_sec: list[float] = []
    cardiac_onsets_sec: list[float] = []
    cardiac_onsets_after_blink_censor_sec: list[float] = []

    # artifact events and SSP
    if params.eog_channel:
        eog_events = mne.preprocessing.find_eog_events(pre, ch_name=params.eog_channel, verbose=False)
        num_blinks = int(eog_events.shape[0])
        blink_onsets_sec = [float(evt[0] / pre.info["sfreq"]) for evt in eog_events]
        eog_projs, _ = mne.preprocessing.compute_proj_eog(pre, ch_name=params.eog_channel, n_eeg=params.eog_proj_count, verbose=False)
        pre.add_proj(eog_projs)

    if params.ecg_channel:
        ecg_events, _, _ = mne.preprocessing.find_ecg_events(pre, ch_name=params.ecg_channel, verbose=False)
        num_cardiac = int(ecg_events.shape[0])
        cardiac_onsets_sec = [float(evt[0] / pre.info["sfreq"]) for evt in ecg_events]
        cardiac_onsets_after_blink_censor_sec = list(cardiac_onsets_sec)
        if blink_onsets_sec:
            keep_cardiac: list[float] = []
            for c_t in cardiac_onsets_sec:
                near_blink = any(abs(c_t - b_t) <= params.remove_cardiac_near_blink_sec for b_t in blink_onsets_sec)
                if near_blink:
                    num_cardiac_removed_near_blink += 1
                else:
                    keep_cardiac.append(c_t)
            cardiac_onsets_after_blink_censor_sec = keep_cardiac
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
    noise_cov_trace_per_censor: list[float] = []
    for epoch in ep_data:
        cov = np.cov(epoch, bias=False)
        noise_cov_trace_per_censor.append(float(np.trace(cov)))

    # post QC PSD/peaks
    post_qc = run_qc(pre, QCParams())
    post_psd = pre.compute_psd(method="welch", picks="eeg")
    post_freqs = post_psd.freqs
    post_mean_psd = post_psd.get_data().mean(axis=0)
    notch_perf: dict[str, float] = {}
    for freq in notch_hz:
        pre_idx = int(np.argmin(np.abs(pre_freqs - freq)))
        post_idx = int(np.argmin(np.abs(post_freqs - freq)))
        before = max(float(pre_mean_psd[pre_idx]), np.finfo(float).eps)
        after = max(float(post_mean_psd[post_idx]), np.finfo(float).eps)
        notch_perf[f"{float(freq):g}Hz"] = float(10.0 * np.log10(before / after))

    return PreprocResult(
        reference=ref,
        high_pass_hz=cfg.high_pass,
        low_pass_hz=h_freq,
        notch_hz=notch_hz,
        num_blinks=num_blinks,
        num_cardiac=num_cardiac,
        num_cardiac_removed_near_blink=num_cardiac_removed_near_blink,
        num_cardiac_after_blink_censor=len(cardiac_onsets_after_blink_censor_sec),
        num_bad_segments=len(onsets),
        bad_segment_onsets_sec=[float(x) for x in onsets],
        blink_onsets_sec=blink_onsets_sec,
        cardiac_onsets_sec=cardiac_onsets_sec,
        cardiac_onsets_after_blink_censor_sec=cardiac_onsets_after_blink_censor_sec,
        avg_per_channel_post=[float(x) for x in avg_post],
        std_per_channel_post=[float(x) for x in std_post],
        noise_cov_trace_per_censor=noise_cov_trace_per_censor,
        post_qc=post_qc,
        notch_performance_db=notch_perf,
        reference_channels=reference_channels,
        cleaned_raw=pre,
    )
