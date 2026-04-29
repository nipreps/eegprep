"""Runtime configuration models for EEGPrep."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    bids_root: Path
    derivatives_root: Path
    analysis_level: str
    participant_labels: list[str]
    task: str | None = None
    session_label: str | None = None
    work_dir: Path | None = None
    nprocs: int = 1
    omp_nthreads: int = 1
    level: str = "minimal"
    eeg_reference: str = "average"
    montage: str = "auto"
    high_pass: float = 0.3
    low_pass: float | None = None
    notch: str = "auto"
    eog_channel: str | None = None
    ecg_channel: str | None = None
    source_recon: str = "none"
