"""CLI parser for EEGPrep BIDS App interface."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eegprep")
    parser.add_argument("bids_root", type=Path)
    parser.add_argument("derivatives_root", type=Path)
    parser.add_argument("analysis_level", choices=["participant"])
    parser.add_argument("--participant-label", nargs="+", default=[])
    parser.add_argument("--task", default=None)
    parser.add_argument("--session-label", default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--omp-nthreads", type=int, default=1)
    parser.add_argument("--level", choices=["minimal", "full", "features"], default="minimal")
    parser.add_argument("--eeg-reference", default="average")
    parser.add_argument("--montage", default="auto")
    parser.add_argument("--high-pass", type=float, default=0.3)
    parser.add_argument("--low-pass", type=float, default=None)
    parser.add_argument("--notch", default="auto")
    parser.add_argument("--eog-channel", default=None)
    parser.add_argument("--ecg-channel", default=None)
    parser.add_argument("--source-recon", choices=["none", "template", "subject"], default="none")
    return parser
