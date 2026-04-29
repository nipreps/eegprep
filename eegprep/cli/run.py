"""Entrypoint for EEGPrep CLI."""

from eegprep.cli.parser import build_parser
from eegprep.config import RunConfig
from eegprep.workflows.base import init_eegprep_wf


def main() -> None:
    args = build_parser().parse_args()
    cfg = RunConfig(
        bids_root=args.bids_root,
        derivatives_root=args.derivatives_root,
        analysis_level=args.analysis_level,
        participant_labels=args.participant_label,
        task=args.task,
        session_label=args.session_label,
        work_dir=args.work_dir,
        nprocs=args.nprocs,
        omp_nthreads=args.omp_nthreads,
        level=args.level,
        eeg_reference=args.eeg_reference,
        montage=args.montage,
        high_pass=args.high_pass,
        low_pass=args.low_pass,
        notch=args.notch,
        eog_channel=args.eog_channel,
        ecg_channel=args.ecg_channel,
        source_recon=args.source_recon,
    )
    wf = init_eegprep_wf(cfg)
    wf.run()


if __name__ == "__main__":
    main()
