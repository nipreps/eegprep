"""Single-subject workflow composition."""

from dataclasses import dataclass
from uuid import uuid4

from mne_bids import read_raw_bids

from eegprep.config import RunConfig
from eegprep.cli.workflow import collect_eeg_data, write_run_config
from eegprep.workflows.eeg.features import run_features
from eegprep.workflows.eeg.outputs import write_subject_stub_outputs
from eegprep.workflows.eeg.preprocess import run_preprocess
from eegprep.workflows.eeg.qc import run_qc


@dataclass
class SingleSubjectWorkflow:
    cfg: RunConfig
    subject_id: str

    def run(self) -> None:
        runs = collect_eeg_data(self.cfg, self.subject_id)
        if not runs:
            raise RuntimeError(f"No EEG runs found for sub-{self.subject_id}")

        for bids_path in runs:
            run_uuid = str(uuid4())
            write_run_config(self.cfg, self.subject_id, run_uuid)
            raw = read_raw_bids(bids_path=bids_path, verbose=False)
            qc_result = run_qc(raw)
            preproc_result = run_preprocess(raw, self.cfg, qc_result)
            feature_result = run_features(raw) if self.cfg.level == "features" else None
            write_subject_stub_outputs(self.cfg, self.subject_id, qc_result, preproc_result, feature_result)


def init_single_subject_wf(cfg: RunConfig, subject_id: str) -> SingleSubjectWorkflow:
    return SingleSubjectWorkflow(cfg=cfg, subject_id=subject_id)
