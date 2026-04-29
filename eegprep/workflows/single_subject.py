"""Single-subject workflow composition."""

from dataclasses import dataclass

from mne_bids import BIDSPath, read_raw_bids

from eegprep.config import RunConfig
from eegprep.workflows.eeg.features import run_features
from eegprep.workflows.eeg.outputs import write_subject_stub_outputs
from eegprep.workflows.eeg.preprocess import run_preprocess
from eegprep.workflows.eeg.qc import run_qc


@dataclass
class SingleSubjectWorkflow:
    cfg: RunConfig
    subject_id: str

    def run(self) -> None:
        bids_path = BIDSPath(root=self.cfg.bids_root, subject=self.subject_id, task=self.cfg.task, session=self.cfg.session_label, datatype="eeg")
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        qc_result = run_qc(raw)
        preproc_result = run_preprocess(raw, self.cfg, qc_result)
        feature_result = run_features(raw) if self.cfg.level == "features" else None
        write_subject_stub_outputs(self.cfg, self.subject_id, qc_result, preproc_result, feature_result)


def init_single_subject_wf(cfg: RunConfig, subject_id: str) -> SingleSubjectWorkflow:
    return SingleSubjectWorkflow(cfg=cfg, subject_id=subject_id)
