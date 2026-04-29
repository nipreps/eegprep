"""Top-level workflow factory."""

from eegprep.config import RunConfig
from eegprep.workflows.single_subject import init_single_subject_wf


class EEGPrepWorkflow:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.subject_workflows = [init_single_subject_wf(cfg, sub) for sub in cfg.participant_labels]

    def run(self) -> None:
        for wf in self.subject_workflows:
            wf.run()


def init_eegprep_wf(cfg: RunConfig) -> EEGPrepWorkflow:
    return EEGPrepWorkflow(cfg)
