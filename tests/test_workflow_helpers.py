from pathlib import Path

from eegprep.cli.workflow import collect_eeg_data, write_run_config
from eegprep.config import RunConfig


def _cfg(tmp_path: Path) -> RunConfig:
    return RunConfig(
        bids_root=tmp_path / "bids",
        derivatives_root=tmp_path / "derivatives",
        analysis_level="participant",
        participant_labels=["01"],
    )


def test_collect_eeg_data_finds_edf_file(tmp_path: Path):
    cfg = _cfg(tmp_path)
    eeg_dir = cfg.bids_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-01_task-rest_run-01_eeg.edf").write_bytes(b"test")

    runs = collect_eeg_data(cfg, "01")

    assert len(runs) == 1
    assert runs[0].task == "rest"
    assert runs[0].run == "01"


def test_write_run_config_writes_toml(tmp_path: Path):
    cfg = _cfg(tmp_path)

    out_file = write_run_config(cfg, "01", "abc123")

    assert out_file.exists()
    text = out_file.read_text(encoding="utf-8")
    assert "[run]" in text
    assert "analysis_level = \"participant\"" in text
