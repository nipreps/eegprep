from pathlib import Path

from eegprep.cli.parser import build_parser
from eegprep.cli.participants import discover_participants


def test_parser_accepts_minimal_invocation():
    parser = build_parser()
    args = parser.parse_args(["/bids", "/deriv", "participant"])
    assert args.level == "minimal"
    assert args.source_recon == "none"


def test_discover_participants_from_bids_root(tmp_path: Path):
    (tmp_path / "sub-01").mkdir()
    (tmp_path / "sub-02").mkdir()
    (tmp_path / "code").mkdir()

    labels = discover_participants(tmp_path)

    assert labels == ["01", "02"]
