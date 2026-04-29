from eegprep.cli.parser import build_parser


def test_parser_accepts_minimal_invocation():
    parser = build_parser()
    args = parser.parse_args(["/bids", "/deriv", "participant"])
    assert args.level == "minimal"
    assert args.source_recon == "none"
