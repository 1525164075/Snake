from snake_rl.cli import build_train_parser


def test_train_parser_defaults():
    parser = build_train_parser()
    args = parser.parse_args([])
    assert args.config == "config/default.yaml"
