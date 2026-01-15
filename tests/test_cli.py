from snake_rl.cli import build_train_parser


def test_train_parser_defaults():
    parser = build_train_parser()
    args = parser.parse_args([])
    assert args.config == "config/default.yaml"
    assert args.mode == "train"


def test_play_mode_args():
    parser = build_train_parser()
    args = parser.parse_args(["--mode", "play", "--checkpoint", "model.pt"])
    assert args.mode == "play"
    assert args.checkpoint == "model.pt"
    assert args.fps == 10
    assert args.episodes == 3
    assert args.scale == 20
    assert args.max_steps is None
