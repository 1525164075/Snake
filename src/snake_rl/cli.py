import argparse


def build_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "ga", "play"], default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--scale", type=int, default=20)
    return parser
