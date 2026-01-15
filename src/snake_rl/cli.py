import argparse


def build_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "ga"], default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--checkpoint", default="")
    return parser
