from snake_rl.cli import build_train_parser


def main():
    parser = build_train_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
