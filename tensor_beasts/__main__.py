from tensor_beasts.config import load_config
from tensor_beasts.main import main, parse_args


def run() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    print(config)
    main(config)


if __name__ == "__main__":
    run()
