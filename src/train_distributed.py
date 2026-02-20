from phyloformer.training.config import (
    build_parser_with_config,
    validate_args,
)
from phyloformer.training.runner import run_training


def main():
    parser = build_parser_with_config()
    args = parser.parse_args()
    validate_args(args)
    run_training(args)


if __name__ == "__main__":
    main()
