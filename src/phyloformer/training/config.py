import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from .configs.architecture_conf import build_architecture_config
from .configs.compute_conf import build_compute_config
from .configs.data_conf import build_data_config
from .configs.log_conf import build_log_config
from .configs.starting_conf import build_starting_point_config
from .configs.train_conf import build_train_config
from .configs.utils_conf import build_utils_config


def build_config() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file. CLI flags override config values.",
    )
    return parser


def build_parser(config_parser: ArgumentParser) -> ArgumentParser:
    parser = ArgumentParser(
        "train PF instance",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[config_parser],
    )
    build_data_config(parser)
    build_starting_point_config(parser)
    build_architecture_config(parser)
    build_train_config(parser)
    build_log_config(parser)
    build_compute_config(parser)
    build_utils_config(parser)
    return parser


def validate_args(args):
    if args.train_trees is None or args.train_alignments is None:
        raise ValueError(
            "You must provide both --train-trees and --train-alignments "
            "(either through CLI flags or --config)."
        )
    if args.base_model is not None and args.load_checkpoint is not None:
        raise ValueError("Use either --base-model or --load-checkpoint, not both.")
    if args.precompute_distance_cache and args.distance_cache_dir is None:
        raise ValueError(
            "--precompute-distance-cache requires --distance-cache-dir to be set."
        )
    if args.precompute_alignment_cache and args.alignment_cache_dir is None:
        raise ValueError(
            "--precompute-alignment-cache requires --alignment-cache-dir to be set."
        )

    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1.")
    if args.accumulate_grad_batches < 1:
        raise ValueError("--accumulate-grad-batches must be >= 1.")


def build_parser_with_config() -> ArgumentParser:
    config_parser = build_config()
    pre_args, _ = config_parser.parse_known_args()
    parser = build_parser(config_parser)

    if pre_args.config is not None:
        print(f"Loaded config: {pre_args.config}")
        config = load_json_config(pre_args.config)
        apply_config_defaults(parser, config, pre_args.config)
    return parser


def load_json_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a JSON object (key/value pairs): {config_path}"
        )
    return config


def apply_config_defaults(parser, config, config_path):
    valid_keys = {action.dest for action in parser._actions}
    unknown = sorted(set(config.keys()) - valid_keys)
    if unknown:
        raise ValueError(
            f"Unknown keys in config file {config_path}: {', '.join(unknown)}"
        )
    parser.set_defaults(**config)
