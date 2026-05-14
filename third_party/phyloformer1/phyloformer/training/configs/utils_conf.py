from argparse import ArgumentParser

def build_utils_config(parser: ArgumentParser) -> None:
    utils_grp = parser.add_argument_group(
        "UTILS", description="Utilities that are run instead of training"
    )
    utils_grp.add_argument(
        "--find-batch-size",
        action="store_true",
        help="Run the lightning batch_size finder (skips training)",
    )

    utils_grp.add_argument(
        "--profile", action="store_true", help="Run profiler for a few steps and exit"
    )
