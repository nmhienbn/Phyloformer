from argparse import ArgumentParser

def build_starting_point_config(parser: ArgumentParser) -> None:
    start_grp = parser.add_argument_group(
        "STARTING POINT",
        description=(
            "Model starting point, specifying one of these options will "
            "override options from the [ARCHITECTURE] parameter group."
        ),
    )
    start_grp.add_argument(
        "--load-checkpoint",
        "-c",
        default=None,
        help="Path to checkpoint to resume training from",
    )
    start_grp.add_argument(
        "--base-model", "-m", required=False, type=str, help="Base model to fine-tune"
    )
