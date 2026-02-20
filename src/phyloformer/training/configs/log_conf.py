from argparse import ArgumentParser

def build_log_config(parser: ArgumentParser) -> None:
    log_grp = parser.add_argument_group("LOGGING")
    log_grp.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Output directory to save losses and checkpoints",
    )
    log_grp.add_argument(
        "--log-every",
        "-E",
        default=100,
        type=int,
        help="Log training loss every n steps",
    )
    log_grp.add_argument(
        "--project-name",
        "-p",
        required=False,
        default="PHYLOFORMER_EXPERIMENTS",
        help="Project in which to save this run on WandB",
    )
    log_grp.add_argument(
        "--run-name",
        "-N",
        required=False,
        default=None,
        help="Name to give to the run on WandB",
    )
