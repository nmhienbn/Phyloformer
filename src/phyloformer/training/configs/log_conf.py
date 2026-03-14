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
    log_grp.add_argument(
        "--wandb-sync",
        dest="wandb_sync",
        action="store_true",
        help="Enable online WandB logging (sync during training)",
    )
    log_grp.add_argument(
        "--no-wandb-sync",
        dest="wandb_sync",
        action="store_false",
        help="Disable online WandB logging and keep runs offline",
    )
    log_grp.set_defaults(wandb_sync=False)
