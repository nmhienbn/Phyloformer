from argparse import ArgumentParser

def build_compute_config(parser: ArgumentParser) -> None:
    compute_grp = parser.add_argument_group(
        "COMPUTE",
        description=(
            "Hardware/distribution settings. Can be provided explicitly without SLURM."
        ),
    )
    compute_grp.add_argument(
        "--accelerator",
        default=None,
        choices=["cpu", "gpu", "cuda"],
        help="Lightning accelerator to use. Defaults to CUDA if available.",
    )
    compute_grp.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of devices per node to use (e.g. 6 for 6 GPUs).",
    )
    compute_grp.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Number of nodes for distributed training.",
    )
    compute_grp.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Lightning strategy (e.g. ddp).",
    )
    compute_grp.add_argument(
        "--ignore-slurm-env",
        action="store_true",
        help="Ignore SLURM_* environment variables even if they are set.",
    )
