from argparse import ArgumentParser

def build_train_config(parser: ArgumentParser) -> None:
    train_grp = parser.add_argument_group(
        "TRAINING", description="Training control parameters"
    )
    train_grp.add_argument(
        "--nb-epochs", "-e", default=100, type=int, help="Number of epochs to train for"
    )
    train_grp.add_argument(
        "--warmup-steps", "-w", default=5000, type=int, help="Number of warmup steps"
    )
    train_grp.add_argument(
        "--learning-rate",
        "-l",
        default=1e-4,
        type=float,
        help="Traget starting learning rate",
    )
    train_grp.add_argument(
        "--check-val-every",
        "-C",
        default=10_000,
        type=int,
        help="Check validation dataset every n steps",
    )
    train_grp.add_argument(
        "--batch-size", "-s", default=4, type=int, help="Training batch size"
    )
    train_grp.add_argument(
        "--accumulate-grad-batches",
        default=1,
        type=int,
        help=(
            "Number of micro-batches to accumulate before each optimizer step. "
            "Use with a smaller --batch-size to reduce VRAM while keeping "
            "a similar effective global batch size."
        ),
    )
    train_grp.add_argument(
        "--workers-train",
        default=None,
        type=int,
        help=(
            "Number of dataloader workers for training. "
            "If not provided, auto-derived from SLURM_CPUS_PER_TASK/cpu_count()."
        ),
    )
    train_grp.add_argument(
        "--workers-val",
        default=None,
        type=int,
        help=(
            "Number of dataloader workers for validation. "
            "If not provided, auto-derived from SLURM_CPUS_PER_TASK/cpu_count()."
        ),
    )
    train_grp.add_argument(
        "--prefetch-factor",
        default=2,
        type=int,
        help=(
            "DataLoader prefetch_factor used when num_workers > 0 "
            "(persistent_workers is enabled automatically)."
        ),
    )
    train_grp.add_argument(
        "--max-steps", "-M", default=None, type=int, help="Max number of training steps"
    )
    train_grp.add_argument(
        "--no-improvement-stop",
        "-n",
        default=5,
        type=int,
        help="Number of checks with no improvement before stopping early",
    )
    train_grp.add_argument(
        "--hard-loss-ceiling",
        "-L",
        default=3.0,
        type=float,
        help="Max value of loss over which the training stops",
    )
    train_grp.add_argument(
        "--precision",
        default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help=(
            "Trainer precision mode. Use bf16-mixed on A100 for lower memory "
            "and higher throughput."
        ),
    )
    train_grp.add_argument(
        "--loss",
        default="mae",
        choices=["mae", "mre", "quartet_siamese"],
        help=(
            "Training loss. Use 'mae' for PFBase pretraining and "
            "'mre' or 'quartet_siamese' for fine-tuning."
        ),
    )
    train_grp.add_argument(
        "--quartet-sigma",
        default=0.0,
        type=float,
        help=(
            "Sigma term for QuartetSiameseLoss. "
            "Used only when --loss quartet_siamese."
        ),
    )
