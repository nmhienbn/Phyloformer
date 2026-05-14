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
        choices=[
            "mae",
            "mre",
            "quartet_close",
            "quartet_push",
            "quartet_combined",
        ],
        help=(
            "Training loss. Use 'mae' for PFBase pretraining and "
            "'mre', 'quartet_close', 'quartet_push', or 'quartet_combined' "
            "for fine-tuning."
        ),
    )
    train_grp.add_argument(
        "--quartet-close-sigma",
        default=1.0,
        type=float,
        help=(
            "Sigma term for the quartet close loss. "
            "Used only when --loss quartet_close."
        ),
    )
    train_grp.add_argument(
        "--quartet-push-lambda",
        default=0.5,
        type=float,
        help=(
            "Weight of quartet push term in quartet_push loss "
            "(loss = MRE + lambda * quartet_term)."
        ),
    )
    train_grp.add_argument(
        "--quartet-push-margin",
        default=0.05,
        type=float,
        help=(
            "Margin used by the quartet push term. "
            "Used only when --loss quartet_push."
        ),
    )
    train_grp.add_argument(
        "--quartet-num-samples",
        default=20,
        type=int,
        help=(
            "Number of random quartets sampled per batch when using "
            "--loss quartet_close, quartet_push, or quartet_combined."
        ),
    )
    train_grp.add_argument(
        "--quartet-combined-lambda",
        default=1.0,
        type=float,
        help=(
            "Weight of combined quartet term in quartet_combined loss "
            "(loss = MRE + lambda_q * additivity)."
        ),
    )
    train_grp.add_argument(
        "--quartet-combined-margin",
        default=0.05,
        type=float,
        help=(
            "Margin used by the quartet combined term. "
            "Used only when --loss quartet_combined."
        ),
    )
