#!/usr/bin/env python3

import argparse
import json
import math
import os
import pathlib
import random
import re
import sys
from pprint import pprint
from multiprocessing import cpu_count

import lightning
import lightning.pytorch.loggers as log
import torch  # type:ignore
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner import Tuner
from torch.optim import Adam  # type:ignore
from torch.utils.data import DataLoader  # type:ignore
from transformers import get_linear_schedule_with_warmup

from phyloformer.data import (
    PhyloDataset,
    precompute_alignment_cache,
    precompute_distance_cache,
)
from phyloformer.model import Phyloformer


def MAE(
    input: torch.Tensor, target: torch.Tensor, sqrt_preds: bool = False
) -> torch.Tensor:
    """Computes the Mean Absolute Error"""

    if sqrt_preds:
        input = input**2
    return torch.nn.L1Loss()(input, target).detach()


def MRE(
    input: torch.Tensor, target: torch.Tensor, sqrt_preds: bool = False
) -> torch.Tensor:
    """Computes the Mean Relative Error"""
    if sqrt_preds:
        input = input**2
    return torch.mean(torch.abs(input - target) / target).detach()


def listdir_paths(root):
    return [os.path.join(root, file) for file in os.listdir(root)]


# Removes all extensions (useful for still matching on predicted trees)
def stem(path):
    filename = pathlib.PurePath(path)
    return str(filename.stem).removesuffix("".join(filename.suffixes))


def make_pairs(treefiles, alnfiles, regex):
    """Find pairs of corresponding trees and MSAs"""
    alndict = {
        stem(alnfile): alnfile
        for alnfile in alnfiles
        if alnfile.endswith(".fa") or alnfile.endswith(".fasta")
    }
    pairs = []
    for treefile in treefiles:
        if not (treefile.endswith(".nwk") or treefile.endswith(".newick")):
            continue
        if regex is not None and not regex.search(treefile):
            continue
        alnfile = alndict.get(stem(treefile))
        if alnfile is None:
            print(f"Tree: {treefile}")
            print(f"Tree stem: {stem(treefile)}")
            raise IndexError(f"Tree: {treefile} has no corresponding alignment.")
        pairs.append((treefile, alnfile))
    return pairs


def choose_data(
    train_alignments, train_trees, train_regex, val_alignments, val_trees, val_regex
):
    """Find and select training and validation tree/MSA pairs"""
    # Choose training and validation examples
    if val_alignments is None and val_trees is None:
        regex = re.compile(train_regex) if train_regex is not None else None
        pairs = make_pairs(
            listdir_paths(train_trees), listdir_paths(train_alignments), regex
        )
        # Split data
        val_index = int(len(pairs) * 0.1)
        random.shuffle(pairs)
        val_pairs = pairs[:val_index]
        train_pairs = pairs[val_index:]
    elif val_alignments is not None and val_trees is not None:
        train_regex = re.compile(train_regex) if train_regex is not None else None
        val_regex = re.compile(val_regex) if val_regex is not None else None
        train_pairs = make_pairs(
            listdir_paths(train_trees),
            listdir_paths(train_alignments),
            train_regex,
        )
        val_pairs = make_pairs(
            listdir_paths(val_trees), listdir_paths(val_alignments), val_regex
        )
    else:
        raise ValueError(
            "You must either specify both validation trees and alignments "
            "or none of them."
        )

    return train_pairs, val_pairs


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


def resolve_trainer_hardware(args):
    trainer_hw = {}

    if args.accelerator is not None:
        trainer_hw["accelerator"] = args.accelerator
    if args.devices is not None:
        if args.devices < 1:
            raise ValueError("--devices must be >= 1.")
        trainer_hw["devices"] = args.devices
    if args.num_nodes is not None:
        if args.num_nodes < 1:
            raise ValueError("--num-nodes must be >= 1.")
        trainer_hw["num_nodes"] = args.num_nodes
    if args.strategy is not None:
        trainer_hw["strategy"] = args.strategy

    using_slurm_env = (
        (not args.ignore_slurm_env) and os.environ.get("SLURM_NODELIST") is not None
    )
    if using_slurm_env:
        trainer_hw.setdefault("accelerator", "gpu")
        trainer_hw.setdefault("devices", int(os.environ["SLURM_GPUS_ON_NODE"]))
        trainer_hw.setdefault("num_nodes", int(os.environ["SLURM_NNODES"]))
        if trainer_hw.get("devices", 1) > 1:
            trainer_hw.setdefault("strategy", "ddp")

    if "accelerator" not in trainer_hw:
        trainer_hw["accelerator"] = "cuda" if torch.cuda.is_available() else "cpu"

    if trainer_hw.get("devices", 1) > 1 and "strategy" not in trainer_hw:
        trainer_hw["strategy"] = "ddp"

    world_size = trainer_hw.get("devices", 1) * trainer_hw.get("num_nodes", 1)

    return trainer_hw, world_size, using_slurm_env


class LightningAxialTransformer(lightning.LightningModule):
    """Lighnint Object for Phyloformer training"""

    def __init__(
        self,
        nb_blocks: int,
        nb_heads: int,
        embed_dim: int,
        dropout: float,
        learning_rate: float,
        warmup_steps: int,
        total_steps: int,
        batch_size: int,
        optim_func,
        criterion,
    ):
        super().__init__()

        # Initialize model
        self.model = Phyloformer(
            n_blocks=nb_blocks,
            n_heads=nb_heads,
            h_dim=embed_dim,
            dropout=dropout,
        )

        # Optimizer stuff
        self.optim_func = optim_func
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.criterion = criterion
        self.batch_size = batch_size

        self.save_hyperparameters()
        # self.save_hyperparameters(ignore=["criterion"])

    def configure_optimizers(self):
        optimizer = self.optim_func(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model(x.float())
        loss = self.criterion(y_hat, y.type_as(y_hat).squeeze())
        self.log("train_loss", loss)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model(x.float())
        y = y.type_as(y_hat).squeeze()

        # Compute validation metrics and log them
        loss = self.criterion(y_hat, y)
        d = {"val_mre": MRE(y_hat, y, False), "val_mae": MAE(y_hat, y, False)}
        self.log_dict(dict(val_loss=loss, **d), sync_dist=True)

        return dict(loss=loss, **d)


class PhyloDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        train_pairs,
        val_pairs,
        batch_size,
        workers_train,
        workers_val,
        prefetch_factor,
        train_distance_cache_dir=None,
        val_distance_cache_dir=None,
        train_alignment_cache_dir=None,
        val_alignment_cache_dir=None,
    ):
        super().__init__()
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.batch_size = batch_size
        self.workers_train = workers_train
        self.workers_val = workers_val
        self.prefetch_factor = prefetch_factor
        self.train_distance_cache_dir = train_distance_cache_dir
        self.val_distance_cache_dir = val_distance_cache_dir
        self.train_alignment_cache_dir = train_alignment_cache_dir
        self.val_alignment_cache_dir = val_alignment_cache_dir

    def _loader_kwargs(self, num_workers):
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            dataset=PhyloDataset(
                self.train_pairs,
                distance_cache_dir=self.train_distance_cache_dir,
                alignment_cache_dir=self.train_alignment_cache_dir,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            **self._loader_kwargs(self.workers_train),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=PhyloDataset(
                self.val_pairs,
                distance_cache_dir=self.val_distance_cache_dir,
                alignment_cache_dir=self.val_alignment_cache_dir,
            ),
            batch_size=self.batch_size,
            **self._loader_kwargs(self.workers_val),
        )


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file. CLI flags override config values.",
    )
    pre_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        "train PF instance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[config_parser],
    )

    # DATA
    data_grp = parser.add_argument_group("data", description="Data IO parameters")
    data_grp.add_argument(
        "--train-trees", "-t", required=False, help="Directory with training trees"
    )
    data_grp.add_argument(
        "--train-alignments",
        "-a",
        required=False,
        help="Directory with training alignments",
    )
    data_grp.add_argument(
        "--val-trees", "-T", required=False, help="Directory with validation trees"
    )
    data_grp.add_argument(
        "--val-alignments",
        "-A",
        required=False,
        help="Directory with validation alignments",
    )
    data_grp.add_argument(
        "--train-regex", "-r", default=None, help="Regex to filter training examples"
    )
    data_grp.add_argument(
        "--val-regex", "-R", default=None, help="Regex to filter validation examples"
    )
    data_grp.add_argument(
        "--distance-cache-dir",
        default=None,
        type=str,
        help=(
            "Directory for cached distance targets (.pt). "
            "Uses <distance-cache-dir>/train and <distance-cache-dir>/val."
        ),
    )
    data_grp.add_argument(
        "--precompute-distance-cache",
        action="store_true",
        help=(
            "Precompute and save all target distance vectors before training "
            "(requires --distance-cache-dir)."
        ),
    )
    data_grp.add_argument(
        "--overwrite-distance-cache",
        action="store_true",
        help="Overwrite existing cached target distances when precomputing.",
    )
    data_grp.add_argument(
        "--alignment-cache-dir",
        default=None,
        type=str,
        help=(
            "Directory for cached one-hot alignments (.pt). "
            "Uses <alignment-cache-dir>/train and <alignment-cache-dir>/val."
        ),
    )
    data_grp.add_argument(
        "--precompute-alignment-cache",
        action="store_true",
        help=(
            "Precompute and save all one-hot alignments before training "
            "(requires --alignment-cache-dir)."
        ),
    )
    data_grp.add_argument(
        "--overwrite-alignment-cache",
        action="store_true",
        help="Overwrite existing cached alignments when precomputing.",
    )

    # STARTING POINT
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

    # ARCHITECTURE
    arch_grp = parser.add_argument_group("MODEL ARCHITECTURE")
    arch_grp.add_argument(
        "--dropout", "-D", default=0.0, type=float, help="Dropout proportion"
    )
    arch_grp.add_argument(
        "--nb-blocks", "-b", default=6, type=int, help="Number of PF blocks"
    )
    arch_grp.add_argument(
        "--embed-dim", "-d", default=64, type=int, help="Number of embedding dimensions"
    )
    arch_grp.add_argument(
        "--nb-heads", "-H", default=4, type=int, help="Number of attention heads"
    )

    # TRAINING CONTROL
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

    # LOGGING
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

    # COMPUTE
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

    # MISC
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

    if pre_args.config is not None:
        config = load_json_config(pre_args.config)
        apply_config_defaults(parser, config, pre_args.config)

    args = parser.parse_args()
    if args.train_trees is None or args.train_alignments is None:
        raise ValueError(
            "You must provide both --train-trees and --train-alignments "
            "(either through CLI flags or --config)."
        )
    if args.precompute_distance_cache and args.distance_cache_dir is None:
        raise ValueError(
            "--precompute-distance-cache requires --distance-cache-dir to be set."
        )
    if args.precompute_alignment_cache and args.alignment_cache_dir is None:
        raise ValueError(
            "--precompute-alignment-cache requires --alignment-cache-dir to be set."
        )

    # Initialize logger
    wandb_logger = log.WandbLogger(
        save_dir=args.output_dir,
        project=args.project_name,
        name=args.run_name,
        offline=True,
    )

    print(f"Training with args:\n{args}")

    VAL_CHECK_STEPS = args.check_val_every
    LOGGING_STEPS = args.log_every
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1.")

    N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    NUM_WORKERS = N_CPUS // 2

    global WORKERS_TRAIN
    global WORKERS_VAL
    default_workers_train = max(NUM_WORKERS, N_CPUS - NUM_WORKERS)
    default_workers_val = min(NUM_WORKERS, N_CPUS - NUM_WORKERS)

    WORKERS_TRAIN = (
        args.workers_train
        if args.workers_train is not None
        else default_workers_train
    )
    WORKERS_VAL = args.workers_val if args.workers_val is not None else default_workers_val
    if WORKERS_TRAIN < 0 or WORKERS_VAL < 0:
        raise ValueError("--workers-train and --workers-val must be >= 0.")

    print(
        f"Assigning {WORKERS_TRAIN} training, and {WORKERS_VAL} validation data-loading workers"
    )

    # Make output directory for logging and checkpoints
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seeds
    seed = 1337
    lightning.pytorch.seed_everything(seed, workers=True)

    bs = args.batch_size
    hd = args.embed_dim
    nb = args.nb_blocks
    lr = args.learning_rate
    wp = args.warmup_steps
    dr = args.dropout

    train_pairs, val_pairs = choose_data(
        args.train_alignments,
        args.train_trees,
        args.train_regex,
        args.val_alignments,
        args.val_trees,
        args.val_regex,
    )

    train_distance_cache_dir = None
    val_distance_cache_dir = None
    if args.distance_cache_dir is not None:
        cache_root = pathlib.Path(args.distance_cache_dir)
        train_distance_cache_dir = str(cache_root / "train")
        val_distance_cache_dir = str(cache_root / "val")

        if args.precompute_distance_cache:
            print("Precomputing distance cache for training split...")
            precompute_distance_cache(
                train_pairs,
                train_distance_cache_dir,
                overwrite=args.overwrite_distance_cache,
            )
            print("Precomputing distance cache for validation split...")
            precompute_distance_cache(
                val_pairs,
                val_distance_cache_dir,
                overwrite=args.overwrite_distance_cache,
            )

    train_alignment_cache_dir = None
    val_alignment_cache_dir = None
    if args.alignment_cache_dir is not None:
        cache_root = pathlib.Path(args.alignment_cache_dir)
        train_alignment_cache_dir = str(cache_root / "train")
        val_alignment_cache_dir = str(cache_root / "val")

        if args.precompute_alignment_cache:
            print("Precomputing alignment cache for training split...")
            precompute_alignment_cache(
                train_pairs,
                train_alignment_cache_dir,
                overwrite=args.overwrite_alignment_cache,
            )
            print("Precomputing alignment cache for validation split...")
            precompute_alignment_cache(
                val_pairs,
                val_alignment_cache_dir,
                overwrite=args.overwrite_alignment_cache,
            )

    trainer_hardware_args, world_size, using_slurm_env = resolve_trainer_hardware(args)
    if pre_args.config is not None:
        print(f"Loaded config: {pre_args.config}")
    if using_slurm_env:
        print("Using SLURM environment variables for distributed setup.")

    datamodule = PhyloDataModule(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        batch_size=args.batch_size,
        workers_train=WORKERS_TRAIN,
        workers_val=WORKERS_VAL,
        prefetch_factor=args.prefetch_factor,
        train_distance_cache_dir=train_distance_cache_dir,
        val_distance_cache_dir=val_distance_cache_dir,
        train_alignment_cache_dir=train_alignment_cache_dir,
        val_alignment_cache_dir=val_alignment_cache_dir,
    )
    total_steps = (
        math.ceil(len(train_pairs) / (args.batch_size * world_size)) * args.nb_epochs
    )

    criterion = torch.nn.L1Loss()
    model = LightningAxialTransformer(
        nb_blocks=args.nb_blocks,
        nb_heads=args.nb_heads,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        batch_size=args.batch_size,
        optim_func=Adam,
        criterion=criterion,
    )

    identifier = (
        f"LR_{args.learning_rate}_O_Adam_"
        f"L_L1_E_{args.nb_epochs}_BS_{args.batch_size}_"
        f"NB_{args.nb_blocks}_NH_{args.nb_heads}_HD_{args.embed_dim}_"
        f"D_{0.0}_W{args.warmup_steps}"
    )

    # Load weights from pre-trained PF instance
    if args.base_model is not None:
        ckpt = torch.load(args.base_model, map_location="cpu")
        model = LightningAxialTransformer(**ckpt["hyper_parameters"])
        model.load_state_dict(ckpt["state_dict"])
        del ckpt  # Free space used by the checkpoint

    # Load hyper-parameters if starting up from a checkpoint
    if args.load_checkpoint is not None:
        ckpt = torch.load(args.load_checkpoint, map_location="cpu")
        model = LightningAxialTransformer(**ckpt["hyper_parameters"])

        # This is a little hacky...
        k, v = [(k, v) for k, v in ckpt["callbacks"].items() if "ModelCheckpoint" in k][
            0
        ]
        identifier = v["dirpath"].split("./")[-1].removeprefix("checkpoints_")
        del ckpt

    # Find batch size and exit if necessary
    if args.find_batch_size:
        trainer = lightning.Trainer()
        tuner = Tuner(trainer)
        bs = tuner.scale_batch_size(
            model,
            mode="binsearch",
        )
        print(f"Lightning found an optimal batch size of: {bs}.")
        sys.exit(0)

    # Manually save hyperparameter string just in case
    wandb_logger.log_hyperparams({"identifier": identifier})

    # Early stopping callbacks if needed
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, f"checkpoints_{identifier}"),
            filename="{epoch}-{step}-{val_loss:.4f}-{train_loss:.4f}",
            save_top_k=-1,  # Keep all checkpoints
            save_last=True,  # Add symbolic link to point to last checkpoint
            every_n_train_steps=VAL_CHECK_STEPS,
            save_on_train_epoch_end=False,  # Save after validation so the value is correct in filename
        )
    ]
    if args.hard_loss_ceiling is not None:
        callbacks.append(
            EarlyStopping(
                monitor="train_loss",
                mode="min",
                check_finite=True,
                verbose=True,
                patience=10_000,  # Simulate infinite patience so that this only checks for divergence
                divergence_threshold=args.hard_loss_ceiling,
            )
        )
    if args.no_improvement_stop is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=args.no_improvement_stop,
                verbose=True,
            )
        )

    trainer_args = {
        "max_epochs": args.nb_epochs,
        "log_every_n_steps": LOGGING_STEPS,
        "val_check_interval": VAL_CHECK_STEPS,
        "logger": wandb_logger,
        "callbacks": callbacks,
        "precision": args.precision,
        **trainer_hardware_args,
    }

    # Run profiler for 30 steps
    if args.profile:
        trainer_args["max_steps"] = 10
        # trainer_args["profiler"] = "simple"
        trainer_args["profiler"] = PyTorchProfiler(
            dirpath=os.path.join(args.output_dir, f"profile_{identifier}"),
            profile_memory=True,
            record_shapes=True,
            with_modules=True,
        )

    print("INIT TRAINER WITH ARGS:")
    pprint(trainer_args)

    # Initialize trainer
    trainer = lightning.Trainer(**trainer_args)

    # Get training arguments
    train_args = dict(model=model)
    if args.load_checkpoint is not None:
        # train_args["model"] = LightningAxialTransformer.load_from_checkpoint(args.load_checkpoint)
        train_args["ckpt_path"] = args.load_checkpoint

    print("LAUNCHING TRAINING WITH ARGS:")
    pprint(train_args)

    # Train the model
    trainer.fit(**train_args, datamodule=datamodule)
