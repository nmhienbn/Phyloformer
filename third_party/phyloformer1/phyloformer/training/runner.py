import math
import os
import sys
from pprint import pprint

import lightning
import lightning.pytorch.loggers as log
import torch
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner import Tuner
from torch.optim import Adam

from phyloformer.training.data_module import PhyloDataModule
from phyloformer.training.hardware import (
    resolve_trainer_hardware,
    resolve_worker_counts,
)
from phyloformer.training.lightning_module import LightningAxialTransformer
from phyloformer.training.data_loader import choose_data
from phyloformer.training.runtime import (
    build_callbacks,
    build_loss,
    load_pretrained_state_dict,
    prepare_caches,
)


def run_training(args):
    # Initialize logger
    wandb_logger = log.WandbLogger(
        save_dir=args.output_dir,
        project=args.project_name,
        name=args.run_name,
        offline=not args.wandb_sync,
    )

    print(f"Training with args:\n{args}")

    VAL_CHECK_STEPS = args.check_val_every
    LOGGING_STEPS = args.log_every

    WORKERS_TRAIN, WORKERS_VAL = resolve_worker_counts(args)

    # Make output directory for logging and checkpoints
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seeds
    seed = 1337
    lightning.pytorch.seed_everything(seed, workers=True)

    train_pairs, val_pairs = choose_data(
        args.train_alignments,
        args.train_trees,
        args.train_regex,
        args.val_alignments,
        args.val_trees,
        args.val_regex,
    )

    (
        train_distance_cache_dir,
        val_distance_cache_dir,
        train_alignment_cache_dir,
        val_alignment_cache_dir,
    ) = prepare_caches(args, train_pairs, val_pairs)

    trainer_hardware_args, world_size, using_slurm_env = resolve_trainer_hardware(args)
    if using_slurm_env:
        print("Using SLURM environment variables for distributed setup.")

    effective_global_batch_size = (
        args.batch_size * world_size * args.accumulate_grad_batches
    )
    print(
        "Effective global batch size "
        f"(micro_batch x world_size x accumulate): "
        f"{args.batch_size} x {world_size} x {args.accumulate_grad_batches} "
        f"= {effective_global_batch_size}"
    )

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
    train_batches_per_epoch = math.ceil(
        len(train_pairs) / (args.batch_size * world_size)
    )
    optimizer_steps_per_epoch = math.ceil(
        train_batches_per_epoch / args.accumulate_grad_batches
    )
    total_steps = optimizer_steps_per_epoch * args.nb_epochs

    criterion, loss_tag = build_loss(args)

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
        n_seqs=args.n_seqs,
    )

    identifier = (
        f"LR_{args.learning_rate}_O_Adam_"
        f"L_{loss_tag}_E_{args.nb_epochs}_BS_{args.batch_size}_"
        f"ACC_{args.accumulate_grad_batches}_"
        f"NB_{args.nb_blocks}_NH_{args.nb_heads}_HD_{args.embed_dim}_"
        f"D_{0.0}_W{args.warmup_steps}"
    )

    # Load weights from pre-trained PF instance
    if args.base_model is not None:
        load_pretrained_state_dict(model, args.base_model)
        print(f"Loaded base model weights from: {args.base_model}")

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

    callbacks = build_callbacks(args, identifier, VAL_CHECK_STEPS)

    trainer_args = {
        "max_epochs": args.nb_epochs,
        "log_every_n_steps": LOGGING_STEPS,
        "val_check_interval": VAL_CHECK_STEPS * args.accumulate_grad_batches,
        "logger": wandb_logger,
        "callbacks": callbacks,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "precision": args.precision,
        **trainer_hardware_args,
    }
    if args.max_steps is not None:
        trainer_args["max_steps"] = args.max_steps

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
