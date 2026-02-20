import os
import pathlib

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from phyloformer.losses.quartet_siamese import QuartetSiameseLoss
from phyloformer.data import (
    precompute_alignment_cache,
    precompute_distance_cache,
)
from phyloformer.losses.mre import MRELoss


def load_pretrained_state_dict(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Unsupported checkpoint format at {checkpoint_path}. "
            "Expected a state_dict or a Lightning checkpoint with 'state_dict'."
        )

    # Lightning checkpoints store keys as model.*, raw model checkpoints usually do not.
    candidates = [state_dict]
    if not any(str(k).startswith("model.") for k in state_dict.keys()):
        candidates.append({f"model.{k}": v for k, v in state_dict.items()})

    last_error = None
    for candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except RuntimeError as err:
            last_error = err

    raise RuntimeError(
        f"Failed loading checkpoint weights from {checkpoint_path}: {last_error}"
    )


def prepare_caches(args, train_pairs, val_pairs):
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
    return (
        train_distance_cache_dir,
        val_distance_cache_dir,
        train_alignment_cache_dir,
        val_alignment_cache_dir,
    )


def build_loss(args):
    if args.loss == "mae":
        criterion = torch.nn.L1Loss()
        loss_tag = "L1"
    elif args.loss == "mre":
        criterion = MRELoss()
        loss_tag = "MRE"
    elif args.loss == "quartet_siamese":
        criterion = QuartetSiameseLoss(sigma=args.quartet_sigma)
        loss_tag = f"QSIAM_S{args.quartet_sigma:g}"
    else:
        raise ValueError(f"Unsupported --loss value: {args.loss}")

    return criterion, loss_tag


def build_callbacks(args, identifier, VAL_CHECK_STEPS):

    # Early stopping callbacks if needed
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, f"checkpoints_{identifier}"),
            filename="{epoch}-{step}-{val_loss:.4f}-{train_loss:.4f}",
            save_top_k=-1,  # Keep all checkpoints
            save_last=True,  # Add symbolic link to point to last checkpoint
            every_n_train_steps=VAL_CHECK_STEPS * args.accumulate_grad_batches,
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
    return callbacks
