import os
import pathlib

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from phyloformer.losses import (
    QuartetMRELoss,
    QuartetPushCloseMRELoss,
    QuartetSiameseLoss,
    QuartetSiameseMRELoss,
    QuartetSoftmaxLoss,
)
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

    # Fine-tuning should load only backbone weights, not criterion/scheduler/etc.
    backbone = model.model
    target_state = backbone.state_dict()
    backbone_state = {}
    skipped_prefix = {}

    for src_key, src_tensor in state_dict.items():
        if src_key.startswith("model."):
            dst_key = src_key[len("model.") :]
        elif src_key.startswith("criterion."):
            skipped_prefix[src_key] = "criterion key"
            continue
        else:
            # Support raw Phyloformer checkpoints without Lightning's "model." prefix.
            dst_key = src_key

        if dst_key not in target_state:
            skipped_prefix[src_key] = "unknown key"
            continue

        backbone_state[dst_key] = src_tensor

    if not backbone_state:
        raise RuntimeError(
            f"No compatible backbone weights found in checkpoint: {checkpoint_path}"
        )

    loadable_state = {}
    skipped = dict(skipped_prefix)
    for dst_key, src_tensor in backbone_state.items():
        if target_state[dst_key].shape != src_tensor.shape:
            skipped[f"model.{dst_key}"] = (
                f"shape mismatch {tuple(src_tensor.shape)} -> "
                f"{tuple(target_state[dst_key].shape)}"
            )
            continue
        loadable_state[dst_key] = src_tensor

    if not loadable_state:
        raise RuntimeError(
            "No shape-compatible backbone tensors found in checkpoint "
            f"{checkpoint_path}."
        )

    backbone.load_state_dict(loadable_state, strict=False)
    print(
        "Loaded pretrained backbone weights: "
        f"{len(loadable_state)}/{len(target_state)} tensors from {checkpoint_path}."
    )
    if skipped:
        preview = ", ".join(
            f"{k} ({v})" for k, v in list(skipped.items())[:5]
        )
        print(
            "Skipped incompatible checkpoint tensors: "
            f"{len(skipped)} total. Examples: {preview}"
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


def _build_mae_loss(_args):
    return torch.nn.L1Loss(), "L1"


def _build_mre_loss(_args):
    return MRELoss(), "MRE"


def _build_quartet_siamese_loss(args):
    criterion = QuartetSiameseLoss(
        sigma=args.quartet_sigma,
        num_quartets=args.quartet_num_samples,
    )
    loss_tag = f"QSIAM_S{args.quartet_sigma:g}_Q{args.quartet_num_samples}"
    return criterion, loss_tag


def _build_quartet_softmax_loss(args):
    criterion = QuartetSoftmaxLoss(
        temperature=args.quartet_softmax_temperature,
        sigma=args.quartet_sigma,
        num_quartets=args.quartet_num_samples,
    )
    loss_tag = (
        f"QSOFT_T{args.quartet_softmax_temperature:g}_"
        f"S{args.quartet_sigma:g}_Q{args.quartet_num_samples}"
    )
    return criterion, loss_tag


def _build_quartet_mre_loss(args):
    criterion = QuartetMRELoss(
        lambda_q=args.quartet_mre_lambda,
        margin=args.quartet_mre_margin,
        num_quartets=args.quartet_num_samples,
    )
    loss_tag = (
        f"QMRE_L{args.quartet_mre_lambda:g}_"
        f"M{args.quartet_mre_margin:g}_Q{args.quartet_num_samples}"
    )
    return criterion, loss_tag


def _build_quartet_siamese_mre_loss(args):
    criterion = QuartetSiameseMRELoss(
        sigma=args.quartet_sigma,
        num_quartets=args.quartet_num_samples,
    )
    loss_tag = f"QSIAM_MRE_S{args.quartet_sigma:g}_Q{args.quartet_num_samples}"
    return criterion, loss_tag


def _build_quartet_push_close_loss(args):
    criterion = QuartetPushCloseMRELoss(
        lambda_q=args.quartet_lambda_q,
        margin=args.quartet_margin,
        num_quartets=args.quartet_num_samples,
    )
    loss_tag = (
        f"QPUSH_L{args.quartet_lambda_q:g}_"
        f"M{args.quartet_margin:g}_Q{args.quartet_num_samples}"
    )
    return criterion, loss_tag


LOSS_BUILDERS = {
    "mae": _build_mae_loss,
    "mre": _build_mre_loss,
    "quartet_siamese": _build_quartet_siamese_loss,
    "quartet_softmax": _build_quartet_softmax_loss,
    "quartet_mre": _build_quartet_mre_loss,
    "quartet_siamese_mre": _build_quartet_siamese_mre_loss,
    "quartet_push_close": _build_quartet_push_close_loss,
}


def build_loss(args):
    try:
        builder = LOSS_BUILDERS[args.loss]
    except KeyError as exc:
        valid = ", ".join(sorted(LOSS_BUILDERS.keys()))
        raise ValueError(
            f"Unsupported --loss value: {args.loss}. Valid values: {valid}"
        ) from exc
    return builder(args)


def build_callbacks(args, identifier, VAL_CHECK_STEPS):

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
    return callbacks
