import os
from multiprocessing import cpu_count
import torch



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

    using_slurm_env = (not args.ignore_slurm_env) and os.environ.get(
        "SLURM_NODELIST"
    ) is not None
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


def resolve_worker_counts(args):
    N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    NUM_WORKERS = N_CPUS // 2

    default_workers_train = max(NUM_WORKERS, N_CPUS - NUM_WORKERS)
    default_workers_val = min(NUM_WORKERS, N_CPUS - NUM_WORKERS)

    workers_train = (
        args.workers_train if args.workers_train is not None else default_workers_train
    )
    workers_val = (
        args.workers_val if args.workers_val is not None else default_workers_val
    )
    if workers_train < 0 or workers_val < 0:
        raise ValueError("--workers-train and --workers-val must be >= 0.")

    print(
        f"Assigning {workers_train} training, and {workers_val} validation data-loading workers"
    )

    return workers_train, workers_val
