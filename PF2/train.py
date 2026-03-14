# pyright: basic

import getpass
import grp
import io
import os
import tarfile
from functools import partial
from math import cos, exp, log, pi
from pprint import pprint
from sys import platform
from time import time
from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from wandb.integration.lightning.fabric import WandbLogger

from BayesNJ.core import (
    DistributedSameSizeSampler,
    MultisizeUnambiguousMergeOrderDataset,
    NJDataset,
    UnambiguousMergeOrderDataset,
    batch_tree_probability_with_merges_branches,
)
from BayesNJ.evopf import EvoPF
from BayesNJ.pf_sdk.pf.data import (
    CheckpointableDsitributedSampler,
    CheckpointableRandomSampler,
)
from BayesNJ.pf_sdk.pf.modules import PhyloformerSeq, PhyloformerSeqMixed, seq2pairs
from BayesNJ.pf_sdk.pf.training import generate_run_name
from BayesNJ.treefuncs import batch_compute_tree_logprob

SAVE_CHECKPOINT_SPACE = os.environ.get("PF_SAVE_CHECKPOINT_SPACE", "False").lower() in [
    "true",
    "t",
    "1",
]


def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def _get_linear_schedule_with_warmup_lr_lambda_scaled(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    base_size: int,
    curr_size: int,
):
    scale = float(curr_size / base_size)

    if current_step < num_warmup_steps:
        return scale * float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        scale
        * float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def linear_growth_annealing(x, start, sched_steps):
    return min(1.0, start + x / sched_steps)


def cosine_growth_annealing(x, start, sched_steps):
    cos_value = start + 0.5 * (1.0 - start) * (1 + cos(pi * x / sched_steps))
    return max(1 - cos_value, float(x > sched_steps))


def exp_growth_annealing(x, start, sched_steps):
    return min(exp(-log(start) * (x - sched_steps) / sched_steps), 1)


def smooth_clamp(x, max_bound):
    return torch.tanh(x / max_bound) * max_bound


ANNEALING_START = float(os.environ.get("PF_ANNEALING_START", 1e-10))
ANNEALING_FUNCTYPE = os.environ.get("PF_ANNEALING_TYPE", "linear").upper()
if ANNEALING_FUNCTYPE == "LINEAR":
    ANNEALING_FUNC = linear_growth_annealing
elif ANNEALING_FUNCTYPE == "COSINE":
    ANNEALING_FUNC = cosine_growth_annealing
elif ANNEALING_FUNCTYPE == "EXPONENTIAL":
    ANNEALING_FUNC = exp_growth_annealing
else:
    raise ValueError(f"Unknown annealing type: {ANNEALING_FUNCTYPE}")


class MultisizeLRScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        base_size: Optional[int],
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ) -> None:
        lr_lambda = partial(
            _get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.base_size = base_size or 1.0
        self.curr_size = base_size or 1.0
        super().__init__(optimizer, lr_lambda, last_epoch)

    def set_curr_size(self, curr_size: int):
        self.curr_size = curr_size

    def state_dict(self):
        state = super().state_dict()
        state["base_size"] = self.base_size
        return state

    def load_state_dict(self, state_dict):
        if state_dict is None:
            return

        base_size = state_dict.pop("base_size")
        super().load_state_dict(state_dict)
        self.base_size = base_size
        self.curr_size = base_size

    def get_lr(self) -> list[float]:
        scale = max(1.0, float(self.curr_size / self.base_size))
        return [
            base_lr * scale * lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)
        ]


class CallBack1:
    def on_validation_end(self, loss):
        return (loss * 2).item()


class CallBack2:
    def on_validation_end(self, loss):
        return (-loss).item()


class BayesNJTrainer:
    def __init__(
        self,
        # Model Architecture parameters
        evopf,
        embed_dim,
        pair_dim,
        n_blocks,
        n_heads,
        optimize_brlens,
        mixed_atten,
        symmetric,
        dm_MLP,
        # Training parameters
        batch_size,
        epochs,
        warmup,
        learning_rate,
        weight_decay,
        seed,
        project,
        project_root,
        log_every,
        val_every,
        mixed_precision,
        use_deepspeed,
        use_flexattention,
        use_l1_loss,
        use_unambiguous_order,
        # Data Paths
        treedir_t,
        msadir_t,
        treedir_v,
        msadir_v,
        cache_root,
        run_name=None,
        temperature=None,
        brlen_weight=None,
        brlen_annealing=None,
        no_topo_steps=None,
        gradient_clipping_value=None,
        resuming=False,
        base_size=None,
        lgnrl_mu_x_min: Optional[float] = None,
        lgnrl_mu_x_max: Optional[float] = None,
        lgnrl_sigma_x_min: Optional[float] = None,
        lgnrl_sigma_x_max: Optional[float] = None,
        lgnrl_lsigma_div: Optional[float] = None,
    ):
        if base_size is None and len(treedir_t) > 1:
            raise ValueError(
                "If using more than 1 training size then you must specify "
                "the base batch_size for the schedule so that LR can be "
                "rescaled w.r.t. batch size."
            )

        self.evopf = evopf
        self.embed_dim = embed_dim
        self.pair_dim = pair_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.optimize_brlens = optimize_brlens
        self.mixed_atten = mixed_atten
        self.symmetric = symmetric
        self.dm_MLP = dm_MLP
        self.batch_sizes = batch_size
        self.epochs = epochs
        self.warmup = warmup
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.project = project
        self.project_root = project_root
        self.log_every = log_every
        self.val_every = val_every
        self.mixed_precision = mixed_precision
        self.use_deepspeed = use_deepspeed
        self.use_flexattention = use_flexattention
        self.use_l1_loss = use_l1_loss
        self.use_unambiguous_order = use_unambiguous_order
        self.treedirs_t = treedir_t
        self.msadirs_t = msadir_t
        self.treedirs_v = treedir_v
        self.msadirs_v = msadir_v
        self.cache_root = cache_root
        self.run_name = run_name or generate_run_name()
        self.temperature = temperature
        self.brlen_weight = brlen_weight
        self.brlen_annealing = brlen_annealing
        self.no_topo_steps = no_topo_steps or 0
        self.gradient_clipping_value = gradient_clipping_value
        self.resuming = resuming
        self.base_size = base_size
        self.loss_clamp_params = dict(
            lgnrl_mu_x_min=lgnrl_mu_x_min,
            lgnrl_mu_x_max=lgnrl_mu_x_max,
            lgnrl_sigma_x_min=lgnrl_sigma_x_min,
            lgnrl_sigma_x_max=lgnrl_sigma_x_max,
            lgnrl_lsigma_div=lgnrl_lsigma_div,
        )

        self.run_params = dict(
            evopf=self.evopf,
            embed_dim=self.embed_dim,
            pair_dim=self.pair_dim,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            optimize_brlens=self.optimize_brlens,
            mixed_atten=self.mixed_atten,
            symmetric=self.symmetric,
            dm_MLP=self.dm_MLP,
            batch_size=self.batch_sizes,
            epochs=self.epochs,
            warmup=self.warmup,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            seed=self.seed,
            project=self.project,
            project_root=self.project_root,
            log_every=self.log_every,
            val_every=self.val_every,
            mixed_precision=self.mixed_precision,
            use_deepspeed=self.use_deepspeed,
            use_flexattention=self.use_flexattention,
            use_l1_loss=self.use_l1_loss,
            use_unambiguous_order=self.use_unambiguous_order,
            treedir_t=self.treedirs_t,
            msadir_t=self.msadirs_t,
            treedir_v=self.treedirs_v,
            msadir_v=self.msadirs_v,
            cache_root=self.cache_root,
            run_name=self.run_name,
            temperature=self.temperature,
            brlen_weight=self.brlen_weight,
            brlen_annealing=self.brlen_annealing,
            no_topo_steps=self.no_topo_steps,
            gradient_clipping_value=self.gradient_clipping_value,
            base_size=self.base_size,
            **self.loss_clamp_params,
        )

        self.start_epoch = 0
        self.start_step = 0

        self.checkpoint_to_load = None
        self.resume_mid_checkpoint = False

        # Seed RNGs
        L.seed_everything(self.seed)

        # Setup training disk locations
        self._init_dirs()

        # Get user info for checkpoint saving
        self._init_user_info()

        # Setup data loaders
        # self._init_dataloaders()

        # Setup model and optimizer
        self._init_model_and_opt()

        # Create fabric object
        self.fabric = L.Fabric(
            loggers=self._init_loggers(),
            callbacks=self._init_callbacks(),
            plugins=self._get_plugins(),
            precision=self._get_precision(),
            **self._get_fabric_args(),
        )

        self.tempdir = self.get_tempdir()

        self.l1_loss = torch.nn.L1Loss() if self.use_l1_loss else None

    def _init_dirs(self):
        self.projdir = (
            self.project
            if self.project_root is None
            else os.path.join(self.project_root, self.project)
        )
        self.rundir = os.path.join(self.projdir, self.run_name)
        self.ckptdir = os.path.join(self.rundir, "checkpoints")
        self.logdir = os.path.join(self.rundir, "logs")

    def _get_pf_envvars(self):
        env = dict()
        for k, v in os.environ.items():
            if k.upper().startswith("PF"):
                env[k] = v
        return env

    def _init_loggers(self):
        # Init loggers
        csvlogger = CSVLogger(save_dir=self.logdir, version=None)
        wandblogger = WandbLogger(
            name=self.run_name,
            save_dir=self.logdir,
            offline=True,
            project=self.project,
            group=self.run_name,
            config={**self.run_params, "envvar": self._get_pf_envvars()},
        )

        return [csvlogger, wandblogger]

    def _init_callbacks(self):
        # TODO
        return []

    def _init_user_info(self):
        self.gid = os.getgid()
        self.uid = os.getuid()
        self.uname = getpass.getuser()
        self.gname = grp.getgrgid(self.gid).gr_name

    def _get_precision(self):
        return "16-mixed" if self.mixed_precision and not self.use_deepspeed else None

    def _get_plugins(self):
        if self.use_deepspeed:
            return [
                L.fabric.plugins.precision.DeepSpeedPrecision(
                    "16-mixed" if self.mixed_precision else "32-true"
                )
            ]
        else:
            return None

    def _get_fabric_args(self) -> dict[str, Any]:
        if platform == "darwin":
            return dict()
        else:
            import idr_torch  # Only available on Jean-Zay # type: ignore

            if idr_torch.size > 1:  # type: ignore
                return dict(
                    devices=idr_torch.size,
                    strategy=(
                        L.fabric.strategies.DeepSpeedStrategy(
                            stage=0, precision=self._get_precision()
                        )
                        if self.use_deepspeed
                        else "ddp"
                    ),
                    accelerator="gpu",
                    num_nodes=idr_torch.num_nodes,
                )
            else:
                return dict()

    def get_tempdir(self) -> str:
        if platform == "darwin":
            return "/tmp"
        else:
            return os.getenv("JOBSCRATCH", "/tmp")

    def _get_dataset(self, trees, msas, cache):
        if self.use_unambiguous_order:
            return UnambiguousMergeOrderDataset(trees, msas)
        return NJDataset(
            trees,
            msas,
            cache,
            get_all_data=self.optimize_brlens or self.use_l1_loss,
        )

    def get_training_sampler(self, dataset, batch_size, seed):
        if platform == "darwin":
            return CheckpointableRandomSampler(dataset, batch_size, seed)
        else:
            import idr_torch  # Only available on Jean-Zay # type: ignore

            if idr_torch.size > 1:  # type: ignore
                return CheckpointableDsitributedSampler(
                    dataset,
                    batch_size,
                    seed,
                    shuffle=True,
                    drop_last=True,
                    frac_subsample=0.1,
                )
            else:
                return CheckpointableRandomSampler(dataset, batch_size, seed)

    def _init_dataloaders(self):
        if len(self.treedirs_t) > 1:
            self.loader_t = DataLoader(
                (
                    ds := MultisizeUnambiguousMergeOrderDataset(
                        self.treedirs_t, self.msadirs_t
                    )
                ),
                num_workers=4 if platform != "darwin" else 0,
                pin_memory=True,
                batch_sampler=DistributedSameSizeSampler(
                    ds,
                    base_size=50,
                    base_batch_size=self.base_size,
                    shuffle=True,
                    seed=self.seed,
                ),
            )
            self.loader_v = DataLoader(
                (
                    ds := MultisizeUnambiguousMergeOrderDataset(
                        self.treedirs_v, self.msadirs_v
                    )
                ),
                num_workers=4 if platform != "darwin" else 0,
                pin_memory=True,
                batch_sampler=DistributedSameSizeSampler(
                    ds,
                    base_size=50,
                    base_batch_size=(self.base_size or 1) * 2,
                    shuffle=False,
                    seed=self.seed,
                ),
            )
        else:
            self.loader_t = DataLoader(
                (
                    ds := self._get_dataset(
                        self.treedirs_t[0],
                        self.msadirs_t[0],
                        f"{self.cache_root}/train",
                    )
                ),
                batch_size=self.batch_sizes[0],
                sampler=self.get_training_sampler(
                    dataset=ds,
                    batch_size=self.batch_sizes[0],
                    seed=self.seed,
                ),
                shuffle=True,
                num_workers=4 if platform != "darwin" else 0,
                pin_memory=True,
            )
            self.loader_v = DataLoader(
                self._get_dataset(
                    self.treedirs_v[0], self.msadirs_v[0], f"{self.cache_root}/val"
                ),
                batch_size=self.batch_sizes[0],
                shuffle=True,
                num_workers=4 if platform != "darwin" else 0,
                pin_memory=True,
            )

    def _init_model_and_opt(self):
        if self.evopf:
            self.model = EvoPF(
                n_blocks=self.n_blocks,
                n_heads=self.n_heads,
                h_dim=self.embed_dim,
                pair_dim=self.pair_dim,
                use_opm=True,
                distance_mlp=self.dm_MLP,
                symmetric=self.symmetric,
                use_deepspeed=self.use_deepspeed,
                use_flexattention=self.use_flexattention,
                use_bilinear_embedder=self.use_unambiguous_order,
                use_brlens=self.optimize_brlens,
            )
        elif self.mixed_atten:
            self.model = PhyloformerSeqMixed(
                n_blocks=self.n_blocks,
                n_heads=self.n_heads,
                h_dim=self.embed_dim,
            )
        else:
            self.model = PhyloformerSeq(
                n_blocks=self.n_blocks,
                n_heads=self.n_heads,
                h_dim=self.embed_dim,
            )

        self.opt = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _load_model_weights(self, weights):
        print(self.model)
        self.model.load_state_dict(weights)

    def make_tar_info(self, filename, filesize):
        info = tarfile.TarInfo(name=filename)
        info.size = filesize
        info.mtime = time()
        info.gid = self.gid
        info.uid = self.uid
        info.uname = self.uname
        info.gname = self.gname

        return info

    def save_checkpoint(self, model, opt, sched, val_loss, end_of_epoch=False):
        """
        Save a complete checkpoint of the trainer
        """
        if self.fabric.is_global_zero:  # Only save on main device
            state = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": self.epoch,
                "step": self.global_step,
                "epoch_step": self.epoch_step if not end_of_epoch else 0,
                "val_loss": val_loss.item(),
                "hparams": self.run_params,
            }

            s = "_epoch" if end_of_epoch else ""

            ckptname = f"epoch-{self.epoch}-step-{self.global_step}-vloss-{val_loss.item():.4e}{s}.ckpt"

            if not SAVE_CHECKPOINT_SPACE:
                # Create file info for tar archive
                bytebuffer = io.BytesIO()
                torch.save(state, bytebuffer)
                info = self.make_tar_info(ckptname, bytebuffer.tell())
                bytebuffer.seek(0)  # Make bytebuffer cursor is at start
                with tarfile.open(
                    os.path.join(self.ckptdir, "checkpoints.tar"), "a"
                ) as tar:
                    tar.addfile(info, bytebuffer)

            # Kepp last checkpoint
            torch.save(state, os.path.join(self.ckptdir, "latest.ckpt"))

            # Keep last epoch checkpoint
            if end_of_epoch:
                torch.save(state, os.path.join(self.ckptdir, "last_epoch.ckpt"))

            # Keep current best checkpoint
            if val_loss < self.best_val_loss:
                torch.save(state, os.path.join(self.ckptdir, "best_val_loss.ckpt"))
                self.best_val_loss = val_loss

        self.fabric.barrier()

    def set_checkpoint_to_load(self, ckpt_obj, end_of_epoch_checkpoint: bool = True):
        self.checkpoint_to_load = ckpt_obj
        if end_of_epoch_checkpoint:
            self.checkpoint_to_load["epoch"] += 1
        else:
            # if len(ckpt_obj["hparams"]["msadir_t"]) > 1:
            #     raise ValueError("Cannot resume mid-checkpoint with several datasets")
            self.resume_mid_checkpoint = True

    def get_evopf_pred(self, evopf, msa, return_tril: bool):
        B, _, _, n = msa.size()
        i, j = torch.tril_indices(n, n, -1)

        msa_emb = None
        if self.use_unambiguous_order:
            dm, msa_emb = evopf(msa.float())
        else:
            dm = evopf(msa.float())

        if (return_tril and self.symmetric) or (
            not return_tril and not self.symmetric
        ):  # all good
            ret_dm = dm
        elif return_tril and not self.symmetric:
            ret_dm = dm[:, i, j]
        else:
            dm_sq = torch.zeros((B, n, n)).to(dm)
            dm_sq[:, i, j] = dm
            ret_dm = dm_sq + dm_sq.transpose(-1, -2)

        if self.use_unambiguous_order:
            return ret_dm, msa_emb
        else:
            return ret_dm

    def get_pfseq_pred(self, pfseq, msa, return_tril: bool):
        B, _, _, n = msa.size()
        i, j = torch.tril_indices(n, n, -1)

        dm = pfseq(msa.float())

        if return_tril:  # all good
            return dm
        else:
            dm_sq = torch.zeros((B, n, n)).to(dm)
            dm_sq[:, i, j] = dm
            return dm_sq + dm_sq.transpose(-1, -2)

    def get_batch_L1_loss(self, model, batch):
        assert self.use_l1_loss and self.l1_loss is not None

        msa, splits, brlens, _, _, _ = batch
        n = msa.size(-1)

        if self.evopf:
            dm_vec = self.get_evopf_pred(model, msa, True)
        else:
            dm_vec = self.get_pfseq_pred(model, msa, True)

        # get DM from splits and branch lengths
        s2p = seq2pairs(n, lower=True, diff=True).to(self.fabric.device)
        dm_target = ((splits @ s2p.T).abs() * brlens.unsqueeze(-1)).sum(-2)

        loss = self.l1_loss(dm_vec, dm_target)

        # We don't need gradients for these (just for logging)
        mse = torch.nn.functional.mse_loss(dm_vec.detach(), dm_target.detach())
        mre = ((dm_vec.detach() - dm_target.detach()).abs() / dm_target.detach()).mean()

        # clean up
        del msa, brlens, splits

        return loss, mse, mre

    def get_batch_NJ_loss(self, model, batch):
        """
        Run the forward pass of the model and compute the logprobabilty of
        executing the merges of the true tree given the inferred distance matrix
        """

        if self.optimize_brlens:
            msa, _, brlens, merge_order, _, _ = batch
        else:
            msa, merge_order, _, _ = batch
            brlens = None
        batch_size, _, _, n_seqs = msa.shape

        if self.evopf:
            dm = self.get_evopf_pred(model, msa, False)
        else:
            dm = self.get_pfseq_pred(model, msa, False)

        # compute loss
        logprob_topo, neg_logprob_brlens = batch_tree_probability_with_merges_branches(
            dm, merge_order, self.temperature
        )
        neg_logprob_topo = -logprob_topo

        # if self.optimize_brlens:
        #     assert (
        #         brlens is not None
        #     ), "Should not be attempting to optimize branch lengths without loading the true ones"
        #
        #     # Estimating joint branch length probability with Fitch & Margoliash variance term
        #     br_sqdiff = (brlens - pred_brlens).pow(2)
        #     fm_var = 2 * brlens.pow(2)
        #     neg_logprob_brlens = (br_sqdiff / fm_var).sum(dim=-1)
        #
        if self.brlen_weight is not None:
            neg_logprob_brlens = neg_logprob_brlens * self.brlen_weight
        # else:
        #     neg_logprob_brlens = torch.zeros_like(neg_logprob_topo)

        # Normalize by number of sequences for multi-size training
        neg_logprob_tree = (neg_logprob_topo + neg_logprob_brlens) / n_seqs

        # clean up
        del msa, brlens, merge_order

        return (
            neg_logprob_tree.mean(),
            neg_logprob_topo.mean(),
            neg_logprob_brlens.mean(),
        )

    def get_batch_unambiguous_order_loss(self, model, batch, verbose=True):
        """
        Run the forward pass of the model and compute the logprobabilty of
        executing the merges of the true tree given the inferred distance matrix
        """

        msa, merge_order, brlens, _ = batch
        batch_size, _, _, n_seqs = msa.shape

        dm, msa_emb = self.get_evopf_pred(model, msa, False)

        # compute loss
        logprob_topo, logprob_gamma, logprob_beta, logprob_brlens, mse_brlens = (
            batch_compute_tree_logprob(
                model,
                msa_emb,  # type: ignore
                dm,
                brlens.to(dm),
                merge_order.to(dm),
                self.temperature,
                topo_only=not self.optimize_brlens,
                ignore_topo=self.global_step < self.no_topo_steps,
                verbose=verbose,
            )
        )

        # Normalize by number of sequences for multi-size training
        neg_logprob_topo = -logprob_topo / n_seqs
        # neg_logprob_brlens = -(logprob_gamma + logprob_beta) / n_seqs
        neg_logprob_brlens = -logprob_brlens / n_seqs
        neg_mse_brlens = -mse_brlens / n_seqs

        # Anneal branch length loss
        if self.brlen_annealing is not None and self.beta_i is not None:
            neg_logprob_tree = neg_logprob_topo + self.beta_i * neg_logprob_brlens
        else:
            neg_logprob_tree = neg_logprob_topo + neg_logprob_brlens

        # clean up
        del msa, brlens, merge_order

        return (
            neg_logprob_tree.mean(),
            neg_logprob_topo.mean(),
            neg_logprob_brlens.mean(),
            (-logprob_gamma / n_seqs).mean(),
            (-logprob_beta / n_seqs).mean(),
            neg_mse_brlens.mean(),
        )

    def log_dict(self, metrics):
        self.fabric.log_dict(
            {"epoch": self.epoch, "global_step": self.global_step, **metrics}
        )

    def train_epoch(self, model, loaders_t, loaders_v, opt, sched):
        # Train loop
        model.train()

        print(model)

        # Init trackers
        accum = torch.zeros(1, device=model.device)
        accum_topo = torch.zeros(1, device=model.device)
        accum_brlen = torch.zeros(1, device=model.device)
        accum_gamma = torch.zeros(1, device=model.device)
        accum_beta = torch.zeros(1, device=model.device)
        accum_mse = torch.zeros(1, device=model.device)
        counter = 0

        # Generate random dataloader order
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed + self.epoch)

        # Compute brlen annealing schedule
        H = self.brlen_annealing or 1.0

        for lidx in torch.randperm(len(loaders_t), generator=rng):
            loader_t = loaders_t[lidx]

            # Distributed sampler reproducibility
            if hasattr(loader_t.sampler, "set_epoch"):
                loader_t.sampler.set_epoch(self.epoch)
            if hasattr(loader_t.sampler, "set_starting_step"):
                loader_t.sampler.set_starting_step(self.epoch_step)

            for batch in tqdm(loader_t, leave=False, desc="TRAIN"):
                # Forward pass
                opt.zero_grad()

                # annealing term if needed (as in ARTree) if needed shifted by nb of steps we ignore topology for
                # self.beta_i = min(1.0, 1e-10 + max((self.global_step - self.no_topo_steps) / H, 0))
                # self.beta_i = min(1.0, 1e-20 + self.global_step / H)
                if self.brlen_annealing is not None:
                    self.beta_i = ANNEALING_FUNC(self.global_step, ANNEALING_START, H)
                else:
                    self.beta_i = 1.0

                # # Disable annealing if ignoring topology
                # if self.global_step < self.no_topo_steps:
                #     self.beta_i = 1.0

                if self.use_l1_loss:
                    # MAE, MSE,     MRE
                    loss, topoloss, brlenloss = self.get_batch_L1_loss(model, batch)
                else:
                    if self.use_unambiguous_order:
                        loss, topoloss, brlenloss, gammaloss, betaloss, mseloss = (
                            self.get_batch_unambiguous_order_loss(model, batch)
                        )

                    else:
                        loss, topoloss, brlenloss = self.get_batch_NJ_loss(model, batch)

                # Backward pass
                self.fabric.backward(loss)

                if self.gradient_clipping_value is not None:
                    self.fabric.clip_gradients(model, opt, self.gradient_clipping_value)

                opt.step()
                sched.set_curr_size(batch[0].size(0))
                sched.step()

                # Accumulate train loss
                accum += loss.detach()
                accum_topo += topoloss.detach()
                accum_brlen += brlenloss.detach()
                accum_mse += mseloss.detach()
                if self.use_unambiguous_order:
                    accum_gamma += gammaloss.detach()
                    accum_beta += betaloss.detach()
                counter += 1

                # Clean up
                del batch
                del loss, topoloss, brlenloss, mseloss
                if self.use_unambiguous_order:
                    del betaloss, gammaloss

                # Log if needed
                if self.global_step % self.log_every == 0 and self.global_step > 0:
                    avg_loss = (accum / counter).item()
                    lr = opt.param_groups[0]["lr"]

                    if self.use_l1_loss:
                        logdict = {
                            "mae_train": avg_loss,
                            "mse_train": (accum_topo / counter).item(),
                            "mre_train": (accum_brlen / counter).item(),
                            "lr": lr,
                        }
                    else:
                        logdict = {"neglogprob_train": avg_loss, "lr": lr}

                        if self.optimize_brlens:
                            # Log topological and Branch length components separately
                            add_dict = dict(
                                neglogprob_topo_train=(accum_topo / counter).item(),
                                neglogprob_brlen_train=(accum_brlen / counter).item(),
                                neglogprob_gamma_train=(accum_gamma / counter).item(),
                                neglogprob_beta_train=(accum_beta / counter).item(),
                                mse_train=(accum_mse / counter).item(),
                                beta_i=self.beta_i,
                            )
                            logdict = {**logdict, **add_dict}
                        elif self.use_unambiguous_order:
                            # Log topological and Branch length components separately
                            add_dict = dict(
                                neglogprob_topo_train=(accum_topo / counter).item(),
                                neglogprob_brlen_train=(accum_brlen / counter).item(),
                                neglogprob_gamma_train=(accum_gamma / counter).item(),
                                neglogprob_beta_train=(accum_beta / counter).item(),
                                mse_train=(accum_mse / counter).item(),
                                beta_i=self.beta_i,
                            )
                            logdict = {**logdict, **add_dict}
                    self.log_dict({**logdict, "dataloader": lidx})

                    # Reset trackers to 0
                    accum = torch.zeros_like(accum)
                    accum_topo = torch.zeros_like(accum_topo)
                    accum_brlen = torch.zeros_like(accum_brlen)
                    accum_mse = torch.zeros_like(accum_mse)
                    if self.use_unambiguous_order:
                        accum_gamma = torch.zeros_like(accum_gamma)
                        accum_beta = torch.zeros_like(accum_beta)
                    counter = 0

                if (
                    self.val_every is not None
                    and self.global_step % self.val_every == 0
                    and self.global_step > 0
                ):
                    self.validate_epoch(
                        model, loaders_v, opt, sched, end_of_epoch=False
                    )

                # Increment counter
                self.global_step += 1
                self.epoch_step += 1

    def validate_epoch(self, model, loaders, opt, sched, end_of_epoch=False):
        model.eval()

        # Init accumulators
        accum = torch.zeros(1, device=model.device)
        n_batches = 0

        H = self.brlen_annealing or 1.0

        with torch.no_grad():
            for lidx, loader in enumerate(loaders):
                for batch in tqdm(loader, leave=False, desc="VAL  "):
                    if self.use_l1_loss:
                        # MAE, MSE,     MRE
                        loss, topoloss, brlenloss = self.get_batch_L1_loss(model, batch)
                    elif self.use_unambiguous_order:
                        loss, topoloss, brlenloss, gammaloss, betaloss, mseloss = (
                            self.get_batch_unambiguous_order_loss(
                                model, batch, verbose=False
                            )
                        )
                    else:
                        loss, topoloss, brlenloss = self.get_batch_NJ_loss(model, batch)

                    if self.use_l1_loss:
                        logdict = {
                            "mae_val": loss.item(),
                            "mse_val": topoloss.item(),
                            "mre_val": brlenloss.item(),
                        }
                    else:
                        logdict = {"neglogprob_val": loss.item()}
                        if self.optimize_brlens:
                            # Log topological and Branch length components separately
                            add_dict = dict(
                                neglogprob_topo_val=topoloss.item(),
                                neglogprob_brlen_val=brlenloss.item(),
                                neglogprob_gamma_val=gammaloss.item(),
                                neglogprob_beta_val=betaloss.item(),
                                mse_val=mseloss.item(),
                            )
                            logdict = {**logdict, **add_dict}
                        elif self.use_unambiguous_order:
                            # Log topological and Branch length components separately
                            add_dict = dict(
                                neglogprob_topo_val=topoloss.item(),
                                neglogprob_brlen_val=brlenloss.item(),
                                neglogprob_gamma_val=gammaloss.item(),
                                neglogprob_beta_val=betaloss.item(),
                                mseloss_val=mseloss.item(),
                            )
                            logdict = {**logdict, **add_dict}

                    self.log_dict({**logdict, "dataloader": lidx})

                    accum += loss.detach()
                    n_batches += 1

        distrib_val_loss = accum / n_batches

        avg_val_loss: torch.Tensor = self.fabric.all_reduce(  # type: ignore
            distrib_val_loss,
            reduce_op="mean",
        )

        self.save_checkpoint(model, opt, sched, avg_val_loss, end_of_epoch)

        return avg_val_loss

    def fit(self):
        # Init fabric stuff
        self.fabric.launch()

        # We need to do this after having launched fabric since it can rely on the distributed framework being in place already
        self._init_dataloaders()

        loaders_t = [
            self.fabric.setup_dataloaders(self.loader_t, use_distributed_sampler=False)
        ]
        loaders_v = [
            self.fabric.setup_dataloaders(self.loader_v, use_distributed_sampler=False)
        ]

        # Setup scheduler
        total_steps = self.epochs * sum(len(ld) for ld in loaders_t)
        n_warmup = int(w) if (w := self.warmup) > 1 else int(total_steps * w)
        sched = MultisizeLRScheduler(
            self.opt,
            num_warmup_steps=n_warmup,
            num_training_steps=total_steps,
            base_size=self.base_size,
        )

        model, opt = self.fabric.setup(self.model, self.opt)

        # # Mark additional forward methods
        model.mark_forward_method("embed_parent_node")
        # model.mark_forward_method("predict_lognormal_params")
        # model.mark_forward_method("predict_beta_params")

        # Setup tracking variables
        self.epoch = self.start_epoch
        self.global_step = self.start_step
        self.epoch_step = 0
        self.best_val_loss = torch.tensor([torch.inf], device=model.device)

        # Make sure output directories exist
        if self.fabric.is_global_zero:
            os.makedirs(self.ckptdir, exist_ok=self.resuming)
            os.makedirs(self.logdir, exist_ok=self.resuming)

        # Checkpoint loading if needed
        if self.checkpoint_to_load is not None:
            self.start_epoch = self.checkpoint_to_load["epoch"]
            self.global_step = self.checkpoint_to_load["step"] + 1
            self.epoch_step = self.checkpoint_to_load.get("epoch_step", 0)
            self.best_val_loss = torch.tensor(
                [self.checkpoint_to_load["val_loss"]], device=model.device
            )
            model.load_state_dict(self.checkpoint_to_load["model"])
            opt.load_state_dict(self.checkpoint_to_load["optimizer"])
            sched.load_state_dict(self.checkpoint_to_load["scheduler"])

        # Run hyper-parameters
        if self.fabric.global_rank == 0:
            for logger in self.fabric.loggers:
                logger.log_hyperparams(self.run_params)
                if isinstance(logger, WandbLogger):
                    logger.watch(model)  # Track model gradients

        print("Running with following settings:")
        pprint(self.run_params)

        try:
            for epoch in trange(self.start_epoch, self.epochs):
                self.epoch = epoch

                # Run Training loop on epoch
                self.train_epoch(model, loaders_t, loaders_v, opt, sched)
                self.epoch_step = 0  # reset step at end of epoch

                # Run end of epoch validation
                self.validate_epoch(model, loaders_v, opt, sched, end_of_epoch=True)

        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM at epoch {self.epoch}.")
            print(f"\t loader with batch_size: {sched.curr_size}")
            raise e


def main():
    # Parse CLI arguments
    import argparse

    parser = argparse.ArgumentParser("Train a BayesNJ instance")
    subparsers = parser.add_subparsers(dest="command")

    # Commands
    trainer = subparsers.add_parser(name="train", description="Launch a training run")
    tuner = subparsers.add_parser(name="finetune", description="Fine tune weights")
    tuner.add_argument(
        "checkpoint", help="checkpoint file containing architecture and weights"
    )

    ## Model architecture (Only from scratch training)
    trainer.add_argument("-V", "--evopf", action="store_true")
    trainer.add_argument("-e", "--embed-dim", type=int, default=64)
    trainer.add_argument("-P", "--pair-dim", type=int, default=256)
    trainer.add_argument("-b", "--n-blocks", type=int, default=6)
    trainer.add_argument("-H", "--n-heads", type=int, default=4)
    trainer.add_argument("-M", "--mix-attention", action="store_true")
    trainer.add_argument("-y", "--symmetric", action="store_true")
    trainer.add_argument("-d", "--dm-MLP", action="store_true")

    ## From sratch and fine-tuning arguments
    for cmd in [trainer, tuner]:
        ## Training parameters
        cmd.add_argument("-E", "--epochs", type=int, default=1_000)
        cmd.add_argument("-w", "--warmup", type=float, default=0.1)
        cmd.add_argument("-r", "--learning-rate", type=float, default=1e-4)
        cmd.add_argument("-D", "--weight-decay", type=float, default=0)
        cmd.add_argument("-S", "--seed", type=int, default=1337)
        cmd.add_argument("-p", "--project", type=str, default=None)
        cmd.add_argument("-R", "--project-root", type=str, default=None)
        cmd.add_argument("-l", "--log-every", type=int, default=50)
        cmd.add_argument("-v", "--validate-every", type=int, default=None)
        cmd.add_argument("-x", "--mixed-precision", action="store_true")
        cmd.add_argument("--deepspeed", action="store_true")
        cmd.add_argument("--flexattention", action="store_true")
        cmd.add_argument("-L", "--l1-loss", action="store_true")
        cmd.add_argument("-u", "--unambiguous-order", action="store_true")
        cmd.add_argument("-m", "--temperature", type=float, default=None)
        cmd.add_argument("-B", "--branch-lengths", action="store_true")
        cmd.add_argument("-W", "--branch-lengths-weight", type=float, default=None)
        cmd.add_argument("-N", "--branch-length-annealing", type=int, default=None)
        cmd.add_argument("-n", "--no-topo-steps", type=int, default=None)
        cmd.add_argument("-C", "--clip-gradients", type=float, default=None)

        ## Data paths
        cmd.add_argument("--base-batch-size", type=int, required=False)
        cmd.add_argument("-s", "--batch-size", nargs="+", type=int, default=1)
        cmd.add_argument("-t", "--train-trees", nargs="+", type=str, required=True)
        cmd.add_argument("-T", "--val-trees", nargs="+", type=str, required=True)
        cmd.add_argument("-a", "--train-alns", nargs="+", type=str, required=True)
        cmd.add_argument("-A", "--val-alns", nargs="+", type=str, required=True)
        cmd.add_argument("-c", "--cache-root", type=str, required=True)

        ## Loss clamps
        cmd.add_argument("--lgnrl_mu_x_min", type=float)
        cmd.add_argument("--lgnrl_mu_x_max", type=float)
        cmd.add_argument("--lgnrl_sigma_x_min", type=float)
        cmd.add_argument("--lgnrl_sigma_x_max", type=float)
        cmd.add_argument("--lgnrl_lsigma_div", type=float)

    # Resume command
    resumer = subparsers.add_parser(name="resume", description="Resume a training run")
    resumer.add_argument("checkpoint")

    args = parser.parse_args()

    __import__("pprint").pprint(args)

    if args.command == "train":
        trainer = BayesNJTrainer(
            evopf=args.evopf,
            embed_dim=args.embed_dim,
            pair_dim=args.pair_dim,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            optimize_brlens=args.branch_lengths,
            mixed_atten=args.mix_attention,
            symmetric=args.symmetric,
            dm_MLP=args.dm_MLP,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup=args.warmup,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            project=args.project,
            project_root=args.project_root,
            log_every=args.log_every,
            val_every=args.validate_every,
            mixed_precision=args.mixed_precision,
            use_deepspeed=args.deepspeed,
            use_flexattention=args.flexattention,
            use_l1_loss=args.l1_loss,
            use_unambiguous_order=args.unambiguous_order,
            treedir_t=args.train_trees,
            msadir_t=args.train_alns,
            treedir_v=args.val_trees,
            msadir_v=args.val_alns,
            cache_root=args.cache_root,
            temperature=args.temperature,
            brlen_weight=args.branch_lengths_weight,
            brlen_annealing=args.branch_length_annealing,
            no_topo_steps=args.no_topo_steps,
            gradient_clipping_value=args.clip_gradients,
            base_size=args.base_batch_size,
            lgnrl_mu_x_min=args.lgnrl_mu_x_min,
            lgnrl_mu_x_max=args.lgnrl_mu_x_max,
            lgnrl_sigma_x_min=args.lgnrl_sigma_x_min,
            lgnrl_sigma_x_max=args.lgnrl_sigma_x_max,
            lgnrl_lsigma_div=args.lgnrl_lsigma_div,
        )
    elif args.command == "finetune":
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        prms = ckpt["hparams"]

        trainer = BayesNJTrainer(
            evopf=prms["evopf"],
            embed_dim=prms["embed_dim"],
            pair_dim=prms["pair_dim"],
            n_heads=prms["n_heads"],
            n_blocks=prms["n_blocks"],
            optimize_brlens=prms["optimize_brlens"],
            mixed_atten=prms["mixed_atten"],
            symmetric=prms["symmetric"],
            dm_MLP=prms["dm_MLP"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup=args.warmup,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            project=args.project,
            project_root=args.project_root,
            log_every=args.log_every,
            val_every=args.validate_every,
            mixed_precision=args.mixed_precision,
            use_flexattention=prms["use_flexattention"],
            use_deepspeed=prms["use_deepspeed"],
            use_l1_loss=args.l1_loss,
            use_unambiguous_order=prms["use_unambiguous_order"],
            treedir_t=args.train_trees,
            msadir_t=args.train_alns,
            treedir_v=args.val_trees,
            msadir_v=args.val_alns,
            cache_root=args.cache_root,
            temperature=args.temperature,
            brlen_weight=args.branch_lengths_weight,
            brlen_annealing=args.branch_length_annealing,
            no_topo_steps=args.no_topo_steps,
            gradient_clipping_value=args.clip_gradients,
            base_size=args.base_batch_size,
            lgnrl_mu_x_min=args.lgnrl_mu_x_min,
            lgnrl_mu_x_max=args.lgnrl_mu_x_max,
            lgnrl_sigma_x_min=args.lgnrl_sigma_x_min,
            lgnrl_sigma_x_max=args.lgnrl_sigma_x_max,
            lgnrl_lsigma_div=args.lgnrl_lsigma_div,
        )
        trainer._load_model_weights(ckpt["model"])

    elif args.command == "resume":
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        trainer = BayesNJTrainer(**ckpt["hparams"], resuming=True)
        trainer.set_checkpoint_to_load(
            ckpt, end_of_epoch_checkpoint="_epoch.ckpt" in args.checkpoint
        )
    else:
        raise ValueError(f"Unrecognized command: {args.command}")

    # Run training loop
    trainer.fit()


if __name__ == "__main__":
    main()
