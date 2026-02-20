
import lightning
from torch.utils.data import DataLoader  # type:ignore

from phyloformer.data import (
    PhyloDataset,
)

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
