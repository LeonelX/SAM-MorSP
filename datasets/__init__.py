from .seg import SAMSegDataset

from core.registry import register_dataset

def reister_all_datasets():
    register_dataset("SAMSegDataset")(SAMSegDataset)
reister_all_datasets()

__all__ = ['SAMSegDataset']
