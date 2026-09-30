"""LIBERO dataset support for Motus."""

__all__ = ["LiberoMotusDataset"]


def __getattr__(name):
    if name == "LiberoMotusDataset":
        from .libero_dataset import LiberoMotusDataset

        return LiberoMotusDataset
    raise AttributeError(name)
