"""Cross-layer feature extraction and qubit-budget reduction."""
from .crosslayer import compute_message_features, FEATURE_SPEC, FEATURE_NAMES, FEATURE_BLOCKS
from .dataset import WindowedDataset, build_dataset
from .reduce import BlockReducer

__all__ = [
    "compute_message_features",
    "FEATURE_SPEC",
    "FEATURE_NAMES",
    "FEATURE_BLOCKS",
    "WindowedDataset",
    "build_dataset",
    "BlockReducer",
]
