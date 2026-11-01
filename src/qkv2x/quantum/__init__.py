"""Quantum feature map, kernels, trainable-map optimisation and QML diagnostics."""
from .featuremap import CrossLayerFeatureMap
from .kernels import (
    bloch_features,
    fidelity_gram,
    fidelity_kernel_blocks,
    projected_gram,
    projected_kernel_blocks,
    add_shot_noise,
    psd_clip,
)
from .train import train_feature_map
from .diagnostics import (
    centered_kta,
    geometric_difference,
    kernel_offdiag_variance,
    effective_dimension,
)
from .provenance import (
    IRRECOVERABLE,
    PROVENANCE_FIELDS,
    capture_run_provenance,
    safe_capture,
    software_fingerprint,
)

__all__ = [
    "CrossLayerFeatureMap",
    "bloch_features",
    "fidelity_gram",
    "fidelity_kernel_blocks",
    "projected_gram",
    "projected_kernel_blocks",
    "add_shot_noise",
    "psd_clip",
    "train_feature_map",
    "centered_kta",
    "geometric_difference",
    "kernel_offdiag_variance",
    "effective_dimension",
    "IRRECOVERABLE",
    "PROVENANCE_FIELDS",
    "capture_run_provenance",
    "safe_capture",
    "software_fingerprint",
]
