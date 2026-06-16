"""MO-MGRAP-ISAC: GPU-accelerated multi-objective multitask pre-optimization for
movable-antenna-and-RIS-aided ISAC.

This package implements the spec in ``MO_MGRAP_ISAC_spec.md``. Everything is
written from scratch in PyTorch (device-agnostic: uses CUDA when available,
otherwise CPU). The two orthogonal axes of the design are kept conceptually
separate throughout:

* multitask axis  = propagation regime  (LoS / Rayleigh / Rician)  -> transfer
* multi-objective axis = (comm fairness, sensing fairness)         -> Pareto

Module layout follows Sec 11 of the spec.
"""

from .config import Config, default_config, smoke_config  # noqa: F401

__all__ = ["Config", "default_config", "smoke_config"]
