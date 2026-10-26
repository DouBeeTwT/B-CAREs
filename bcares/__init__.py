"""B-CAREs — shared helpers for training, inference and report generation.

The package intentionally contains only glue code (configuration, label maps,
report rendering). The model, losses and decoders live in the `BMIL` package at
the repository root.
"""

__version__ = "1.1.0"
__all__ = ["__version__"]
