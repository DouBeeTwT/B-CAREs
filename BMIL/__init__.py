"""B-CAREs model package (model, losses, decoder, dataset, training helpers).

The submodules are imported lazily (PEP 562) on first attribute access:

    import BMIL
    net = BMIL.model.BMIL(args=args)
    crit = BMIL.loss.FCOSLoss(args=args)
    dec = BMIL.utils.decoder.FCOSDecoder(args=args)

Lazy loading matters because the submodules refer back to the package
(``from ..model import backbone`` and similar). Eagerly importing all four from
this file produced

    ImportError: cannot import name 'data' from partially initialized module 'BMIL'

whenever a submodule was imported first. With the lazy form, every import order
works, and ``from BMIL.data.dataset import COCODataset`` also keeps working.
"""

from __future__ import annotations

import importlib
from typing import List

__version__ = "1.1.0"

_SUBMODULES = ("data", "loss", "model", "utils")

__all__: List[str] = list(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module("{}.{}".format(__name__, name))
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(list(globals()) + list(_SUBMODULES)))
