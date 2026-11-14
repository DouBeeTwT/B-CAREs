"""Training utilities: decoder, logging helpers, trainers and plots.

The members are resolved lazily (PEP 562) because ``BMIL/model/fds.py`` imports
``BMIL.utils.lds`` while ``BMIL.utils.trainer`` imports ``BMIL.loss.loss``; eager
imports here produced

    ImportError: cannot import name 'MMDLoss' from partially initialized module
    'BMIL.loss.loss' (most likely due to a circular import)

whenever ``BMIL.model`` was imported first. Lazy access keeps the public API:

    BMIL.utils.decoder.FCOSDecoder(...)   # module
    BMIL.utils.trainer(...)               # function
    BMIL.utils.tester(...)                # function
"""

from __future__ import annotations

import importlib
from typing import Any

#: public name -> (module, attribute); ``None`` means "the module itself"
_MEMBERS = {
    "decoder": ("BMIL.utils.decoder", None),
    "lds": ("BMIL.utils.lds", None),
    "metrics": ("BMIL.utils.metrics", None),
    "ploter": ("BMIL.utils.ploter", "ploter"),
    "tester": ("BMIL.utils.tester", "tester"),
    "trainer": ("BMIL.utils.trainer", "trainer"),
}

__all__ = list(_MEMBERS)


def __getattr__(name: str) -> Any:
    if name not in _MEMBERS:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    module_name, attribute = _MEMBERS[name]
    module = importlib.import_module(module_name)
    return module if attribute is None else getattr(module, attribute)


def __dir__():
    return sorted(set(list(globals()) + list(_MEMBERS)))
