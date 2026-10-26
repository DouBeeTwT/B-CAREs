"""YAML configuration handling shared by train.py / report.py / predict.py / demo.py.

Design
------
A run is described by a YAML file (see ``configs/``).  Command-line flags always
win over the YAML values, and every historical flag keeps working on its own, so
that

    python train.py --root ./Data --batch_size 32

and

    python train.py --config configs/train_multitask.yaml --batch_size 8

are both valid.  ``--config`` is deliberately *not* a plain ``store_true`` flag:
it is read in a first pass, converted into ``--flag value`` pairs, prepended to
``sys.argv`` and then parsed normally by the script's own parser.

Errors are raised as :class:`ConfigError` with the offending key, so a typo in a
YAML file fails loudly instead of being silently ignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

__all__ = [
    "ConfigError",
    "load_config",
    "flatten",
    "config_to_argv",
    "parse_with_config",
    "resolve_device",
    "str2bool",
    "TRAIN_KEYMAP",
    "INFER_KEYMAP",
    "PREDICT_KEYMAP",
]

PathLike = Union[str, "Path"]


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or has unknown keys."""


# --------------------------------------------------------------------------- #
# key maps: "<yaml.dotted.key>" -> argparse dest
# --------------------------------------------------------------------------- #
TRAIN_KEYMAP: Dict[str, str] = {
    "data.root": "root",
    "data.batch_size": "batch_size",
    "data.image_size": "figure_size",
    "data.num_workers": "num_workers",
    "model.backbone_name": "backbone_name",
    "model.protonet_name": "protonet_name",
    "model.class_num": "class_num",
    "model.in_channels": "in_channels",
    "model.hidden_channels": "hidden_channels",
    "model.neck_layers": "neck_layers",
    "model.need_P2": "need_P2",
    "model.need_P6": "need_P6",
    "model.need_P7": "need_P7",
    "model.mi_size": "mi_size",
    "model.stride": "stride",
    "model.use_sample": "use_sample",
    "model.sample_radius": "sample_radius",
    "model.base_num": "base_num",
    "model.blender_size": "blender_size",
    "model.attention_size": "attention_size",
    "loss.task_weights": "loss_weight",
    "loss.alpha_cls": "alpha_cls",
    "loss.gamma_cls": "gamma_cls",
    "loss.alpha_bag": "alpha_bag",
    "loss.gamma_bag": "gamma_bag",
    "loss.iou_name": "iou_name",
    "loss.score_threshold": "score_threshold",
    "loss.nms_threshold": "nms_threshold",
    "loss.max_object_num": "max_object_num",
    "loss.topn": "topn",
    "optim.learning_rate": "learning_rate",
    "optim.weight_decay": "weight_decay",
    "optim.epoch_max": "epoch_max",
    "optim.epoch_qp": "epoch_qp",
    "optim.seed": "seed",
    "runtime.device": "device",
    "runtime.world_size": "world_size",
    "runtime.debug_anomaly": "debug_anomaly",
    "logging.wandb": "wandb",
    "logging.project_name": "project_name",
    "logging.exp_name": "exp_name",
}

INFER_KEYMAP: Dict[str, str] = {
    "input.image": "image",
    "weights.birads": "birads_weights",
    "weights.subtype": "subtype_weights",
    "weights.lymph": "lymph_weights",
    "device": "device",
    "output_dir": "output_dir",
    "subtype_score": "subtype_score",
    "clinical.age": "age",
    "clinical.gender": "gender",
    "clinical.menopause": "menopause",
    "clinical.family_history": "family",
    "clinical.smoke": "smoke",
    "clinical.drink": "drink",
    "llm.enabled": "llm",
    "llm.base_url": "llm_base_url",
    "llm.model": "llm_model",
    "llm.api_key_env": "llm_api_key_env",
    "llm.language": "language",
    "explain.enabled": "explain",
    "explain.shap_max_evals": "shap_max_evals",
}

PREDICT_KEYMAP: Dict[str, str] = {
    "input.dir": "image_dir",
    "input.labels_csv": "labels_csv",
    "input.limit": "limit",
    "weights.birads": "birads_weights",
    "weights.subtype": "subtype_weights",
    "weights.lymph": "lymph_weights",
    "device": "device",
    "output_dir": "output_dir",
    "batch_size": "batch_size",
    "subtype_score": "subtype_score",
    "random_weights": "random_weights",
}

# boolean args declared with `action="store_true"` (passed as `--flag`, not `--flag true`)
BOOL_FLAGS = {"wandb", "llm", "explain", "random_weights", "dry_run"}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_config(path: PathLike) -> Dict[str, Any]:
    """Read a YAML configuration file and return it as a plain dict."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"configuration file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - depends on user input
        raise ConfigError(f"could not parse {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a mapping at the top level")
    return data


def flatten(cfg: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten ``{"data": {"root": "x"}}`` into ``{"data.root": "x"}``."""
    flat: Dict[str, Any] = {}
    for key, value in cfg.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, dotted + "."))
        else:
            flat[dotted] = value
    return flat


def config_to_argv(cfg: Mapping[str, Any], keymap: Mapping[str, str]) -> List[str]:
    """Convert a config dict into a list of ``--flag value`` tokens."""
    argv: List[str] = []
    for key, value in flatten(cfg).items():
        if key not in keymap:
            raise ConfigError(
                f"unknown configuration key: {key!r} "
                f"(allowed: {', '.join(sorted(keymap))})"
            )
        dest = keymap[key]
        if value is None:
            continue
        if isinstance(value, bool):
            if dest in BOOL_FLAGS:
                if value:
                    argv.append(f"--{dest}")
            else:
                argv += [f"--{dest}", "true" if value else "false"]
        elif isinstance(value, (list, tuple)):
            if not value:
                continue
            argv.append(f"--{dest}")
            argv += [str(item) for item in value]
        else:
            argv += [f"--{dest}", str(value)]
    return argv


def parse_with_config(
    parser: argparse.ArgumentParser,
    keymap: Mapping[str, str],
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse argv, letting ``--config`` supply defaults that the CLI can override."""
    tokens = list(sys.argv[1:] if argv is None else argv)

    # first pass: only --config (unknown flags ignored)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(tokens)

    if known.config:
        defaults = config_to_argv(load_config(known.config), keymap)
        # argparse keeps the *last* value for a repeated option, so anything the
        # user typed after --config wins over the YAML.
        tokens = defaults + tokens

    return parser.parse_args(tokens)


# --------------------------------------------------------------------------- #
# misc helpers
# --------------------------------------------------------------------------- #
def resolve_device(spec: Optional[str]) -> str:
    """``auto`` -> cuda:0 when available, otherwise cpu."""
    text = "auto" if spec is None else str(spec).strip()
    if text.lower() == "auto":
        import torch  # local import: keeps this module importable without torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return text


def str2bool(value: Union[bool, str]) -> bool:
    """argparse type for real booleans (``--flag true`` / ``--flag false``)."""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")
