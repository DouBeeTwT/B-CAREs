"""Configuration files must stay in sync with the argparse flags."""

from __future__ import annotations

from pathlib import Path

import pytest

from bcares.config import (
    INFER_KEYMAP,
    PREDICT_KEYMAP,
    TRAIN_KEYMAP,
    ConfigError,
    config_to_argv,
    flatten,
    load_config,
    str2bool,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


def test_config_directory_is_not_empty():
    assert CONFIGS, "no YAML files found in configs/"


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_config_keys_are_known(path):
    """Every key in every shipped config must map to a real flag."""
    cfg = load_config(path)
    keymap = TRAIN_KEYMAP if "train" in path.name else (
        INFER_KEYMAP if "infer" in path.name else PREDICT_KEYMAP)
    argv = config_to_argv(cfg, keymap)  # raises ConfigError on unknown keys
    assert argv, "config produced no arguments"
    assert all(token.startswith("--") or token for token in argv)


def test_unknown_key_is_reported():
    with pytest.raises(ConfigError, match="unknown configuration key"):
        config_to_argv({"data": {"nope": 1}}, TRAIN_KEYMAP)


def test_flatten():
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}


def test_bool_flags_and_values():
    argv = config_to_argv({"logging": {"wandb": True}}, TRAIN_KEYMAP)
    assert argv == ["--wandb"]
    argv = config_to_argv({"logging": {"wandb": False}}, TRAIN_KEYMAP)
    assert argv == []
    argv = config_to_argv({"model": {"need_P2": False}}, TRAIN_KEYMAP)
    assert argv == ["--need_P2", "false"]


def test_list_values():
    argv = config_to_argv({"loss": {"task_weights": [0.8, 1, 1, 1, 1]}}, TRAIN_KEYMAP)
    assert argv == ["--loss_weight", "0.8", "1", "1", "1", "1"]


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config(ROOT / "configs" / "does-not-exist.yaml")


def test_str2bool():
    assert str2bool("true") is True
    assert str2bool("False") is False
    assert str2bool(True) is True
    with pytest.raises(Exception):
        str2bool("maybe")
