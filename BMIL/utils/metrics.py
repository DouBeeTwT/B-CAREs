"""Small logging/distributed helpers used by BMIL.utils.trainer.

The training loop was originally written against the ``references`` package that
ships with the torchvision detection reference scripts, which is not installable
from PyPI.  Only three helpers were needed, so they are vendored here.

Adapted from ``torchvision/references/detection/utils.py``
(Copyright (c) Soumith Chintala, BSD-3-Clause), trimmed to what B-CAREs uses:
``SmoothedValue``, ``MetricLogger`` and ``reduce_dict``.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

__all__ = ["SmoothedValue", "MetricLogger", "reduce_dict"]


class SmoothedValue:
    """Track a series of values and provide access to a smoothed window average."""

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def value(self):
        return self.deque[-1] if self.deque else float("nan")

    def __str__(self) -> str:
        return self.fmt.format(median=self.median, avg=self.avg,
                               global_avg=self.global_avg, value=self.value)


class MetricLogger:
    """Accumulate named meters and print them at a fixed interval."""

    def __init__(self, delimiter: str = "\t", logger=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if not isinstance(value, (int, float)):
                raise TypeError("{} should be a tensor or a number, got {}".format(key, type(value)))
            self.meters[key].update(value)

    def __getattr__(self, name):
        # allows metric_logger.loss accessors, as in the torchvision helper
        if name in self.meters:
            return self.meters[name]
        raise AttributeError(name)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def log_every(self, iterable, print_freq: int, header: Optional[str] = None):
        i = 0
        total = len(iterable)
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        start_time = time.time()
        end = start_time

        if not header:
            header = ""
        self._log("{}{}: {}".format(
            header, 8 * " ", "  ".join("{}: {}".format(k, v) for k, v in self.meters.items())))

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total - 1:
                eta_seconds = iter_time.global_avg * (total - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self._log(
                    "{}{eta}: {eta_string}  {iter}: {iter_time}  {data}: {data_time}  {meters}".format(
                        header, eta="eta:", eta_string=eta_string,
                        iter="time:", iter_time=iter_time, data="data:", data_time=data_time,
                        meters="  ".join("{}: {}".format(k, str(v)) for k, v in self.meters.items())))
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self._log("{} Total time: {} ({:.4f} s / it)".format(
            header, total_time_str, total_time / max(total, 1)))

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def reduce_dict(input_dict: Dict[str, Any], average: bool = True) -> Dict[str, Any]:
    """Average the values of a dict across processes (no-op in single-process runs)."""
    world_size = dist.get_world_size() if is_dist_avail_and_initialized() else 1
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for key in sorted(input_dict.keys()):
            names.append(key)
            values.append(input_dict[key])
        values = torch.stack(values, 0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        return dict(zip(names, values))
