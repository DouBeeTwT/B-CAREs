import torch
from torch import Tensor
from typing import Dict

def ploter(preds:Dict[str, Tensor]):
    print(preds["scores"])
    return 0