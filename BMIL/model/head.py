import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict
from torch import Tensor
from .fds import FDS
import torchvision

class MultipleHead(nn.Module):
    def __init__(self,
                 in_channels:int=256,
                 num_class:int=7,
                 num_attention:int=4*14*14,
                 num_layers:int=4):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))
        self.public_head = nn.Sequential(*layers)

        self.cls_head = nn.Sequential(nn.Conv2d(in_channels, num_class, 3, 1, 1), nn.Softmax(dim=1))
        self.reg_head = nn.Sequential(nn.Conv2d(in_channels, 4, 3, 1, 1), nn.Sigmoid())
        self.ctr_head = nn.Sequential(nn.Conv2d(in_channels, 1, 3, 1, 1))
        self.atn_head = nn.Sequential(nn.Conv2d(in_channels, num_attention, 3, 1, 1))
            
    def forward(self, X:Tensor, labels:Tensor=None, epoch=0) -> Dict[str, Tensor]:
        preds:Dict[str, Tensor] = {}
        X = self.public_head(X)
        
        preds["fea"] = X                # [B, in_ch, w, h]
        preds["cls"] = self.cls_head(X) # [B, C,     w, h]
        preds["reg"] = self.reg_head(X) # [B, 4,     w, h]
        preds["ctr"] = self.ctr_head(X) # [B, 1,     w, h]
        preds["atn"] = self.atn_head(X) # [B, K*W*H, w, h]
        
        return preds
    
class Label_Head(nn.Module):
    def __init__(self,
                 in_channels:int=784,
                 out_channels:int=2,
                 hidden_channels:int=100):
        super().__init__()
        
        self.layers = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1),
                                    nn.Flatten(),
                                    nn.Linear(in_channels, hidden_channels),
                                    nn.Linear(hidden_channels, out_channels))
    def forward(self, X:Tensor, boxes:Tensor) -> Tensor:
        X = torchvision.ops.roi_align(input=X, boxes=boxes, output_size=(7,7), spatial_scale=1.0/8.0)
        X = self.layers(X)
        return X