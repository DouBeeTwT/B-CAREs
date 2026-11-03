import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Prediction position : 
class Prediction_pos_Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_size, in_channels, out_channels, device):
        if type(in_channels) is int:
            in_channel_x = in_channels
            in_channel_y = in_channels
        elif type(in_channels) is list and len(in_channels) == 2:
            in_channel_x = in_channels[0]
            in_channel_y = in_channels[1]

        if type(out_channels) is int:
            out_channel_x = out_channels
            out_channel_y = out_channels
        elif type(out_channels) is list and len(out_channels) == 2:
            out_channel_x = out_channels[0]
            out_channel_y = out_channels[1]

        shift_x = (torch.arange(0, in_channel_x, device=device)+0.5) / in_channel_x * out_channel_x
        shift_y = (torch.arange(0, in_channel_y, device=device)+0.5) / in_channel_y * out_channel_y
        pos = torch.tensor([[[x, y] for y in shift_y] for x in shift_x], device=device).permute(2,1,0).contiguous().unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return pos
    
class ProtoNet(nn.Module):
    def __init__(self, in_channels:int=256, out_channels:int=7):
        super().__init__()
        layers = []
        for _ in range(3):
            layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))
        self.proto1 = nn.Sequential(*layers)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.proto2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.proto1(x)
        x = self.upsample(x)
        x = self.proto2(x)
        return x

class DeepLabV3Decoder(nn.Module):
    def __init__(self, in_channels:int, hidden_channels:int, out_channels:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels//4)
        self.relu = nn.ReLU(inplace=True)
 
        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(in_channels//4+hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_channels, out_channels, 1, stride=1)
        )
 
    def forward(self, x, low_level_features):
        low_level_features = self.relu(self.bn1(self.conv1(low_level_features)))
 
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

        return x # shape [B=32, K=7, 224, 224]

class blender(object):
    def __init__(self, R:int=28, K:int=4, M:int=4) -> None:
        self.R = R
        self.K = K
        self.M = M

    def __call__(self, atn:Tensor, bases:Tensor, gt_boxes:Tensor=None, pd_boxes:Tensor=None) -> Tensor:
        """
        Input:
            atn:    [B, K, M, M]
            bases:  [1, K, W, H]
            gt_boxes:  [1, 5]
            pd_boxes:  [B, 5]
        output:
            m_d:    [B, R, R]
        """
        if gt_boxes is not None:
            r_d = torchvision.ops.roi_align(input=bases, boxes=gt_boxes, output_size=(self.R, self.R))
            a_d = F.interpolate(atn, [self.R, self.R])
            s_d = F.softmax(a_d, dim=1)
            m_d = s_d * r_d.expand(s_d.size(0), -1, -1, -1)
        elif pd_boxes is not None:
            r_d = torchvision.ops.roi_align(input=bases, boxes=pd_boxes, output_size=(self.R, self.R))
            a_d = F.interpolate(atn, [self.R, self.R])
            s_d = F.softmax(a_d, dim=1)
            m_d = s_d * r_d
        return m_d.sum(dim=1)
        


class Blender(nn.Module):
    def __init__(self, image_w:int, image_h:int):
        super().__init__()
        self.image_w = image_w
        self.image_h= image_h

    def forward(self, batch_A:torch.Tensor, batch_B:torch.Tensor, batch_Box:torch.Tensor=None) -> torch.Tensor:
        """
        Input:
            A: Attention        | [B, N, K*W*H] | [N, K*W*H] -> [1, N, K*W*H]
            B: Base (Score Map) | [B, K, W, H]  | [K, W, H]  -> [1, K, W, H]
            Box: Box            | [B, N, 4]     | [N, 4]     -> [1, N, 4]     4:x_min, y_min, x_max, y_max
        Output:
            C: Combined Mask    | [B, N, Figure_size_W, Figure_size_H] | [N, Figure_size_W, Figure_size_H]
        """
        if len(batch_A.shape) == 2:
            batch_A.unsqueeze(0)
        if len(batch_B.shape) == 3:
            batch_B.unsqueeze(0)
        if batch_Box is not None and len(batch_Box.shape) == 2:
            batch_Box.unsqueeze(0)

        batch_Mask = []
        if batch_Box is None:
            for A, B in zip(batch_A, batch_B):
                Mask = self.calc_in_batch(A, B)
                batch_Mask.append(Mask)
        else:
            for A, B, Box in zip(batch_A, batch_B, batch_Box):
                Mask = self.calc_in_batch(A, B, Box)
                batch_Mask.append(Mask)
        batch_Mask = torch.stack(batch_Mask)
        if batch_Mask.shape[0] == 1:
            batch_Mask = batch_Mask.squeeze(0)
        return batch_Mask
        
        
      
    def calc_in_batch(self, A:torch.Tensor, B:torch.Tensor, Box:torch.Tensor=None) -> torch.Tensor:
        """
        Input:
            A: Attention        | [N, K*W*H]
            B: Base (Score Map) | [K, W, H]
            Box: Box            | [N, 4]  4:x_min, y_min, x_max, y_max
        Output:
            C: Combined Mask    | [N, Figure_size_W, Figure_size_H]
        """
        K, W, H = B.shape
        A = A.view(-1, K, W, H)
        A = F.softmax(A, dim=1)
        A = A.view(-1, K*W*H)
        B = B.reshape(1, -1).squeeze(1)
        C = [torch.mul(B, a).view(K, W, H).sum(dim=0).unsqueeze(0) for a in A]
        C = torch.cat(C, dim=0)
        C = F.interpolate(C.unsqueeze(0), [self.image_w, self.image_h], mode='bilinear').squeeze(0)
        if Box is None:
            return C
        else:
            O = torch.zeros_like(C)
            for i in range(Box.shape[0]):
                O[i, Box[i,0]:Box[i,2], Box[i,1]:Box[i,3]] = C[i, Box[i,0]:Box[i,2], Box[i,1]:Box[i,3]]
            return O

if __name__ == "__main__":
    atn = torch.rand(size=(10, 4, 4, 4))
    map = torch.rand(size=(1, 4, 224, 224))
    boxes = torch.tensor([[0, 120, 15, 170, 35]]).float()
    my_blender = blender()
    m_d = my_blender(atn, map, boxes)