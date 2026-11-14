import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# FPN : [C2, C3, C4, C5] -> [P2, P3, P4, P5, P6, P7]
class FPN(nn.Module):
    def __init__(self, in_channels_list:list[int, int, int, int], out_channels,
                 need_P2:bool = True, need_P6:bool = True, need_P7:bool = True):
        super().__init__()
        self.need_P2 = need_P2
        self.need_P6 = need_P6 and need_P7
        self.need_P7 = need_P7
        # 特征维度对齐
        self.C_to_P_layers = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list]
        )
        # 下采样
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)) for _ in range(2)]
        )
        # 后激活
        self.pred_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)) for _ in in_channels_list]
        )


    def forward(self, feature_maps_from_backbone) -> list[torch.Tensor]:
        feature_maps_list = []
        
        P2 = self.C_to_P_layers[0](feature_maps_from_backbone['C2'])
        P3 = self.C_to_P_layers[1](feature_maps_from_backbone['C3'])
        P4 = self.C_to_P_layers[2](feature_maps_from_backbone['C4'])
        P5 = self.C_to_P_layers[3](feature_maps_from_backbone['C5'])

        P4 = P4 + F.interpolate(P5, size=P4.shape[2], mode='bilinear')
        P3 = P3 + F.interpolate(P4, size=P3.shape[2], mode='bilinear')
        P2 = P2 + F.interpolate(P3, size=P2.shape[2], mode='bilinear')
        
        if self.need_P2:
            P2 = self.pred_layers[0](P2)
            feature_maps_list.append(P2)
        P3 = self.pred_layers[1](P3)
        feature_maps_list.append(P3)
        P4 = self.pred_layers[2](P4)
        feature_maps_list.append(P4)
        P5 = self.pred_layers[3](P5)
        feature_maps_list.append(P5)

        if self.need_P6:
            P6 = self.downsample_layers[0](P5)
            feature_maps_list.append(P6)
        if self.need_P7:
            P7 = self.downsample_layers[1](P6)
            feature_maps_list.append(P7)

        return feature_maps_list

class UpSample(nn.Module):
    def __init__(self, channel_in:int, channel_out:int, mode:str="nearest", size:int=None) -> None:
        """
        Args:
            feture_input: in_channels for Conv2d()
            feture_output: out_channels for Conv2d()
            mode: algorithm used for upsampling. Default: 'nearest'
                  'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact' 
        """
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1)
        self.mode = mode
        self.size = size

    def forward(self, x:Tensor, feature_map:Tensor=None) -> Tensor:
        if self.size is not None:
            result = self.layer(F.interpolate(x, size=self.size, mode=self.mode))
        else:
            result = self.layer(F.interpolate(x, scale_factor=2, mode=self.mode))
        
        if feature_map is not None:
            result = torch.cat((result, feature_map), dim=1)
        return result

class DownSample(nn.Module):
    def __init__(self, channel_in:int, channel_out:int, is_act:bool=False) -> None:
        """
        Args:
            feture_input: in_channels for Conv2d()
            feture_output: out_channels for Conv2d()
        """
        super(DownSample, self).__init__()
        self.is_act = is_act
        self.layer = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=3,
                      stride=2, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(channel_out)
        )
        self.act = nn.LeakyReLU()
    
    def forward(self, x:Tensor) -> Tensor:
        result = self.layer(x)
        if self.is_act:
            result = self.act(result)
        return result

class BiFPN(nn.Module):
    def __init__(self, feature_list) -> None:
        super(BiFPN, self).__init__()
        # Up Sample Layer
        self.P7i_P6m = UpSample(feature_list[4], feature_list[3])
        self.P6m_P5m = UpSample(feature_list[3], feature_list[2], size=(7,7))
        self.P5m_P4m = UpSample(feature_list[2], feature_list[1])
        self.P4m_P3o = UpSample(feature_list[1], feature_list[0])
        # Down Sample Layer
        self.P3o_P4o = DownSample(feature_list[0], feature_list[1])
        self.P4o_P5o = DownSample(feature_list[1], feature_list[2])
        self.P5o_P6o = DownSample(feature_list[2], feature_list[3])
        self.P6o_P7o = DownSample(feature_list[3], feature_list[4])
        # Swish
        self.swish = nn.SiLU()

    def forward(self, x:list) -> list:
        P3i, P4i, P5i, P6i, P7i = x[0], x[1], x[2], x[3], x[4]
        # Up layer
        P6m = self.swish(P6i + self.P7i_P6m(P7i))
        P5m = self.swish(P5i + self.P6m_P5m(P6m))
        P4m = self.swish(P4i + self.P5m_P4m(P5m))
        P3o = self.swish(P3i + self.P4m_P3o(P4m))
        # Down layer
        P4o = self.swish(P4i + P4m + self.P3o_P4o(P3o))
        P5o = self.swish(P5i + P5m + self.P4o_P5o(P4o))
        P6o = self.swish(P6i + P6m + self.P5o_P6o(P5o))
        P7o = self.swish(P7i + self.P6o_P7o(P6o))
        
        return [P3o, P4o, P5o, P6o, P7o]