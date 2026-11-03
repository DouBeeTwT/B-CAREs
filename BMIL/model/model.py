import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model import backbone, neck, head, block
from torch import Tensor
from typing import List, Dict
import math, cv2

class BMIL(nn.Module):
    def __init__(self, args=None,
                 backbone_name="ResNet101",
                 protonet_name="DeepLabV3",
                 hidden_channels=256,
                 class_num=7,
                 need_P2=False,
                 need_P6=True,
                 need_P7=True,
                 base_num=4,
                 attention_size=14,
                 neck_layers=4,
                 mi_size=32,
                 stride=8,
                 sample_radius=1.5,
                 ):
        super().__init__()
        if args is None:
            self.backbone_name = backbone_name
            self.protonet_name = protonet_name
            self.hidden_channels = hidden_channels
            self.class_num = class_num
            self.need_P2 = need_P2
            self.need_P6 = need_P6
            self.need_P7 = need_P7
            self.base_num = base_num
            self.attention_size = attention_size
            self.neck_layers = neck_layers
            self.mi_size = mi_size
            self.stride = stride
            self.sample_radius = sample_radius
        else:
            self.backbone_name = args.backbone_name
            self.protonet_name = args.protonet_name
            self.hidden_channels = args.hidden_channels
            self.class_num = args.class_num
            self.need_P2 = args.need_P2
            self.need_P6 = args.need_P6
            self.need_P7 = args.need_P7
            self.base_num = args.base_num
            self.attention_size = args.attention_size
            self.neck_layers = args.neck_layers
            self.mi_size = args.mi_size
            self.stride = args.stride
            self.sample_radius = args.sample_radius

        # Backbone
        if self.backbone_name == "ResNet101":
            self.backbone = backbone.ResNet101()
        elif self.backbone_name == "ResNet50":
            self.backbone = backbone.ResNet50()
        elif self.backbone_name == "StarNet":
            self.backbone = backbone.StarNet(base_dim=self.hidden_channels)
        
        # Neck
        self.neck = nn.Sequential(neck.FPN([self.hidden_channels*2**x for x in range(4)],
                                           self.hidden_channels, self.need_P2, self.need_P6, self.need_P7),
                                  #neck.BiFPN(feature_list=[hidden_channels]*5) 
                                  )
        
        # Head
        self.head = head.MultipleHead(in_channels=self.hidden_channels, num_class= self.class_num,
                                      num_attention= self.base_num*self.attention_size*self.attention_size,
                                      num_layers= self.neck_layers)
        self.pos_head = block.Prediction_pos_Net()
        
        # ProtoNet
        if self.protonet_name == "CNN":
            self.protonet = block.ProtoNet(in_channels=self.hidden_channels,
                                           out_channels=self.base_num)
        elif self.protonet_name == "DeepLabV3":
            self.protonet = block.DeepLabV3Decoder(in_channels=self.hidden_channels,
                                                   hidden_channels=self.hidden_channels,
                                                   out_channels=self.base_num)
        
        # Loss Param
        self.mi = [[self.mi_size*2**(x-1), self.mi_size*2**x] for x in range(5)]
        self.mi[0][0] = -1
        #self.mi = [[-1, mi], [mi, mi*2], [mi*2, mi*4], [mi*4, mi*8], [mi*8, mi*32]]
        self.strides = [self.stride*2**i for i in range(5)]
        self.center_sample_radius = self.sample_radius

    def forward(self, X, Y=None, epoch=None):
        """
        Output:
            predictions = {"cls","reg","ctr","pos","atn","map","msk"}
            batch_annotations = [B, (boxes + labels + masks)]
        """
        batch_size, channel_size, height, width = X.shape
        
        # Model
        feature_maps = self.neck(self.backbone(X))
        if self.protonet_name == "CNN":
            map = self.protonet(feature_maps[0])
        elif self.protonet_name == "DeepLabV3":
            map = self.protonet(feature_maps[2], feature_maps[0])
        predictions_dict:dict[str, list] = {"cls":[],"reg":[],"ctr":[],"atn":[],"fea":[],"pos":[],"map":map}
        
        # batch position predictions dict
        if Y is not None:
            fds_labels = []
            for y in Y:
                fds_labels.append(y["labels"])
            fds_labels = torch.cat(fds_labels, dim=1)
        else:
            fds_labels = None

 
        for feature_map in feature_maps[-5:]:
            pred = self.head(feature_map, fds_labels, epoch)
            for key in pred.keys():
                predictions_dict[key].append(pred[key].permute(0,2,3,1))
            pos = self.pos_head(batch_size, feature_map.shape[2], [height, width], device=feature_maps[0].device)
            predictions_dict["pos"].append(pos.permute(0,2,3,1))
        predictions_dict["pos"] = self.get_batch_position_view(predictions_dict["pos"])
        
        # batch position targets
        annotations = []
        if Y is not None:
            for per_image_annotation_dict in Y:
                boxes = per_image_annotation_dict["boxes"]
                labels = per_image_annotation_dict["labels"]
                masks = per_image_annotation_dict["masks"]
                for box, label, mask in zip(boxes, labels, masks):
                    box = box.unsqueeze(0)
                    label = label.unsqueeze(0)
                    mask = mask.unsqueeze(0).view(1,-1)
                    annotation = torch.cat([box, label ,mask], axis=1)
                    annotations.append(annotation)
            annotations = torch.cat(annotations).unsqueeze(1)
            batch_annotations = self.get_batch_position_annotations(predictions_dict, annotations)
        else:
            batch_annotations = None
        for key in ["cls", "reg", "ctr", "atn", "fea"]:
            predictions_dict[key] = self.get_batch_position_view(predictions_dict[key])
        
        """# label head
        if Y is not None:
            vectors = []
            for f,b in zip(feature_maps[0], batch_annotations[:, 0, 0:4]):
                b = torch.cat([torch.tensor([0.0], device=b.device), b]).unsqueeze(0)
                f = f.unsqueeze(0)
                vector = self.label_head(X=f, boxes=b)
                vectors.append(vector)
            vectors = torch.stack(vectors)
            predictions_dict["lab"] = vectors"""

        return predictions_dict, batch_annotations
    def get_batch_position_view(self, X:List[Tensor]) -> Tensor:
        """
        Change the shape of X
        Input: X 
               The shape is [tensor[B, W1, H1, C],
                             tensor[B, W2, H2, C],
                             ...]
        Output: x_reshaped
                The shape is tensor[B, points, C]
                points is W1*H1+W2*H2+...
        """
        x_reshape_list = [x.view(x.shape[0], -1, x.shape[-1]) for x in X]
        x_reshaped = torch.concat(x_reshape_list, dim=1)
        return x_reshaped

    def get_batch_position_annotations(self, preds, annotations, use_center_sample=True):
        '''
        Assign a ground truth target for each position on feature map
        '''
        
        device = annotations.device
        batch_mi, batch_stride = [], []
        for reg_head, mi, stride in zip(preds["reg"], self.mi, self.strides):
            mi = torch.tensor(mi).to(device)
            B, H, W, _ = reg_head.shape
            per_level_mi = torch.zeros(B, H, W, 2).to(device)
            per_level_mi = per_level_mi + mi
            batch_mi.append(per_level_mi)
            per_level_stride = torch.zeros(B, H, W, 1).to(device)
            per_level_stride = per_level_stride + stride
            batch_stride.append(per_level_stride)

        all_points_mi, all_points_stride=[], []
        for per_level_mi, per_level_stride in zip(batch_mi, batch_stride):
            per_level_mi = per_level_mi.view(per_level_mi.shape[0], -1, per_level_mi.shape[-1])
            per_level_stride = per_level_stride.view(per_level_stride.shape[0], -1, per_level_stride.shape[-1])
            all_points_mi.append(per_level_mi)
            all_points_stride.append(per_level_stride)

        all_points_mi = torch.cat(all_points_mi, dim=1)
        all_points_stride = torch.cat(all_points_stride, dim=1)

        batch_targets, mask_targets = [],[]
        for per_image_position, per_image_mi, per_image_stride, per_image_annotations in zip(
                preds["pos"], all_points_mi, all_points_stride, annotations):
            per_image_annotations = per_image_annotations[per_image_annotations[:, 4] >= 0]
            points_num = per_image_position.shape[0]
            mask_size = per_image_annotations.shape[1]-5

            if per_image_annotations.shape[0] == 0:
                # 6:l,t,r,b,class_index,center-ness_gt
                per_image_targets = torch.zeros([points_num, 6], dtype=torch.float32, device=device)
                per_image_masks = torch.zeros([points_num, mask_size], dtype=torch.float32, device=device)
            else:
                annotaion_num = per_image_annotations.shape[0]
                per_image_gt_bboxes = per_image_annotations[:, 0:4]
                candidates = torch.zeros([points_num, annotaion_num, 4], dtype=torch.float32, device=device)
                candidates = candidates + per_image_gt_bboxes.unsqueeze(0)

                per_image_position = per_image_position.unsqueeze(1).repeat(1, annotaion_num, 1)

                if use_center_sample:
                    candidates_center = (candidates[:, :, 2:4] + candidates[:, :, 0:2]) / 2
                    judge_distance = per_image_stride * self.center_sample_radius
                    judge_distance = judge_distance.repeat(1, annotaion_num)

                candidates[:,:,0:2] = per_image_position[:,:,0:2] - candidates[:,:,0:2]
                candidates[:,:,2:4] = candidates[:,:,2:4] - per_image_position[:,:,0:2]

                candidates_min_value, _ = candidates.min(axis=-1, keepdim=True)
                sample_flag = (candidates_min_value[:, :, 0] >0).int().unsqueeze(-1)
                # get all negative reg targets which points ctr out of gt box
                candidates = candidates * sample_flag

                # if use center sample get all negative reg targets which points not in center circle
                if use_center_sample:
                    compute_distance = torch.sqrt((per_image_position[:, :, 0]-candidates_center[:, :, 0])**2 +
                                                  (per_image_position[:, :, 1]-candidates_center[:, :, 1])**2)
                    center_sample_flag = (compute_distance < judge_distance).int().unsqueeze(-1)
                    candidates = candidates * center_sample_flag

                # get all negative reg targets which assign ground turth not in range of mi
                candidates_max_value, _ = candidates.max(axis=-1, keepdim=True)
                per_image_mi = per_image_mi.unsqueeze(1).repeat(1, annotaion_num, 1)
                m1_negative_flag = (candidates_max_value[:, :, 0]>per_image_mi[:, :, 0]).int().unsqueeze(-1)
                candidates = candidates * m1_negative_flag
                m2_negative_flag = (candidates_max_value[:, :, 0]<per_image_mi[:, :, 1]).int().unsqueeze(-1)
                candidates = candidates * m2_negative_flag

                final_sample_flag = candidates.sum(axis=-1).sum(axis=-1)
                final_sample_flag = final_sample_flag > 0
                positive_index = (final_sample_flag == True).nonzero(as_tuple=False).squeeze(dim=-1)

                # if no assign positive sample
                if len(positive_index) == 0:
                    del candidates
                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6], dtype=torch.float32, device=device)
                    per_image_masks = torch.zeros([points_num, mask_size], dtype=torch.float32, device=device)
                else:
                    positive_candidates = candidates[positive_index]

                    del candidates

                    sample_box_gts = per_image_annotations[:, 0:4].unsqueeze(0)
                    sample_box_gts = sample_box_gts.repeat(positive_candidates.shape[0], 1, 1)
                    sample_class_gts = per_image_annotations[:, 4].unsqueeze(-1).unsqueeze(0)
                    sample_class_gts = sample_class_gts.repeat(positive_candidates.shape[0], 1, 1)
                    sample_msk_gts = per_image_annotations[:, 5:].unsqueeze(0)
                    sample_msk_gts = sample_msk_gts.repeat(positive_candidates.shape[0], 1, 1)

                    # 6:l,t,r,b,class_index,center-ness_gt
                    per_image_targets = torch.zeros([points_num, 6], dtype=torch.float32, device=device)
                    per_image_masks = torch.zeros([points_num, mask_size], dtype=torch.float32, device=device)

                    if positive_candidates.shape[1] == 1:
                        # if only one candidate for each positive sample
                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        # class_index value from 1 to 80 represent 80 positive classes
                        # class_index value 0 represenet negative class
                        positive_candidates = positive_candidates.squeeze(1)
                        sample_class_gts = sample_class_gts.squeeze(1)
                        per_image_targets[positive_index, 0:4] = positive_candidates
                        per_image_targets[positive_index, 4:5] = sample_class_gts + 1

                        l = per_image_targets[positive_index, 0:1]
                        t = per_image_targets[positive_index, 1:2]
                        r = per_image_targets[positive_index, 2:3]
                        b = per_image_targets[positive_index, 3:4]
                        per_image_targets[positive_index, 5:6] = torch.sqrt(
                            (torch.min(l, r) / torch.max(l, r)) *
                            (torch.min(t, b) / torch.max(t, b)))
                        per_image_masks[positive_index, :] = sample_msk_gts.squeeze(1)
                    else:
                        # if a positive point sample have serveral object candidates,then choose the smallest area object candidate as the ground turth for this positive point sample
                        gts_w_h = sample_box_gts[:,:,2:4] - sample_box_gts[:,:,0:2]
                        gts_area = gts_w_h[:,:,0] * gts_w_h[:,:,1]
                        positive_candidates_value = positive_candidates.sum(axis=2)

                        # make sure all negative candidates areas==100000000,thus .min() operation wouldn't choose negative candidates
                        INF = 100000000
                        inf_tensor = torch.ones_like(gts_area) * INF
                        gts_area = torch.where(torch.eq(positive_candidates_value, 0.), inf_tensor, gts_area)

                        # get the smallest object candidate index
                        _, min_index = gts_area.min(axis=1)
                        candidate_indexes = (torch.linspace(1, positive_candidates.shape[0], positive_candidates.shape[0])-1).long()
                        final_candidate_reg_gts = positive_candidates[candidate_indexes, min_index, :]
                        final_candidate_cls_gts = sample_class_gts[candidate_indexes, min_index]
                        final_candidate_msk_gts = sample_msk_gts[candidate_indexes, min_index]

                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        per_image_targets[positive_index, 0:4] = final_candidate_reg_gts
                        per_image_targets[positive_index, 4:5] = final_candidate_cls_gts + 1
                        per_image_masks[positive_index, :]  = final_candidate_msk_gts

                        l = per_image_targets[positive_index, 0:1]
                        t = per_image_targets[positive_index, 1:2]
                        r = per_image_targets[positive_index, 2:3]
                        b = per_image_targets[positive_index, 3:4]
                        per_image_targets[positive_index, 5:6] = torch.sqrt(
                            (torch.min(l, r) / torch.max(l, r)) *
                            (torch.min(t, b) / torch.max(t, b)))

            per_image_targets = per_image_targets.unsqueeze(0)
            per_image_masks = per_image_masks.unsqueeze(0)
            batch_targets.append(per_image_targets)
            mask_targets.append(per_image_masks)
        
        batch_targets = torch.cat(batch_targets, dim=0)
        mask_targets = torch.cat(mask_targets, axis=0)
        batch_targets = torch.cat([batch_targets, preds["pos"], mask_targets], dim=2)

        # batch_targets shape:[batch_size, points_num, 8+S*S]
        # 8:    l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
        # S*S:  mask map

        return batch_targets