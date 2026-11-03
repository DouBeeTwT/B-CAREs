import torch 
from torch import Tensor
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import math, cv2
import numpy as np
from ..model.block import Blender, blender

def Divide(outputs, labels, subggroup):
    def del_tensor_ele(arr,index):
        arr1 = arr[0:index]
        arr2 = arr[index+1:]
        return torch.cat((arr1,arr2), dim=0)
    Min_img, Maj_img, sub_maj, Min_lab, Maj_lab, sub_min  = outputs, outputs, subggroup, labels, labels, subggroup
    for j in range(len(subggroup)):
        for x in range(len(sub_maj)):
            if sub_maj[x].tolist()[0] == 0:
                Maj_img = del_tensor_ele(Maj_img,x)
                Maj_lab = del_tensor_ele(Maj_lab,x)
                sub_maj = del_tensor_ele(sub_maj,x)
                break
    for j in range(len(subggroup)):
        for x in range(len(sub_min)):
            if sub_min[x].tolist()[0] == 1:
                Min_img = del_tensor_ele(Min_img,x)
                Min_lab = del_tensor_ele(Min_lab,x)
                sub_min = del_tensor_ele(sub_min,x)
                break

    return Maj_img, Maj_lab, sub_maj, Min_img, Min_lab, sub_min


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        """Min_img,sub_maj, Maj_img, sub_min = outputs, subggroup, outputs, subggroup
        def del_tensor_ele(arr,index):
                arr1 = arr[0:index]
                arr2 = arr[index+1:]
                return torch.cat((arr1,arr2), dim=0)
        for j in range(len(subggroup)):
            for x in range(len(sub_maj)):
                if sub_maj[x].tolist()[0] == 0:
                    Maj_img = del_tensor_ele(Maj_img,x)
                    sub_maj = del_tensor_ele(sub_maj,x)
                    break
        for j in range(len(subggroup)):
            for x in range(len(sub_min)):
                if sub_min[x].tolist()[0] == 1:
                    Min_img = del_tensor_ele(Min_img,x)
                    sub_min = del_tensor_ele(sub_min,x)
                    break

        source = Min_img
        target = Maj_img"""
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = torch.mean(XX + YY - XY - YX)
        return loss

class BatchSpectralShrinkage(nn.Module):

    def __init__(self, k=1):
        super(BatchSpectralShrinkage, self).__init__()
        self.k = k

    def forward(self, feature):
        result = 0
        u, s, v = torch.svd(feature.t())
        num = s.size(0)
        for i in range(self.k):
            result += torch.pow(s[num-1-i], 2)
        return result/max(s)

class FCOSPositions:

    def __init__(self, strides=[8, 16, 32, 64, 128]):
        self.strides = np.array(strides, dtype=np.float32)

    def __call__(self, fpn_feature_sizes):
        '''
        generate one image positions
        '''

        one_image_positions = []
        for stride, fpn_feature_size in zip(self.strides, fpn_feature_sizes):
            featrue_positions = self.generate_positions_on_feature_map(
                fpn_feature_size, stride)
            one_image_positions.append(featrue_positions)

        # if input size:[640,640]
        # one_image_positions shape:[[80, 80, 2],[40, 40, 2],[20, 20, 2],[10, 10, 2],[5, 5, 2]]
        # per position format:[x_center,y_center]
        return one_image_positions

    def generate_positions_on_feature_map(self, feature_map_size, stride):
        '''
        generate one feature map positions
        '''

        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = (np.arange(0, feature_map_size[0]) + 0.5) * stride
        shifts_y = (np.arange(0, feature_map_size[1]) + 0.5) * stride

        # feature_map_positions shape:[w,h,2] -> [h,w,2]
        feature_map_positions = np.array([[[shift_x, shift_y]
                                           for shift_y in shifts_y]
                                          for shift_x in shifts_x],
                                         dtype=np.float32)
        feature_map_positions = np.transpose(feature_map_positions,
                                             axes=(1, 0, 2))
        feature_map_positions = np.ascontiguousarray(feature_map_positions,
                                                     dtype=np.float32)

        # feature_map_positions format: [point_nums,2],2:[x_center,y_center]
        return feature_map_positions

class IoUMethod:

    def __init__(self):
        pass

    def __call__(self, boxes1, boxes2, iou_type='IoU', box_type='xyxy'):
        '''
        box1 format:[...,4]
        box2 format:[...,4]
        xyxy type:[x_min,y_min,x_max,y_max]
        xywh type:[x_center,y_center,w,h]
        '''
        assert iou_type in ['IoU', 'GIoU', 'DIoU', 'CIoU', 'EIoU'], 'wrong IoU type!'
        assert box_type in ['xyxy', 'xywh'], 'wrong box_type type!'

        if box_type == 'xywh':
            # transform format from [x_ctr,y_ctr,w,h] to xyxy
            boxes1_x1y1 = boxes1[..., 0:2] - boxes1[..., 2:4] / 2
            boxes1_x2y2 = boxes1[..., 0:2] + boxes1[..., 2:4] / 2
            boxes1 = torch.cat([boxes1_x1y1, boxes1_x2y2], dim=1)

            boxes2_x1y1 = boxes2[..., 0:2] - boxes2[..., 2:4] / 2
            boxes2_x2y2 = boxes2[..., 0:2] + boxes2[..., 2:4] / 2
            boxes2 = torch.cat([boxes2_x1y1, boxes2_x2y2], dim=1)

        overlap_area_xymin = torch.max(boxes1[..., 0:2], boxes2[..., 0:2])
        overlap_area_xymax = torch.min(boxes1[..., 2:4], boxes2[..., 2:4])
        overlap_area_sizes = torch.clamp(overlap_area_xymax - overlap_area_xymin, min=0)
        overlap_area = overlap_area_sizes[..., 0] * overlap_area_sizes[..., 1]

        boxes1_wh = torch.clamp(boxes1[..., 2:4] - boxes1[..., 0:2], min=1)
        boxes2_wh = torch.clamp(boxes2[..., 2:4] - boxes2[..., 0:2], min=1)

        boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
        boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]

        # compute ious between boxes1 and boxes2
        union_area = boxes1_area + boxes2_area - overlap_area
        union_area = torch.clamp(union_area, min=1e-4)
        ious = overlap_area / union_area

        if iou_type == 'IoU':
            return ious
        else:
            if iou_type in ['GIoU', 'DIoU', 'CIoU', 'EIoU']:
                enclose_area_top_left = torch.min(boxes1[..., 0:2], boxes2[..., 0:2])
                enclose_area_bot_right = torch.max(boxes1[..., 2:4], boxes2[..., 2:4])
                enclose_area_sizes = torch.clamp(enclose_area_bot_right - enclose_area_top_left, min=0)
                if iou_type in ['DIoU', 'CIoU', 'EIoU']:
                    # https://arxiv.org/abs/1911.08287v1
                    # compute DIoU c2 and p2
                    # c2:convex diagonal squared
                    c2 = enclose_area_sizes[..., 0]**2 + enclose_area_sizes[..., 1]**2
                    c2 = torch.clamp(c2, min=1e-4)
                    # p2:center distance squared
                    boxes1_ctr = (boxes1[..., 2:4] + boxes1[..., 0:2]) / 2
                    boxes2_ctr = (boxes2[..., 2:4] + boxes2[..., 0:2]) / 2
                    p2 = (boxes1_ctr[..., 0] - boxes2_ctr[..., 0])**2 + (boxes1_ctr[..., 1] - boxes2_ctr[..., 1])**2
                    if iou_type == 'DIoU':
                        return ious - p2 / c2
                    elif iou_type == 'CIoU':
                        # compute CIoU v and alpha
                        v = (4 / math.pi**2) * torch.pow(
                            torch.atan(boxes2_wh[:, 0] / boxes2_wh[:, 1]) -
                            torch.atan(boxes1_wh[:, 0] / boxes1_wh[:, 1]), 2)

                        with torch.no_grad():
                            alpha = v / torch.clamp(1 - ious + v, min=1e-4)
                        cious = ious - (p2 / c2 + v * alpha)

                        if torch.isnan(cious).any():
                            with open("print.txt", "w") as f:
                                print(boxes1, boxes2, enclose_area_sizes, boxes1_ctr, boxes2_ctr, file=f)

                        return cious
                    elif iou_type == 'EIoU':
                        pw2 = (boxes2_wh[..., 0] - boxes1_wh[..., 0])**2
                        ph2 = (boxes2_wh[..., 1] - boxes1_wh[..., 1])**2
                        cw2 = enclose_area_sizes[..., 0]**2
                        ch2 = enclose_area_sizes[..., 1]**2
                        cw2 = torch.clamp(cw2, min=1e-4)
                        ch2 = torch.clamp(ch2, min=1e-4)

                        return ious - (p2 / c2 + pw2 / cw2 + ph2 / ch2)
                else:
                    enclose_area = enclose_area_sizes[:,0] * enclose_area_sizes[:,1]
                    enclose_area = torch.clamp(enclose_area, min=1e-4)

                    return ious - (enclose_area - union_area) / enclose_area

class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = torch.nn.functional.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class FCOSLoss(nn.Module):
    def __init__(self, args=None,
                 class_num=7,
                 iou_name="CIoU",
                 mi_size=32,
                 stride=8,
                 alpha=0.25,
                 gamma=2,
                 loss_weight=[.5, 1., 1., 1., 1.],
                 use_sample = True,
                 sample_radius = 1.5,
                 figure_size = 224,
                 base_num = 4,
                 attention_size = 14,
                 blender_size = 56):
        super(FCOSLoss, self).__init__()
        if args is None:
            self.class_num = class_num
            self.box_loss_iou_type = iou_name
            self.mi_size = mi_size
            self.stride = stride
            self.alpha = alpha
            self.gamma = gamma
            self.loss_weight = loss_weight
            self.use_center_sample = use_sample
            self.center_sample_radius = sample_radius
            self.figure_size = figure_size
            self.K = base_num
            self.M = attention_size
            self.R = blender_size
        else:
            self.class_num = args.class_num
            self.box_loss_iou_type = args.iou_name
            self.mi_size = args.mi_size
            self.stride = args.stride
            self.alpha = args.alpha_cls
            self.gamma = args.gamma_cls
            self.loss_weight = args.loss_weight
            self.use_center_sample = args.use_sample
            self.center_sample_radius = args.sample_radius
            self.figure_size = args.figure_size
            self.K = args.base_num
            self.M = args.attention_size
            self.R = args.blender_size
        

        assert self.box_loss_iou_type in ['IoU', 'GIoU', 'DIoU', 'CIoU', 'EIoU'], 'wrong IoU type!'
        
        self.mi = [[self.mi_size*2**(x-1), self.mi_size*2**x] for x in range(5)]
        self.mi[0][0] = -1
        self.strides = [self.stride*2**i for i in range(5)]
        self.positions = FCOSPositions(strides=self.strides)
        self.iou_function = IoUMethod()
        self.blender = blender(R=self.R, K=self.K, M=self.M)
        self.focal_loss = focal_loss(alpha = self.alpha,
                                     gamma = self.gamma,
                                     num_classes = self.class_num)

    def forward(self, preds, batch_targets):
        '''
        compute cls loss, reg loss and center-ness loss in one batch
        preds = {"cls","reg","ctr","pos","atn","map"}
        batch_targets
        '''
        
        preds["cls"] = preds["cls"].view(-1, preds["cls"].shape[-1])
        preds["reg"] = preds["reg"].view(-1, preds["reg"].shape[-1])
        preds["ctr"] = preds["ctr"].view(-1, preds["ctr"].shape[-1])
        batch_targets = batch_targets.view(-1, batch_targets.shape[-1])
        
        preds["cls"] = torch.clamp(preds["cls"], min=1e-4, max=1.-1e-4)
        #preds["reg"] = torch.clamp(preds["reg"], min=1e-4, max=self.figure_size)
        preds["ctr"] = torch.clamp(preds["ctr"], min=1e-4, max=1.-1e-4)
        
        cls_loss = self.compute_batch_cls_loss(preds["cls"], batch_targets)
        reg_loss, pred_bboxes = self.compute_batch_reg_loss(preds["reg"], batch_targets)
        ctr_loss = self.compute_batch_ctr_loss(preds["ctr"], batch_targets)
        msk_loss = self.compute_batch_msk_loss2(preds["atn"], preds["map"], pred_bboxes, batch_targets)
        bag_loss = self.compute_batch_bag_loss2(preds, batch_targets, batch_num=preds["map"].shape[0])
        #bag_loss = self.compute_batch_bag_loss3(preds, batch_targets, batch_num=preds["map"].shape[0])

        cls_loss = self.loss_weight[0] * cls_loss
        reg_loss = self.loss_weight[1] * reg_loss
        ctr_loss = self.loss_weight[2] * ctr_loss
        msk_loss = self.loss_weight[3] * msk_loss
        bag_loss = self.loss_weight[4] * bag_loss

        loss_dict = {'cls': cls_loss, 'reg': reg_loss, 'ctr': ctr_loss, 'msk': msk_loss, 'bag':bag_loss}


        return loss_dict

    def compute_batch_cls_loss(self, cls_preds, batch_targets):
        '''
        compute batch focal loss(cls loss)
        cls_preds:[batch_size*points_num,num_classes]
        batch_targets:[batch_size*points_num,8]
        '''
        
        device = cls_preds.device
        positive_points_num = batch_targets[batch_targets[:, 4] > 0].shape[0]
        num_classes = cls_preds.shape[1] # [32*1049, class_num]
        
        if positive_points_num == 0:
            return torch.tensor(0., requires_grad=True).to(device)

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(batch_targets[:, 4].long(), num_classes=num_classes+1) # [32*1049, class_num+1]
        loss_ground_truth = loss_ground_truth[:, 1:]  # [32*1049, class_num]
        loss_ground_truth = loss_ground_truth.float() # [32*1049, class_num]
        alpha_factor = torch.ones_like(cls_preds) * self.alpha # [32*1049, class_num]
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.), alpha_factor, 1. - alpha_factor) # [32*1049, class_num]
        pt = torch.where(torch.eq(loss_ground_truth, 1.), cls_preds, 1. - cls_preds) # [32*1049, class_num]

        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)
        
        #return F.binary_cross_entropy(cls_preds, loss_ground_truth, focal_weight) / positive_points_num

        batch_bce_loss = -(
            loss_ground_truth * torch.log(cls_preds) +
            (1. - loss_ground_truth) * torch.log(1. - cls_preds))
        batch_focal_loss = focal_weight * batch_bce_loss
        batch_focal_loss = batch_focal_loss.sum()
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        batch_focal_loss = batch_focal_loss / positive_points_num
        
        return batch_focal_loss

    def compute_batch_reg_loss(self, reg_preds, batch_targets):
        '''
        compute batch giou loss(reg loss)
        reg_preds:[batch_size*points_num,4]
        batch_targets:[batch_size*anchor_num,8]
        '''
        # only use positive points sample to compute reg loss
        device = reg_preds.device
        #reg_preds = torch.exp(reg_preds)
        reg_preds = reg_preds * (self.figure_size /2.0)
        
        reg_preds = reg_preds[batch_targets[:, 4] > 0]
        batch_targets = batch_targets[batch_targets[:, 4] > 0]
        center_ness_targets = batch_targets[:, 5]

        positive_points_num = batch_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0., requires_grad=True).to(device)

        pred_bboxes_xy_min = batch_targets[:, 6:8] - reg_preds[:, 0:2]
        pred_bboxes_xy_max = batch_targets[:, 6:8] + reg_preds[:, 2:4]
        gt_bboxes_xy_min = batch_targets[:, 6:8] - batch_targets[:, 0:2]
        gt_bboxes_xy_max = batch_targets[:, 6:8] + batch_targets[:, 2:4]

        pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max], dim=1)
        gt_bboxes = torch.cat([gt_bboxes_xy_min, gt_bboxes_xy_max], dim=1)

        ious = self.iou_function(pred_bboxes, gt_bboxes, iou_type=self.box_loss_iou_type, box_type='xyxy')
        
        ious_loss = 1 - ious
        # use center_ness_targets as the weight of gious loss
        ious_loss = ious_loss * center_ness_targets
        ious_loss = ious_loss.mean()

        if ious_loss.isnan():
            ious_loss = torch.tensor(0., requires_grad=True).to(device)

        return ious_loss, pred_bboxes

    def compute_batch_ctr_loss(self, ctr_preds, batch_targets):
        '''
        compute batch center_ness loss(center ness loss)
        center_preds:[batch_size*points_num,4]
        batch_targets:[batch_size*anchor_num,8]
        '''
        # only use positive points sample to compute center_ness loss
        device = ctr_preds.device
        ctr_preds = ctr_preds[batch_targets[:, 4] > 0]
        batch_targets = batch_targets[batch_targets[:, 4] > 0]
        positive_points_num = batch_targets.shape[0]

        if positive_points_num == 0:
            return torch.tensor(0., requires_grad=True).to(device)

        center_ness_targets = batch_targets[:, 5:6]

        center_ness_loss = -(
            center_ness_targets * torch.log(ctr_preds) +
            (1. - center_ness_targets) * torch.log(1. - ctr_preds))
        center_ness_loss = center_ness_loss.sum() / positive_points_num

        return center_ness_loss
    
    def compute_batch_msk_loss1(self, atn_preds:torch.Tensor, score_maps:torch.Tensor, pred_bboxes:torch.Tensor, batch_targets):
        # atn_preds shape:      [batch_size, points_number, k*w*h] -> [positive_points_num, k*w*h]
        # batch_targets shape:  [batch_size*points_number, 8+S*S] -> [positive_points_num, S, S]
        # score_map:            [batch_size, k, w, h] -> [batch_size, points_number,  k, w, h]  -> [positive_points_num, k*w*h]
        # pred_bboxes shape     [positive_points_num, 4]
        device = atn_preds.device
        batch_size, k ,w, h = score_maps.shape
        points_number = atn_preds.shape[1]
        positive_points_num = batch_targets[batch_targets[:,4] > 0].shape[0]
        
        if positive_points_num == 0:
            return torch.tensor(0., requires_grad=True).to(device)
        
        choice_list = batch_targets[:,4] > 0
        batch_targets = batch_targets[choice_list][:, 8:].view(positive_points_num, self.image_w, self.image_h)
        atn_preds = atn_preds.reshape(-1, k*w*h)[choice_list]
        atn_preds = atn_preds.reshape(-1, k ,w*h)
        atn_preds = F.softmax(atn_preds, dim=1)
        atn_preds = atn_preds.reshape(-1, k*w*h)
        score_maps = score_maps.unsqueeze(dim=1).repeat(1, points_number, 1, 1, 1).view(-1, k*w*h)[choice_list]
        
        pred_masks = torch.mul(atn_preds, score_maps).view(positive_points_num, k, w, h).sum(dim=1) # [positive_points_num, w, h]
        pred_masks = F.interpolate(pred_masks.unsqueeze(0), [self.image_w, self.image_h], mode='bilinear').squeeze(0) # [positive_points_num, S, S]
        
        masks = torch.zeros_like(pred_masks)
        pred_bboxes = torch.clamp(pred_bboxes, max=224, min=0)
        Box = pred_bboxes.type(dtype=torch.int)
        for i in range(pred_bboxes.shape[0]):
            masks[i, Box[i,0]:Box[i,2], Box[i,1]:Box[i,3]] = pred_masks[i, Box[i,0]:Box[i,2], Box[i,1]:Box[i,3]]

        # 计算MSE损失
        mask_loss = F.mse_loss(masks, batch_targets, reduction='mean')

        return mask_loss, masks

    def compute_batch_msk_loss2(self, atn_preds:torch.Tensor, score_maps:torch.Tensor, pred_bboxes:torch.Tensor, batch_targets):
        # atn_preds shape:      [batch_size, points_number, k*w*h] -> [batch_size, points_number, k, w, h]
        # batch_targets shape:  [batch_size*points_number, 8+S*S] -> [batch_size, points_number, 8+S*S]
        # score_maps:           [batch_size, k, W, H]
        # pred_bboxes shape     [positive_points_num, 4]
        device = atn_preds.device
        batch_size, k ,w, h = score_maps.shape
        points_number = atn_preds.shape[1]
        positive_points_num = batch_targets[batch_targets[:,4] > 0].shape[0]

        if positive_points_num == 0:
            return torch.tensor(0., requires_grad=True).to(device)
        
        atn_preds = atn_preds.reshape(batch_size, points_number, self.K ,self.M, self.M)
        batch_targets = batch_targets.reshape(batch_size, points_number, -1)

        masks_pd, masks_gt = [], []
        for atn, map, tgt in zip(atn_preds, score_maps, batch_targets):
            # atn: [points_number, k, w, h] -> [positive_points_num, k, w, h]
            # map: [k, W, H]
            # tgt: [points_number, 8+S*S] ---calc box--->  [positive_points_num, box]
            atn = atn[tgt[:, 4] > 0]
            tgt = tgt[tgt[:, 4] > 0]
            map = map.unsqueeze(0)
            if tgt.shape[0] > 0:
                img = tgt[0, 8:].reshape(224, 224)
                gt_bboxes_xy_min = tgt[0, 6:8] - tgt[0, 0:2]
                gt_bboxes_xy_max = tgt[0, 6:8] + tgt[0, 2:4]
                gt_bboxes = torch.cat([gt_bboxes_xy_min, gt_bboxes_xy_max])
                gt_bboxes = torch.cat([torch.tensor([0.0], device=device), gt_bboxes]).unsqueeze(0)
                m_pd = self.blender(atn, map, gt_boxes=gt_bboxes)
                #m_pd = F.sigmoid(m_pd) # new
                gt_bboxes = gt_bboxes.long()
                m_gt = img[gt_bboxes[0,2]:gt_bboxes[0,4], gt_bboxes[0,1]:gt_bboxes[0,3]].unsqueeze(0).unsqueeze(0)
                m_gt = F.interpolate(m_gt, [self.R, self.R], mode='bilinear')
                m_gt = m_gt.squeeze(0).expand(tgt.shape[0], -1, -1)
                masks_pd.append(m_pd.reshape(m_pd.shape[0], -1))
                masks_gt.append(m_gt.reshape(m_gt.shape[0], -1)) 
        masks_pd = torch.cat(masks_pd) # [positive_points_num, R*R]
        masks_gt = torch.cat(masks_gt) # [positive_points_num, R*R]
        
        losses_mask = F.binary_cross_entropy_with_logits(masks_pd, masks_gt, reduction="none") # bce_with_log -> mse+softmax
        batch_targets = batch_targets.reshape(batch_size*points_number, -1)
        ctr_gt = batch_targets[batch_targets[:,4] > 0][:, 5]
        losses_mask = (losses_mask.mean(dim=-1) * ctr_gt).sum()/positive_points_num
        return losses_mask

    def compute_batch_bag_loss1(self, cls_preds, msk_preds, batch_targets, batch_num):
        """
        cls_preds shape: [batch_num*points_num, 7] -> [batch_num, points_num, 7]
        msk_preds shape: [positive_num, 224, 224]
        batch_targets shape: [batch_num*points_num, 8+S*S] -> [batch_num, points_num, 1]
        """

        device = cls_preds.device
        num_classes = cls_preds.shape[1]
        cls_preds = cls_preds.view(batch_num, -1, cls_preds.shape[-1])
        batch_targets = batch_targets.view(batch_num, -1, batch_targets.shape[-1])[:, :, 4:5]
        bag_preds, labels = [], []
        for cls_pred, tgt_pred in zip(cls_preds, batch_targets):
            cls_pred = cls_pred[tgt_pred[:,0] > 0]              # cls_pred shape: [points_num, class_num] -> [positive, class_num]
            if cls_pred.shape[0] > 0:
                bag_pred, _ = torch.max(cls_pred, dim=0)        # bag_pred shape: [class_num]
            else:
                bag_pred = torch.tensor([1e-4], device=device).repeat(num_classes)   
            
            bag_pred_new = torch.zeros(num_classes, device=device)
            if max(bag_pred[1:]) > 0.5:
                bag_pred_new[0] = bag_pred[0]*(1 - max(bag_pred[1:]))
            else:
                bag_pred_new[0] = bag_pred[0]
            for i in range(1, num_classes):
                bag_pred_new[i] = bag_pred[i]

            label, _ = torch.max(tgt_pred, dim=0)                # label shape: [1]
            label = torch.clamp(label.long()-1, min=0)
            bag_preds.append(bag_pred_new)
            labels.append(label)
        bag_preds = torch.stack(bag_preds) # bag_preds: [batch_num, class_num]
        labels = torch.stack(labels).squeeze(1) # labels: [batch_num]
        bag_loss = self.focal_loss(bag_preds, labels)

        return bag_loss

    def compute_batch_bag_loss2(self, preds, batch_targets, batch_num):
        """
        cls_preds shape: [batch_num*points_num, 7] -> [batch_num, points_num, 7]
        msk_preds shape: [positive_num, 224, 224]
        batch_targets shape: [batch_num*points_num, 8+S*S] -> [batch_num, points_num, 1]
        """

        device = preds["cls"].device
        num_classes = preds["cls"].shape[1]
        cls_preds = preds["cls"].view(batch_num, -1, preds["cls"].shape[-1])
        #ctr_preds = preds["ctr"].view(batch_num, -1, preds["ctr"].shape[-1])
        #cls_preds = cls_preds * ctr_preds
        batch_targets = batch_targets.view(batch_num, -1, batch_targets.shape[-1])[:, :, 4:5]
        bag_preds, labels = [], []
        for cls_pred, tgt_pred in zip(cls_preds, batch_targets):
            cls_pred = cls_pred[tgt_pred[:,0] > 0]              # cls_pred shape: [points_num, class_num] -> [positive, class_num]
            if cls_pred.shape[0] > 0:
                bag_pred, _ = torch.max(cls_pred, dim=0)        # bag_pred shape: [class_num]
            else:
                bag_pred = torch.tensor([1e-4], device=device).repeat(num_classes)   
            
            bag_pred_new = torch.zeros(num_classes, device=device)
            if max(bag_pred[1:]) > 0.99:
                sum_p = torch.max(bag_pred[1:]) # SUM or Max ???
                p = F.softmax(torch.tensor([bag_pred[0], sum_p]), dim=0)
                bag_pred_new[0] = p[0]
            else:
                bag_pred_new[0] = bag_pred[0]
            for i in range(1, num_classes):
                bag_pred_new[i] = bag_pred[i]# / sum_p * p[1]
            """else:
                bag_pred_new = F.softmax(bag_pred, dim=0)"""

            label,_ = torch.max(tgt_pred, dim=0)                # label shape: [1]
            label = torch.clamp(label.long()-1, min=0)
            bag_preds.append(bag_pred_new)
            labels.append(label)
        bag_preds = torch.stack(bag_preds) # bag_preds: [batch_num, class_num]
        labels = torch.stack(labels).squeeze(1) # labels: [batch_num]
        bag_loss = self.focal_loss(bag_preds, labels)

        return bag_loss

    def compute_batch_bag_loss3(self, preds, batch_targets, batch_num):
        """
        用以生成肿瘤恶性程度的可解释性评分标准
        """
        device = preds["cls"].device
        num_classes = preds["cls"].shape[1]
        cls_preds = preds["cls"].view(batch_num, -1, preds["cls"].shape[-1])
        ctr_preds = preds["ctr"].view(batch_num, -1, preds["ctr"].shape[-1])
        cls_preds = cls_preds * ctr_preds
        batch_targets = batch_targets.view(batch_num, -1, batch_targets.shape[-1])[:, :, 4:5]
        bag_preds, labels = [], []
        for cls_pred, tgt_pred in zip(cls_preds, batch_targets):
            cls_pred = cls_pred[tgt_pred[:,0] > 0]
            if cls_pred.shape[0] > 0:
                    bag_pred, _ = torch.max(cls_pred, dim=0)
            else:
                    bag_pred = torch.tensor([1e-4], device=device).repeat(num_classes)
            bag_preds.append(bag_pred)
            label,_ = torch.max(tgt_pred, dim=0)
            label = torch.clamp(label.long()-1, min=0)
            labels.append(label)
        bag_preds = torch.stack(bag_preds) # [B, class_num]
        labels = torch.stack(labels).squeeze(1) # [B]
        
        roundness = preds["roundness"] # [B, 1]
        echoness = preds["echoness"] # [B, 3]
        input = torch.cat((bag_preds, roundness, echoness), dim=1) # [B, class_num+4]
        self.logistic = self.logistic.to(device)
        bag_preds = self.logistic(input) # # [B, class_num+4] -> [B, 1]
        #bag_loss = self.focal_loss(bag_preds, labels)
        labels = torch.where(labels > 0, 1.0, 0.0)
        bag_loss = F.mse_loss(bag_preds.squeeze(1), labels, reduction='mean')
        return bag_loss

    def compute_batch_bag_loss4(self, preds, batch_targets, batch_num):
        batch_targets = batch_targets.view(batch_num, -1, batch_targets.shape[-1])[:, :, 4:5]
        labels = []
        for tgt_pred in batch_targets:
            label,_ = torch.max(tgt_pred, dim=0)                # label shape: [1]
            label = torch.clamp(label.long()-1, min=0)
            labels.append(label)
        labels = torch.stack(labels).squeeze(1) # labels: [batch_num]
        bag_loss = self.focal_loss(preds["lab"], labels)
        return bag_loss

    def get_batch_position_annotations(self, cls_heads, reg_heads, ctr_heads, atn_heads,
                                       batch_positions, annotations, use_center_sample=True):
        '''
        Assign a ground truth target for each position on feature map
        '''
        
        device = annotations.device
        batch_mi, batch_stride = [], []
        for reg_head, mi, stride in zip(reg_heads, self.mi, self.strides):
            mi = torch.tensor(mi).to(device)
            B, H, W, _ = reg_head.shape
            per_level_mi = torch.zeros(B, H, W, 2).to(device)
            per_level_mi = per_level_mi + mi
            batch_mi.append(per_level_mi)
            per_level_stride = torch.zeros(B, H, W, 1).to(device)
            per_level_stride = per_level_stride + stride
            batch_stride.append(per_level_stride)

        cls_preds,reg_preds,ctr_preds,atn_preds,all_points_position,all_points_mi,all_points_stride=[],[],[],[],[],[],[]
        for cls_pred, reg_pred, ctr_pred, atn_pred, per_level_position, per_level_mi, per_level_stride in zip(
                cls_heads, reg_heads, ctr_heads, atn_heads, batch_positions, batch_mi, batch_stride):
            cls_pred = cls_pred.view(cls_pred.shape[0], -1, cls_pred.shape[-1])
            reg_pred = reg_pred.view(reg_pred.shape[0], -1, reg_pred.shape[-1])
            ctr_pred = ctr_pred.view(ctr_pred.shape[0], -1, ctr_pred.shape[-1])
            atn_pred = atn_pred.view(atn_pred.shape[0], -1, atn_pred.shape[-1])
            per_level_position = per_level_position.view(per_level_position.shape[0], -1, per_level_position.shape[-1])
            per_level_mi = per_level_mi.view(per_level_mi.shape[0], -1, per_level_mi.shape[-1])
            per_level_stride = per_level_stride.view(per_level_stride.shape[0], -1, per_level_stride.shape[-1])

            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
            ctr_preds.append(ctr_pred)
            atn_preds.append(atn_pred)
            all_points_position.append(per_level_position)
            all_points_mi.append(per_level_mi)
            all_points_stride.append(per_level_stride)

        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)
        ctr_preds = torch.cat(ctr_preds, dim=1)
        atn_preds = torch.cat(atn_preds, dim=1)
        all_points_position = torch.cat(all_points_position, dim=1)
        all_points_mi = torch.cat(all_points_mi, dim=1)
        all_points_stride = torch.cat(all_points_stride, dim=1)

        batch_targets, mask_targets = [],[]
        for per_image_position, per_image_mi, per_image_stride, per_image_annotations in zip(
                all_points_position, all_points_mi, all_points_stride, annotations):
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
        batch_targets = torch.cat([batch_targets, all_points_position, mask_targets], dim=2)

        # batch_targets shape:[batch_size, points_num, 8+S*S]
        # 8:    l,t,r,b,class_index,center-ness_gt,point_ctr_x,point_ctr_y
        # S*S:  mask map
        return cls_preds, reg_preds, ctr_preds, atn_preds, batch_targets