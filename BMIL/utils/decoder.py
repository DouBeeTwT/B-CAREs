import torch, math
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from torchvision.ops import nms
from ..loss.loss import FCOSPositions  # the class lives in BMIL/loss/loss.py;
# the original file imported it from a top-level FCOS_loss module
from typing import List, Dict, Callable
from ..model.block import Blender, blender

class DecodeMethod:
    def __init__(self,
                 max_object_num:int=100,
                 min_score_threshold:float=0.05,
                 topn:int=1000,
                 nms_threshold:float=0.5,
                 base_num = 4,
                 attention_size = 14,
                 blender_size = 28,
                 class_num = 7):
        self.max_object_num = max_object_num
        self.min_score_threshold = min_score_threshold
        self.topn = topn
        self.nms_threshold=nms_threshold
        self.K = base_num
        self.M = attention_size
        self.R = blender_size
        self.class_num = class_num

    def __call__(self, preds:Dict[str, Tensor]) -> Dict[str, Tensor]:
        # preds = {"cls","reg","ctr","pos","atn","map", "scores", "labels", "bboxes"}
        device = preds["scores"].device
        batch_size = preds["scores"].shape[0]
        B, K, W, H = preds["map"].shape
        batch_scores = torch.ones((batch_size, self.max_object_num), device=device) * (-1)
        batch_labels = torch.ones((batch_size, self.max_object_num), device=device) * (-1)
        batch_bboxes = torch.zeros((batch_size, self.max_object_num, 4), device=device)
        batch_attens = torch.zeros((B, self.max_object_num, self.K*self.M*self.M), device=device)
        batch_features = torch.zeros((batch_size, self.max_object_num, 256), device=device)
        batch_cls_prob = torch.zeros((batch_size, self.max_object_num, self.class_num), device=device)

        for i, (scores, labels, bboxes, attens, features, class_prob) in enumerate(
            zip(preds["scores"], preds["labels"], preds["bboxes"], preds["atn"], preds["fea"], preds["cls"])):
            labels = labels[scores > self.min_score_threshold]
            bboxes = bboxes[scores > self.min_score_threshold]
            attens = attens[scores > self.min_score_threshold]
            features = features[scores > self.min_score_threshold]
            class_prob = class_prob[scores > self.min_score_threshold]
            scores = scores[scores > self.min_score_threshold]
            

            if scores.shape[0] != 0:
                index = torch.argsort(-scores)
                scores = scores[index]
                labels = labels[index]
                bboxes = bboxes[index]
                attens = attens[index]
                features = features[index]
                class_prob = class_prob[index]

                if self.topn < scores.shape[0]:
                    scores = scores[0:self.topn]
                    labels = labels[0:self.topn]
                    bboxes = bboxes[0:self.topn]
                    attens = attens[0:self.topn]
                    features = features[0:self.topn]
                    class_prob = class_prob[0:self.topn]
                    
                # nms
                keep = nms(bboxes, scores, self.nms_threshold)
                scores = scores[keep]
                labels = labels[keep]
                bboxes = bboxes[keep]
                attens = attens[keep]
                features = features[keep]
                class_prob = class_prob[keep]

                final_detection_num = min(self.max_object_num, scores.shape[0])

                batch_scores[i, 0:final_detection_num]    = scores[0:final_detection_num]
                batch_labels[i, 0:final_detection_num]    = labels[0:final_detection_num]
                batch_bboxes[i, 0:final_detection_num, :] = bboxes[0:final_detection_num, :]
                batch_attens[i, 0:final_detection_num, :] = attens[0:final_detection_num, :]
                batch_features[i, 0:final_detection_num, :] = features[0:final_detection_num,:]
                batch_cls_prob[i, 0:final_detection_num, :] = class_prob[0:final_detection_num, :]

        preds["scores"] = batch_scores      # batch_scores shape:   [batch_size, max_object_num]
        preds["labels"] = batch_labels      # batch_classes shape:  [batch_size, max_object_num]
        preds["bboxes"] = batch_bboxes      # batch_bboxes shape:   [batch_size, max_object_num, 4]
        preds["atn"]    = batch_attens      # batch_attens shape:   [batch_size, max_object_num, K*4*4]
        preds["fea"]    = batch_features    # batch_features shape: [batch_size, max_object_num, 256]
        preds["cls"]    = batch_cls_prob    # batch_cls_prob shape: [batch_size, max_object_num, 7]

        return preds

class FCOSDecoder:
    def __init__(self, args=None,
                 max_object_num = 7,
                 score_threshold = 0.05,
                 topn = 7,
                 nms_threshold = 0.6,
                 base_num = 4,
                 attention_size = 14,
                 blender_size = 56,
                 figure_size = 224,
                 class_num = 7):
        if args is None:
            self.max_object_num = max_object_num
            self.min_score_threshold = score_threshold
            self.topn = topn
            self.nms_threshold = nms_threshold
            self.K = base_num
            self.M = attention_size
            self.R = blender_size
            self.figure_size = figure_size
            self.class_num = class_num
        else:
            self.max_object_num = args.max_object_num
            self.min_score_threshold = args.score_threshold
            self.topn = args.topn
            self.nms_threshold = args.nms_threshold
            self.K = args.base_num
            self.M = args.attention_size
            self.R = args.blender_size
            self.figure_size = args.figure_size
            self.class_num = args.class_num

        self.decode_function = DecodeMethod(
            max_object_num=self.max_object_num,
            min_score_threshold=self.min_score_threshold,
            topn=self.topn,
            nms_threshold=self.nms_threshold,
            base_num=self.K,
            attention_size=self.M,
            blender_size=self.R,
            class_num=self.class_num)
        self.blender = blender(self.R, self.K, self.M)

    def __call__(self, preds:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # preds = {"cls","reg","ctr","pos","atn","map"}
        # preds["cls"]  shape [batch_size, points_num, class_num]
        # preds["ctr"]  shape [batch_size, points_num, 1]


        preds["scores"], preds["labels"] = torch.max(preds["cls"], axis=2)
        preds["scores"] = torch.sqrt(preds["scores"] * preds["ctr"].squeeze(-1)).type(torch.float32) # [batch_size, points_num]       
        preds["bboxes"] = self.snap_ltrb_to_x1y1x2y2(preds["reg"], preds["pos"]).type(torch.float32)
        # preds = {"cls","reg","ctr","pos","atn","map", "scores", "labels", "bboxes"}

        preds = self.decode_function(preds)
        # preds["atn"].shape    [32, 7, 4*14*14]
        # preds["map"].shape    [32, 4, 224, 224]
        # preds["bboxes"].shape [32, 7, 4]
        preds["bboxes"] = torch.clamp(preds["bboxes"], min=0.0, max=self.figure_size)
        

        masks = []
        for atn, bases, pd_boxes in zip(preds["atn"], preds["map"], preds["bboxes"]):
            atn = atn.reshape(self.max_object_num, self.K, self.M, self.M)
            bases = bases.unsqueeze(0) 
            one_mask = []
            for one_atn, one_box in zip(atn, pd_boxes):
                #pd_box_ind = torch.arange(0, pd_boxes.shape[0], dtype=torch.float32, device=pd_boxes.device).unsqueeze(1)\
                one_atn = one_atn.unsqueeze(0)
                one_pd_box = torch.cat([torch.tensor([0.0], device=pd_boxes.device), one_box]).unsqueeze(0)
                one_mask_pd = self.blender(one_atn, bases, pd_boxes=one_pd_box) #[1, 28, 28]
                one_mask.append(one_mask_pd.squeeze(0))
            one_mask = torch.stack(one_mask)
            masks.append(one_mask)
        preds["msk"] = torch.stack(masks)
        #preds["msk"] = preds["msk"].reshape(-1, preds["msk"].shape[-2], preds["msk"].shape[-1])
        
        #preds = self.create_maskes(preds)
        # preds = {"cls","reg","ctr","pos","atn","map", "scores", "labels", "bboxes", "msk"}
        
        """roundness = self.calc_roundness(preds["msk"], device=preds["msk"].device) # [batch_size*base_size, 1]
        echoness = self.calc_echoness(preds["msk"], images[:,0,:,:]) # [batch_size*base_size, 3]
        B, K, C = preds["logist"].shape
        input = torch.cat((preds["logist"].reshape(-1, C), roundness, echoness), dim=1) # [batch_size*base_size, class_num+4]
        self.logistic = self.logistic.to(input.device)
        logistic = self.logistic(input) # # [batch_size*base_size, class_num+4] -> [batch_size*base_size, class_num]
        preds["logistic"] = logistic.reshape(B, K, -1)"""
        
        # preds["scores"] shape: [batch_size, max_object_num]
        # preds["labels"] shape: [batch_size, max_object_num]
        # preds["bboxes"] shape: [batch_size, max_object_num, 4]
        # preds["logistic"] shape: [batch_size, max_object_num, class_num]
        # preds["msk"] shape: [batch_size, max_object_num, M, M]
        return preds
    
    def snap_ltrb_to_x1y1x2y2(self, preds_reg:Tensor, preds_pos:Tensor) -> Tensor:
        '''
        snap reg preds to pred bboxes
        preds_reg: [batch_size, point_nums, 4]   4:[l,t,r,b]
        preds_pos: [batch_size, point_nums, 2]   2:[point_ctr_x,point_ctr_y]
        '''
        #preds_reg = torch.clamp(preds_reg, min=1e-4)
        #preds_reg = torch.exp(preds_reg)
        preds_reg = preds_reg * (self.figure_size /2.0)
        #preds_reg = torch.clamp(preds_reg, min=0, max=self.figure_size)
        pred_bboxes_xy_min = preds_pos - preds_reg[:, :, 0:2]
        pred_bboxes_xy_max = preds_pos + preds_reg[:, :, 2:4]
        pred_bboxes = torch.concat([pred_bboxes_xy_min, pred_bboxes_xy_max], dim=2).type(dtype=torch.float32)
        pred_bboxes = torch.clamp(pred_bboxes, min=0, max=self.figure_size)
        return pred_bboxes
    
    def create_maskes(self, preds:Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Input:
            preds: {"cls", "reg", "ctr", "pos", "atn", "map", ...}
                    preds["scores"] shape [batch_size, max_object_num]
                    preds["atn"] shape [B, N, K*W*H]
                    preds["map"] shape [B, K, W, H]
        Output:
            preds: {..., "msk"}
                    preds["msk"] shape [B*N, W, H]
        """
        preds["msk"] = []
        for atn, map in zip(preds["atn"], preds["map"]):
            msk = self.blender(atn, map)
            preds["msk"].append(msk)
        preds["msk"] = torch.concat(preds["msk"])

        return preds