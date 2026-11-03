import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from typing import Dict
import numpy as np

def calc_bag_score(preds:Dict[str, Tensor]):
    """
    Input:
        preds["cls"]: [batch_size, points_num, class_num]
        preds["pos"]: [batch_size, points_num, 2]
        preds["msk"]: [batch_size*topk, 224, 224] -> [batch_size, topk, 224, 224] -> [batch_size, 224, 224]
    Output:
        batch_bag:    [batch_size, class_num]
    """
    device = preds["cls"].device
    num_classes = preds["cls"].shape[-1]
    preds["msk"] = preds["msk"].view(preds["cls"].shape[0], -1, preds["msk"].shape[1], preds["msk"].shape[2]).sum(dim=1)
    preds["msk"] -= preds["msk"].min()
    preds["msk"] /= preds["msk"].max()
   
    batch_bag = []
    for cls, pos, msk in zip(preds["cls"], preds["pos"], preds["msk"]):
        bag = []
        for i in range(cls.shape[0]):
            if msk[int(pos[i,0]), int(pos[i,1])]>0.6:
                bag.append(cls[i,:])
        if len(bag) == 0:
            bag.append(torch.tensor([1e-4], device=device).repeat(num_classes))
        bag_pred = torch.stack(bag)
        bag_pred, _ = torch.max(bag_pred, dim=0)

        bag_pred_new = torch.zeros(num_classes, device=device)
        if max(bag_pred[1:]) > 0.75:
            sum_p = torch.sum(bag_pred[1:]) # SUM or Max ???
            p = F.softmax(torch.tensor([bag_pred[0], sum_p]), dim=0)
            bag_pred_new[0] = p[0]
        else:
            bag_pred_new[0] = bag_pred[0]
        for i in range(1, num_classes):
            bag_pred_new[i] = bag_pred[i]# / sum_p * p[1]
        batch_bag.append(bag_pred_new)
    batch_bag = torch.stack(batch_bag)
    
    return batch_bag
    
def bag_score(labels, scores, num_classes:int) -> list:
    # create zero tensor for bag_labels
    bag_scores = torch.zeros(num_classes).to(labels.device)
    bag_scores_new = torch.zeros(num_classes).to(labels.device)
    # for all labels, get max one for each class
    #labels = torch.tensor(labels)
    #labels = labels[labels>-1].squeeze(0).int()
    #scores = torch.tensor(scores)
    #scores = scores[scores>-1].squeeze(0)
    for index in range(len(labels)):
        if scores[index] > bag_scores[labels[index]]:
            bag_scores[labels[index]] = scores[index]
    # if tumor class > 0.5, non tumor = 1 - tumor
    bag_scores_new[0] = bag_scores[0]
    if max(bag_scores[1:]) > 0.5:
        for i in range(1, num_classes):
            bag_scores_new[0] *= 1 - bag_scores[i]
            bag_scores_new[i] = bag_scores[i]
    else:
        for i in range(1, num_classes):
            bag_scores_new[i] = bag_scores[i]
    
    return list(bag_scores_new.cpu().detach().numpy())

def tester(model, decoder, data_loader, device, epoch, num_classes:int=3, accuracy:int=4):
    model.eval()

    # 制作真实标签和预测标签列表
    T_label = []
    P_label = []
    P_score = []
    for images, targets in tqdm(data_loader, ncols=100, leave=False):
        images = list(image.to(device) for image in images)
        images = torch.stack(images)
        #img, lab = img[0], lab[0]
        with torch.no_grad():
            preds, _, = model(images)
            preds = decoder(preds)
            #batch_bag_score = calc_bag_score(preds)
        
        
        """# Roi_Align
        box = preds["bboxes"][:,0,:]
        vectors = []
        for f,b in zip(feature_maps[0], box):
            b = torch.cat([torch.tensor([0.0], device=b.device), b]).unsqueeze(0)
            f = f.unsqueeze(0)
            vector = model.label_head(X=f, boxes=b)
            vectors.append(vector)
        vectors = torch.stack(vectors)
        preds["lab"] = vectors
        for one_target, one_class in zip(targets, preds["lab"]):
            T_label.append(one_target["labels"].item())
            one_class = one_class.detach().cpu().numpy()
            P_label.append(np.argmax(one_class))"""
        
        mode = "Max"
        # ACC: Max Mode
        for one_target, one_class, one_score in zip(targets, preds["labels"], preds["scores"]):
            T_label.append(one_target["labels"].item())
            if mode == "Max":
                P_label.append(one_class[0].item())
            elif mode == "Bag":
                s = bag_score(one_class, one_score, num_classes)
                P_score.append(s)
                P_label.append(np.argmax(s))
            #print(P_label)
        
        """# ACC: Max Mode
        for one_target, one_class in zip(targets, preds["lab"]):
            T_label.append(one_target["labels"].item())
            P_index = torch.argmax(one_class)
            P_label.append(P_index.item())"""

        """# ACC: Newbag Mode
        for one_target, one_score in zip(targets, batch_bag_score):
            T_label.append(one_target["labels"].item())
            score = torch.max(one_score)
            label = torch.argmax(one_score)
            P_score.append(score.item())
            P_label.append(label.item())"""               

    from sklearn.metrics import accuracy_score
    acc = round(accuracy_score(y_true=T_label, y_pred=P_label), accuracy)

    from sklearn.metrics import recall_score
    rec = round(recall_score(y_true=T_label, y_pred=P_label, average="macro"), accuracy)

    from sklearn.metrics import f1_score
    f1 = round(f1_score(y_true=T_label, y_pred=P_label, average='macro'), accuracy)

    from sklearn.metrics import roc_auc_score
    #auc = round(roc_auc_score(y_true=T_label, y_score=P_score, average="macro", multi_class="ovr"), accuracy)
    auc=0.5
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=T_label, y_pred=P_label)

    print("Epoch: [{}] | ACC: {:.4f} | REC: {:.4f} | AUC: {:.4f} | F1: {:.4f}".format(epoch, acc, rec, auc, f1))
    print(cm)

    return preds, acc, rec, f1, cm