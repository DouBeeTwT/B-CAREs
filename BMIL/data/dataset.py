import torch, os
import numpy as np
from PIL import Image

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root # 数据集的根路径
        self.transforms = transforms # 数据集的预处理变形参数
        
        # 路径组合后返回该路径下的排序过的文件名（排序是为了对齐）
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))  # self.imgs  是一个全部待训练图片文件名的有序列表
        self.masks = list(sorted(os.listdir(os.path.join(root, "lab")))) # self.masks 是一个全部掩码图片文件名的有序列表
 

    # 根据idx对应读取待训练图片以及掩码图片
    def __getitem__(self, idx):
        # 根据idx针对img与mask组合路径
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        mask_path = os.path.join(self.root, "lab", self.masks[idx])
        
        
        img = Image.open(img_path).convert("RGB") # 根据路径读取三色图片并转为RGB(L)格式
        mask = Image.open(mask_path) # 根据路径读取掩码图片默认“L”格式
        mask = np.array(mask) # 将mask转为numpy格式，h*w的矩阵,每个元素是一个颜色id
        
        obj_ids = np.unique(mask) # 获取mask中的id组成列表，obj_ids=[0,1,2]
        obj_ids = obj_ids[1:] # 列表中第一个元素代表背景，不属于我们的目标检测范围，obj_ids=[1, 2, ...]
 
        # obj_ids[:,None,None]:[[[1]],[[2]]],masks(2,536,559)
        # 为每一种类别序号都生成一个布尔矩阵，标注每个元素是否属于该颜色
        masks = mask == obj_ids[:, None, None]
 
        # 为每个目标计算边界框，存入boxes
        num_objs = len(obj_ids) # 目标个数N
        boxes = [] # 边界框四个坐标的列表，维度(N,4)
        for i in range(num_objs):
            pos = np.where(masks[i]) # pos为mask[i]值为True的地方,也就是属于该颜色类别的id组成的列表
            xmin = np.min(pos[1]) # pos[1]为x坐标，x坐标的最小值
            xmax = np.max(pos[1])
            ymin = np.min(pos[0]) # pos[0]为y坐标
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        # 将boxes转化为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # 初始化类别标签
        subtype = self.imgs[idx].split("_")[0]
        if subtype == "FBB" or subtype == "A":  
            labels = torch.tensor([[0]], dtype=torch.int64) # labels[1,1] (2,)的array
        elif subtype == "BIC"or subtype == "B":
            labels = torch.tensor([[1]], dtype=torch.int64)
        elif subtype == "DICS"or subtype == "C":
            labels = torch.tensor([[2]], dtype=torch.int64)
        elif subtype == "IDC"or subtype == "D":
            labels = torch.tensor([[3]], dtype=torch.int64)
        elif subtype == "LBC"or subtype == "E":
            labels = torch.tensor([[4]], dtype=torch.int64)
        elif subtype == "PBC":
            labels = torch.tensor([[5]], dtype=torch.int64)
        elif subtype == "UDF":
            labels = torch.tensor([[6]], dtype=torch.int64)
        else:
            print(subtype)

        masks = torch.as_tensor(masks, dtype=torch.uint8) # 将masks转换为tensor
 
        # 将图片序号idx转换为tensor
        image_id = torch.tensor([idx])
        # 计算每个边界框的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 假设所有实例都不是人群
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # iscrowd[0,0] (2,)的array
 
        # 生成一个字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        # 变形transform
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
 
    def __len__(self):
        return len(self.imgs)
    

def collate_fn_coco(batch):
    return tuple(zip(*batch))