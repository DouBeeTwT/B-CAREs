import argparse
import torch
import torch.distributed as dist
from torch.nn import parallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
torch.autograd.set_detect_anomaly(True)
import torch.utils.data
import torchvision.transforms as T
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import BMIL
from BMIL.data.dataset import collate_fn_coco, COCODataset
import os, wandb
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import math
import BMIL.utils

# === #
parser = argparse.ArgumentParser(description="")
# TRAIN ARGS
parser.add_argument('--root',           default="./Data/",   type=str,   help="The root pathway of Dataset")
parser.add_argument('--device',         default="cuda:4",    type=str,   help="Choose which GPU")
parser.add_argument('--world_size',     default=1,           type=int,   help="Number of GPUs")
parser.add_argument('--batch_size',     default=32,          type=int,   help="Batch size")
parser.add_argument('--learning_rate',  default=1e-3,        type=int,   help="Batch size")
parser.add_argument('--epoch_max',      default=150,         type=int,   help="Max epoch for trainig")
parser.add_argument('--epoch_qp',       default=120,         type=int,   help="Start epoch for QP")
# MODEL ARGS
parser.add_argument('--in_channels',    default=3,           type=int,   help="Channels of input layer")
parser.add_argument('--hidden_channels',default=256,         type=int,   help="Channels of hidden layer")
parser.add_argument('--class_num',      default=7,           type=int,   help="Number of prediction classes")
parser.add_argument('--figure_size',    default=224,         type=int,   help="Figure size")
parser.add_argument('--neck_layers',    default=4,           type=int,   help="Public conv layers for neck")
parser.add_argument('--need_P2',        default=False,       type=bool,  help="Whether need P2")
parser.add_argument('--need_P6',        default=True,        type=bool,  help="Whether need P6")
parser.add_argument('--need_P7',        default=True,        type=bool,  help="Whether need P7")
parser.add_argument('--mi_size',        default=32,          type=int,   help="")
parser.add_argument('--stride',         default=8,           type=int,   help="")
parser.add_argument('--use_sample',     default=True,        type=bool,  help="")
parser.add_argument('--sample_radius',  default=1.5,         type=float, help="")
parser.add_argument('--backbone_name',  default="StarNet",   type=str,   help="Choose backbone by name")
parser.add_argument('--protonet_name',  default="DeepLabV3", type=str,   help="Choose protonet by name")
# LOSSES $ DECODER ARGS
parser.add_argument('--loss_weight',    default=[1., 1., 1., 1., 1.], nargs='+', type=float, help="losses weights")
parser.add_argument('--alpha_cls',      default=0.75,        type=float, help="The alpha for Loss_cls")
parser.add_argument('--gamma_cls',      default=2,           type=int,   help="The gamma for Loss_cls")
parser.add_argument('--iou_name',       default="CIoU",      type=str,   help="Choose Iou Method by name")
parser.add_argument('--base_num',       default=4,           type=int,   help="Number of base scroe maps")
parser.add_argument('--blender_size',   default=56,          type=int,   help="The size of blender")
parser.add_argument('--attention_size', default=14,          type=int,   help="The size of attention")
parser.add_argument('--alpha_bag',      default=None,        type=float, help="The alpha for Loss_bag", nargs='+')
parser.add_argument('--gamma_bag',      default=2,           type=int,   help="The gamma for Loss_bag")
parser.add_argument('--score_threshold',default=0.05,        type=float, help="score threshold")
parser.add_argument('--nms_threshold',  default=0.6,         type=float, help="nms threshold")
parser.add_argument('--max_object_num', default=7,           type=int,   help="max_object_num")
parser.add_argument('--topn',           default=50,          type=int,   help="topn")
# WAND ARGS
parser.add_argument("--wandb",          action="store_true")
parser.add_argument('--project_name',   default="QPMC_BIRADS",  type=str, help="project name")
parser.add_argument('--exp_name',       default="Exp001",   type=str, help="experiment name")
parser.add_argument('--seed',           default=1,          type=int, help="Random seed")
args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# === #

model = BMIL.model.BMIL(args = args).to(args.device)

loss_function = BMIL.loss.FCOSLoss(args = args)
decoder = BMIL.utils.decoder.FCOSDecoder(args = args)

data_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize([.2,.2,.2],[.08,.08,.08])
])

dataset_train = COCODataset(root=os.path.join(args.root,'train'), transforms=data_transforms)
data_loader = torch.utils.data.DataLoader(dataset_train,
    batch_size=args.batch_size, shuffle=True,
    collate_fn=collate_fn_coco) # collate_fn是取样本的方法参数

dataset_test =  COCODataset(root=os.path.join(args.root,'val'), transforms=data_transforms)
data_loader_test = torch.utils.data.DataLoader(dataset_test,
    batch_size=args.batch_size, shuffle=False,
    collate_fn=collate_fn_coco)

params = [p for p in model.parameters() if p.requires_grad]
#optimizer = SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
optimizer = Adam(params, lr=args.learning_rate, weight_decay=0.0005)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch_max, eta_min=args.learning_rate/100)

# train
wandb_logger = wandb.init(project=args.project_name, name=args.exp_name) if args.wandb else None
try:
    os.mkdir("pth/{}_{}".format(args.project_name, args.exp_name))
except:
    print("Dir pth/{}_{} already exsist".format(args.project_name, args.exp_name))
acc_max = 0.0
delta_list = [1.0 for _ in range(args.class_num)]
for epoch in range(args.epoch_max):
        loss_total = BMIL.utils.trainer(args, model, loss_function, optimizer, data_loader, args.device, epoch, print_freq=20, delta_list=delta_list)
        # Quasi-Pareto
        if epoch > args.epoch_qp:
            preds, acc, rec, f1, cm = BMIL.utils.tester(model, decoder, data_loader, args.device, epoch, args.class_num)
            for i in range(len(delta_list)):
                delta_list[i] = 1.0 / (1+math.exp((cm[i,i]/sum(cm[i])-acc)/2))
            delta_list = [0.5 if math.isnan(x) else x for x in delta_list]
        # Normal
        preds, acc, rec, f1, cm = BMIL.utils.tester(model, decoder, data_loader_test, args.device, epoch, args.class_num)
        lr_scheduler.step()

        wandb.log({"train/loss":sum(loss for loss in loss_total.values()),
                    "train/cls":loss_total["cls"],
                    "train/reg":loss_total["reg"],
                    "train/ctr":loss_total["ctr"],
                    "train/msk":loss_total["msk"],
                    "train/bag":loss_total["bag"],
                    "val/ACC":acc,
                    "val/REC":rec,
                    "val/F1":f1},
                    step=epoch) if args.wandb else None
        torch.save(model.state_dict(), "pth/{}_{}/Last.pth".format(args.project_name, args.exp_name))
        if acc >= acc_max:
            acc_max =acc
            torch.save(model.state_dict(), "pth/{}_{}/Best.pth".format(args.project_name, args.exp_name))