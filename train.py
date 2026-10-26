"""Train the B-CAREs multi-task network.

Examples
--------
    python train.py --config configs/train_multitask.yaml
    python train.py --config configs/train_multitask.yaml --batch_size 16
    python train.py --root ./Data --device cuda:0 --class_num 7 --backbone_name StarNet
    python train.py --config configs/train_multitask.yaml --dry-run   # smoke test

The YAML file supplies defaults; any flag given on the command line overrides it.
`--dry-run` builds the data pipeline and the model, runs a single forward pass and
exits without training or writing checkpoints.

Expected data layout (see docs/DATA_FORMAT.md):
    <root>/train/...   COCO-style split used for optimisation
    <root>/val/...     COCO-style split used for model selection

Checkpoints are written to pth/<project_name>_<exp_name>/{Last,Best}.pth.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.utils.data
import torchvision.transforms as T
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from bcares.config import TRAIN_KEYMAP, parse_with_config, resolve_device, str2bool

BMIL_MISSING = """
The `BMIL` package is not importable.

`train.py` needs the model / loss / decoder implementation that lives in the
`BMIL/` directory at the root of this repository (and of your working copy).
If you cloned only part of the repository, or run the script from another
directory, either

  * run from the repository root:      cd B-CAREs && python train.py ...
  * or install the package first:      pip install -e .

Run `python train.py --config configs/train_multitask.yaml --dry-run` to check
that the environment is wired correctly.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train B-CAREs (multi-task BI-RADS / subtype / SLN network).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="YAML configuration file")

    # --- data -------------------------------------------------------------
    parser.add_argument("--root", default="./Data/", type=str, help="root path of the dataset")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--num_workers", default=8, type=int, help="DataLoader workers")

    # --- runtime ----------------------------------------------------------
    parser.add_argument("--device", default="auto", type=str, help="auto | cpu | cuda:0 | cuda:1 ...")
    parser.add_argument("--world_size", default=1, type=int, help="number of GPUs (distributed training)")
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument("--debug_anomaly", default=False, type=str2bool, help="torch autograd anomaly mode (slow)")

    # --- optimisation -----------------------------------------------------
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Adam weight decay")
    parser.add_argument("--epoch_max", default=150, type=int, help="total number of epochs")
    parser.add_argument("--epoch_qp", default=120, type=int, help="first epoch of quasi-Pareto re-weighting")

    # --- model ------------------------------------------------------------
    parser.add_argument("--in_channels", default=3, type=int, help="input channels")
    parser.add_argument("--hidden_channels", default=256, type=int, help="hidden channels")
    parser.add_argument("--class_num", default=7, type=int, help="number of classes (7 subtype / 5 BI-RADS / 3 SLN)")
    parser.add_argument("--figure_size", default=224, type=int, help="input image size")
    parser.add_argument("--neck_layers", default=4, type=int, help="shared conv layers in the neck")
    parser.add_argument("--need_P2", default=False, type=str2bool, help="use the P2 level")
    parser.add_argument("--need_P6", default=True, type=str2bool, help="use the P6 level")
    parser.add_argument("--need_P7", default=True, type=str2bool, help="use the P7 level")
    parser.add_argument("--mi_size", default=32, type=int, help="mask/instance size")
    parser.add_argument("--stride", default=8, type=int, help="feature stride")
    parser.add_argument("--use_sample", default=True, type=str2bool, help="sample points around the lesion")
    parser.add_argument("--sample_radius", default=1.5, type=float, help="sampling radius")
    parser.add_argument("--backbone_name", default="StarNet", type=str, help="backbone name")
    parser.add_argument("--protonet_name", default="DeepLabV3", type=str, help="prototype network name")

    # --- losses and decoder ----------------------------------------------
    parser.add_argument("--loss_weight", default=[1.0, 1.0, 1.0, 1.0, 1.0], nargs="+", type=float,
                        help="alpha_1..alpha_5 of Eq. (2)")
    parser.add_argument("--alpha_cls", default=0.75, type=float, help="focal-loss alpha")
    parser.add_argument("--gamma_cls", default=2, type=int, help="focal-loss gamma")
    parser.add_argument("--iou_name", default="CIoU", type=str, help="IoU variant")
    parser.add_argument("--base_num", default=4, type=int, help="number of base score maps")
    parser.add_argument("--blender_size", default=56, type=int, help="blender size")
    parser.add_argument("--attention_size", default=14, type=int, help="attention size")
    parser.add_argument("--alpha_bag", default=None, type=float, nargs="+", help="alpha for the bag loss")
    parser.add_argument("--gamma_bag", default=2, type=int, help="gamma for the bag loss")
    parser.add_argument("--score_threshold", default=0.05, type=float, help="score threshold")
    parser.add_argument("--nms_threshold", default=0.6, type=float, help="NMS threshold")
    parser.add_argument("--max_object_num", default=7, type=int, help="maximum number of objects")
    parser.add_argument("--topn", default=50, type=int, help="TopK candidates kept at inference")

    # --- logging ----------------------------------------------------------
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--project_name", default="QPMC_BIRADS", type=str, help="wandb project name")
    parser.add_argument("--exp_name", default="Exp001", type=str, help="experiment name")
    parser.add_argument("--dry_run", action="store_true", help="build everything, one forward pass, exit")
    return parser


def import_bmil():
    """Import the BMIL package, with an actionable error message."""
    try:
        import BMIL
        import BMIL.model  # noqa: F401
        import BMIL.loss  # noqa: F401
        import BMIL.utils.decoder  # noqa: F401
        from BMIL.data.dataset import collate_fn_coco, COCODataset
    except ImportError as exc:  # pragma: no cover - depends on checkout layout
        raise SystemExit(f"{BMIL_MISSING}\nOriginal error: {exc}") from exc

    # trainer/tester live either in BMIL.utils/__init__.py or in separate modules.
    for module in ("BMIL.utils.trainer", "BMIL.utils.tester"):
        try:
            __import__(module)
        except ImportError:
            pass
    return BMIL, collate_fn_coco, COCODataset


def build_loaders(args, collate_fn_coco, COCODataset):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.2, 0.2, 0.2], [0.08, 0.08, 0.08]),
    ])
    train_set = COCODataset(root=os.path.join(args.root, "train"), transforms=transform)
    val_set = COCODataset(root=os.path.join(args.root, "val"), transforms=transform)

    common = dict(batch_size=args.batch_size, collate_fn=collate_fn_coco,
                  num_workers=args.num_workers)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **common)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader


def dry_run(model, loss_function, train_loader, args) -> int:
    """One forward pass; prints what was verified. Never writes checkpoints."""
    print("[dry-run] batch_size={} device={} class_num={}".format(
        args.batch_size, args.device, args.class_num))
    batch = next(iter(train_loader))
    print("[dry-run] first batch type: {}".format(type(batch).__name__))
    if isinstance(batch, dict):
        print("[dry-run] batch keys: {}".format(sorted(batch)))
        tensor = batch.get("image", batch.get("img"))
    elif isinstance(batch, (list, tuple)):
        tensor = next((item for item in batch if torch.is_tensor(item)), None)
    else:
        tensor = None

    if tensor is None:
        print("[dry-run] could not identify an image tensor in the batch; "
              "data loading works, skipping the forward pass.")
        return 0

    tensor = tensor.to(args.device)
    print("[dry-run] image tensor: shape={} dtype={}".format(tuple(tensor.shape), tensor.dtype))
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
    print("[dry-run] forward pass OK; output type: {}".format(type(outputs).__name__))
    print("[dry-run] OK — data loading, model construction and forward pass work.")
    return 0


def main(argv=None) -> int:
    args = parse_with_config(build_parser(), TRAIN_KEYMAP, argv)
    args.device = resolve_device(args.device)

    if args.debug_anomaly:
        torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    BMIL, collate_fn_coco, COCODataset = import_bmil()

    model = BMIL.model.BMIL(args=args).to(args.device)
    loss_function = BMIL.loss.FCOSLoss(args=args)
    decoder = BMIL.utils.decoder.FCOSDecoder(args=args)

    train_loader, val_loader = build_loaders(args, collate_fn_coco, COCODataset)

    if args.dry_run:
        return dry_run(model, loss_function, train_loader, args)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch_max,
                                    eta_min=args.learning_rate / 100)

    if args.wandb:
        import wandb  # optional dependency

        wandb.init(project=args.project_name, name=args.exp_name)

    ckpt_dir = os.path.join("pth", "{}_{}".format(args.project_name, args.exp_name))
    os.makedirs(ckpt_dir, exist_ok=True)

    acc_max = 0.0
    delta_list = [1.0 for _ in range(args.class_num)]
    checkpoints = {
        "last": os.path.join(ckpt_dir, "Last.pth"),
        "best": os.path.join(ckpt_dir, "Best.pth"),
    }

    for epoch in range(args.epoch_max):
        loss_total = BMIL.utils.trainer(args, model, loss_function, optimizer, train_loader,
                                        args.device, epoch, print_freq=20,
                                        delta_list=delta_list)

        # Quasi-Pareto class re-weighting (Supplementary Equation 11).
        if epoch > args.epoch_qp:
            preds, acc, rec, f1, cm = BMIL.utils.tester(model, decoder, val_loader, args.device,
                                                       epoch, args.class_num)
            for i in range(len(delta_list)):
                delta_list[i] = 1.0 / (1 + math.exp((cm[i, i] / sum(cm[i]) - acc) / 2))
            delta_list = [0.5 if math.isnan(x) else x for x in delta_list]

        preds, acc, rec, f1, cm = BMIL.utils.tester(model, decoder, val_loader, args.device,
                                                   epoch, args.class_num)
        lr_scheduler.step()

        if args.wandb:
            import wandb

            wandb.log({"train/loss": sum(loss for loss in loss_total.values()),
                       "train/cls": loss_total["cls"],
                       "train/reg": loss_total["reg"],
                       "train/ctr": loss_total["ctr"],
                       "train/msk": loss_total["msk"],
                       "train/bag": loss_total["bag"],
                       "val/ACC": acc,
                       "val/REC": rec,
                       "val/F1": f1}, step=epoch)

        torch.save(model.state_dict(), checkpoints["last"])
        if acc >= acc_max:
            acc_max = acc
            torch.save(model.state_dict(), checkpoints["best"])

    print("Training finished. Best validation accuracy: {:.4f}".format(acc_max))
    print("Checkpoints: {} and {}".format(checkpoints["last"], checkpoints["best"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
