import torch, math, sys
# `references` is the torchvision detection helper package, which is not
# installable from PyPI; the three helpers used here are vendored in metrics.py.
from . import metrics as utils
from tqdm import tqdm
from typing import Dict
from ..loss.loss import MMDLoss, BatchSpectralShrinkage

def trainer(args, model, loss_function, optimizer, data_loader, device, epoch, print_freq, delta_list=None) -> Dict[str, float]:
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    mmd = MMDLoss()
    bss = BatchSpectralShrinkage()
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    loss_total = {"cls":0.0, "reg":0.0, "ctr":0.0, "msk":0.0, "bag":0.0}
    length_total = 0.0
    
    #### ========== Train Step ========== ####
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        images = torch.stack(images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if epoch > args.epoch_qp: # 分组
            subgroup_images = {}
            subgroup_targets = {}
            for target,image in zip(targets, images):
                key = target["labels"].item()
                if key not in subgroup_images.keys():
                    subgroup_images[key] = []
                    subgroup_targets[key] = []
                    subgroup_images[key].append(image)
                    subgroup_targets[key].append(target)
                else:
                    subgroup_images[key].append(image)
                    subgroup_targets[key].append(target)
            
            loss_dict = {"cls":0.0, "reg":0.0, "ctr":0.0, "msk":0.0, "bag":0.0, "bss":0.0, "mmd":0.0}
            features_all = []
            features_sub = {}
            for key in subgroup_images.keys():
                images = torch.stack(subgroup_images[key])
                targets = subgroup_targets[key]
                with torch.cuda.amp.autocast(enabled=False):
                    predictions_dict, batch_annotations= model(images, targets, epoch)
                    sub_loss_dict = loss_function(predictions_dict, batch_annotations)
                    for mask_key in sub_loss_dict.keys():
                        loss_dict[mask_key] += sub_loss_dict[mask_key] * delta_list[key]
                    features_all.append(predictions_dict['fea'])
                    if key not in features_sub.keys():
                        features_sub[key] = []
                    features_sub[key].append(predictions_dict['fea'])
                    
            # Loss_BSS
            features_all = torch.cat(features_all)
            features_all = features_all.reshape(features_all.shape[0], -1)
            loss_dict["bss"] = bss(feature = features_all)
            # Loss_MMD
            mmd_list = []
            for key in features_sub.keys():
                features_sub[key] = torch.cat(features_sub[key])
                features_sub[key] = features_sub[key].reshape(features_sub[key].shape[0], -1)
                mmd_list.append(mmd(source = features_all, target = features_sub[key]))
            loss_dict["mmd"] = torch.mean(torch.tensor(mmd_list, requires_grad=True).nan_to_num(nan=0.0))
        else: # 不分组
            predictions_dict, batch_annotations= model(images, targets, epoch)
            loss_dict = loss_function(predictions_dict, batch_annotations)

        
        losses = sum(loss for loss in loss_dict.values())
        for key in loss_total.keys():
            loss_total[key] += loss_dict[key].item() * len(targets)
        length_total = length_total + len(targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    for key in loss_total.keys():
        loss_total[key] /= length_total
    
    #### ========== Smooth Step ========== ####
    """if epoch >= 150:
        print(f"Create Epoch [{epoch}] features of all training data...")
        all_features, all_labels = [[],[],[],[],[]], []
        with torch.no_grad():
            for images, targets in tqdm(data_loader, ncols=100, leave=False):
                images = list(image.to(device) for image in images)
                images = torch.stack(images)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                predictions_dict, batch_annotations, feature_list, labels = model(images, targets, epoch)
                #
                all_labels.append(labels)
                for i in range(5):
                    all_features[i].append(feature_list[i])
    
        all_labels = torch.cat(all_labels, dim=1).squeeze(0)
        for i in range(5):
            all_features[i] = torch.cat(all_features[i])
            model.head.fds[i].update_last_epoch_stats(epoch)
            model.head.fds[i].update_running_stats(all_features[i], all_labels, epoch)"""
    
    return loss_total