import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss
from utils.hausdorff import HausdorffDTLoss
# from utils.boundary_loss import BoundaryLoss
import segmentation_models_pytorch as sm
from utils.boundary_loss import ABL
import monai
from utils.utils import compute_distance_map

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    # iou_score = 0
    # fbeta_score = 0
    val_loss = 0

    # criterion = nn.CrossEntropyLoss()
    criterion = sm.losses.FocalLoss('multiclass')
    # criterion = monai.losses.HausdorffDTLoss(reduction='none')
    # criterion = ABL()

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # calculate validation loss
                loss = criterion(mask_pred, mask_true)
                loss += dice_loss(
                    F.softmax(mask_pred, dim=1).float(),
                    F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
                val_loss += loss.item()

                # # BD + Dice
                # loss = 0.01*criterion(mask_pred, mask_true)
                # loss += dice_loss(
                #     F.softmax(mask_pred, dim=1).float(),
                #     F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                #     multiclass=True
                # )
                
                # convert to one-hot format
                # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                # mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2)
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2)

                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                
                # compute fbeta_score and iou_score
                # tp, fp, fn, tn = sm.metrics.get_stats(mask_pred[:, 1:], mask_true[:, 1:], mode='multilabel', threshold=0.5)
                # fbeta_score += sm.metrics.fbeta_score(tp, fp, fn, tn, beta=0.5, reduction="micro")
                # iou_score += sm.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

    net.train()

    return (val_loss / max(num_val_batches, 1), dice_score / max(num_val_batches, 1))
