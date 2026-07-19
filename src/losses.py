"""Combined BCE-with-logits + Dice loss, plus Dice / IoU metrics."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0, smooth: float = 1.0):
        super().__init__()
        self.register_buffer('pw', torch.tensor(pos_weight))
        self.smooth = smooth

    def forward(self, logits, target):
        bce   = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pw)
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(1,2,3))
        denom = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice  = 1 - (2*inter + self.smooth) / (denom + self.smooth)
        return bce + dice.mean()


def dice_score(logits, target, thresh: float = 0.5, smooth: float = 1.0):
    pred  = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter + smooth) / (denom + smooth)).mean().item()


def iou_score(logits, target, thresh: float = 0.5, smooth: float = 1e-6):
    pred  = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred*target).sum(dim=(1,2,3))
    return ((inter + smooth) / (union + smooth)).mean().item()
