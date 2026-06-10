import torch.nn as nn
import numpy as np
import  torch

def SoftIoULoss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1
    intersection = pred * target
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)
    loss = 1 - loss.mean()
    return loss

def DiceLoss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice.mean()
    return loss

def bce_loss(pred, target, reduction='mean'):
    loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
    return loss_fn(pred, target)
