#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-01-18 22:05:50
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-22 03:06:49
FilePath     : /Documents/nnUNet/nnunetv2/training/loss/focal_loss.py
Description  :
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# taken from https://github.com/JunMa11/SegLoss/blob/master/test/nnUNetV2/loss_functions/focal_loss.py
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLossClass(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction="mean"):
        """
        Multi-class Focal Loss implementation.

        Args:
            alpha (float, list, or np.ndarray, optional): Weighting factor for each class.
                - If a single float is provided, it applies the same weight to all classes.
                - If a list or numpy array is provided, it should have a length equal to the number of classes.
                Default is 1.
            gamma (float, optional): Focusing parameter to reduce the loss contribution from easy examples.
                Default is 2.
            logits (bool, optional): If True, the inputs are expected to be raw logits and softmax will be applied.
                If False, the inputs are expected to be probabilities.
                Default is True.
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'.
                - 'mean': the sum of the output will be divided by the number of elements in the output
                - 'sum': the output will be summed.
                - 'none': no reduction will be applied.
                Default is 'mean'.
        """
        super(FocalLossClass, self).__init__()
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([alpha], dtype=torch.float32)  # Shape: [1]

    def forward(self, inputs, targets):
        device = inputs.device
        num_classes = inputs.size(1)

        # Ensure targets are 1D
        if targets.ndim > 1:
            if targets.size(1) == 1:
                targets = targets.squeeze(1)
            else:
                raise ValueError(
                    f"Targets should have shape [batch_size], but got {targets.shape}"
                )

        # Validate target range
        if targets.min() < 0 or targets.max() >= num_classes:
            raise ValueError(
                f"Targets should be in the range [0, {num_classes - 1}], but got targets.min()={targets.min()} and targets.max()={targets.max()}"
            )

        # Ensure alpha is on the same device as inputs
        if self.alpha.numel() > 1:
            if self.alpha.size(0) != num_classes:
                raise ValueError(
                    f"Alpha length ({self.alpha.size(0)}) does not match number of classes ({num_classes})"
                )
            alpha = self.alpha.to(device)  # Shape: [num_classes]
            alpha = alpha.gather(0, targets.long())  # Shape: [batch_size]
        else:
            alpha = self.alpha.to(device)  # Shape: [1]
            alpha = alpha.expand(targets.size())  # Shape: [batch_size]

        if self.logits:
            # Calculate softmax probabilities for multi-class classification
            probs = F.softmax(inputs, dim=1)  # Shape: [batch_size, num_classes]
        else:
            probs = inputs  # Assumes inputs are probabilities

        # Compute p_t (probability of the true class) using gather
        pt = probs.gather(1, targets.long().unsqueeze(1)).squeeze(
            1
        )  # Shape: [batch_size]

        # Compute focal weight
        focal_weight = alpha * (1 - pt) ** self.gamma  # Shape: [batch_size]

        # Compute focal loss
        loss = -focal_weight * torch.log(pt + 1e-8)  # Shape: [batch_size]

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # Shape: [batch_size]
