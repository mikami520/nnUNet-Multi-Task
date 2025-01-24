#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-01-17 21:49:21
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-18 23:32:09
FilePath     : /Documents/nnUNet/nnunetv2/training/loss/robust_ce_loss.py
Description  :
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, size_average=None, ignore_index: int = -100, reduce=True, label_smoothing: float = 0):
        super().__init__(weight=weight, ignore_index=ignore_index, size_average=size_average, reduce=reduce, label_smoothing=label_smoothing)
        self.label_smoothing = label_smoothing
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """

    def __init__(
        self,
        weight=None,
        ignore_index: int = -100,
        k: float = 10,
        label_smoothing: float = 0,
    ):
        self.k = k
        super(TopKLoss, self).__init__(
            weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing
        )

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(
            res.view((-1,)), int(num_voxels * self.k / 100), sorted=False
        )
        return res.mean()


def compute_weighted_f1_score(logits, targets, num_classes):
    """
    Compute Weighted-Averaged F1-Score in PyTorch for multi-class classification.

    Args:
        logits (torch.Tensor): Model output logits of shape [B, num_classes].
        targets (torch.Tensor): Ground truth class labels of shape [B].
        num_classes (int): Number of classes.

    Returns:
        float: Weighted-Averaged F1-Score.
    """
    # Step 1: Convert logits to predicted classes
    predicted_classes = torch.argmax(logits, dim=1)  # Shape: [B]

    # Step 2: Initialize variables for F1 computation
    class_counts = torch.zeros(
        num_classes, dtype=torch.float32
    )  # Count of samples in each class
    true_positives = torch.zeros(num_classes, dtype=torch.float32)
    false_positives = torch.zeros(num_classes, dtype=torch.float32)
    false_negatives = torch.zeros(num_classes, dtype=torch.float32)

    # Step 3: Iterate over each class to calculate TP, FP, FN
    for c in range(num_classes):
        class_mask = targets == c
        class_counts[c] = class_mask.sum()  # Total samples for class c
        true_positives[c] = ((predicted_classes == c) & class_mask).sum()
        false_positives[c] = ((predicted_classes == c) & ~class_mask).sum()
        false_negatives[c] = ((predicted_classes != c) & class_mask).sum()

    # Step 4: Compute precision, recall, and F1 for each class
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Step 5: Compute weighted F1-Score
    weighted_f1 = (f1_per_class * class_counts).sum() / class_counts.sum()

    return weighted_f1.item()
