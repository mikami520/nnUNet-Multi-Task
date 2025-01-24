#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-01-17 21:49:21
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-19 05:47:30
FilePath     : /Documents/nnUNet/nnunetv2/training/nnUNetTrainer/variants/lr_schedule/nnUNetTrainerCosAnneal.py
Description  :
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerCosAnneal(nnUNetTrainer):
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler
