#!/usr/bin/env python
# coding=utf-8
'''
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-01-17 21:49:21
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-19 07:03:04
FilePath     : /Documents/nnUNet/nnunetv2/training/loss/deep_supervision.py
Description  : 
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import torch
from torch import nn


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), (
            "At least one weight factor should be != 0.0"
        )
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), (
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        )
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1,) * len(args[0])
        else:
            weights = self.weight_factors

        # for i, inputs in enumerate(zip(*args)):
        #     if weights[i] != 0.0:
        #         print(self.loss, weights[i], self.loss(*inputs))
        
        return sum(
            [
                weights[i] * self.loss(*inputs)
                for i, inputs in enumerate(zip(*args))
                if weights[i] != 0.0
            ]
        )
