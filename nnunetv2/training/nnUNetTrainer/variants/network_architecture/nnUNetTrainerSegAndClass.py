#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-01-17 23:57:32
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-23 18:44:12
FilePath     : /Documents/nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerSegAndClass.py
Description  : network architecture for segmentation and classification
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    isfile,
    save_json,
    maybe_mkdir_p,
)
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import (
    ContrastTransform,
    BGContrast,
)
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import (
    ApplyRandomBinaryOperatorTransform,
)
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import (
    MoveSegAsOneHotToDataTransform,
)
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import (
    Convert3DTo2DTransform,
    Convert2DTo3DTransform,
)
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import (
    ConvertSegmentationToRegionsTransform,
)
from torch import autocast, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    resample_and_save,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import (
    get_patch_size,
)
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import (
    DC_and_CE_loss,
    DC_and_BCE_loss,
    DC_and_Focal_loss,
)
from nnunetv2.training.loss.focal_loss import FocalLossClass
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import (
    get_tp_fp_fn_tn,
    MemoryEfficientSoftDiceLoss,
    MemoryEfficientSoftDiceLossNonNegative,
)
from nnunetv2.training.loss.robust_ce_loss import (
    RobustCrossEntropyLoss,
    compute_weighted_f1_score,
)
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerSegAndClass(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.initial_lr = 1e-3
        self.num_subtype_classes = 3
        self.num_epochs = 150
        self.label_smooth = 0.1
        self.save_every = 3
        self.lambda_seg = 1.0
        self.lambda_class = 1.0
        self.enable_deep_supervision = True
        self.weight_decay = 1e-4

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)
            in_channel = self.network.encoder.output_channels[-1]
            self.class_decoder = MultiScaleAttentiveClassificationDecoder(
                in_channels=in_channel,
                num_classes=self.num_subtype_classes,
                conv_name=self.configuration_manager.network_arch_conv_name,
                res_channels=in_channel,
                dropout_p=0.4,
                deep_supervision=self.enable_deep_supervision,
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)
                self.class_decoder = torch.compile(self.class_decoder)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss_seg, self.loss_class = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss_seg = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            loss_seg = DC_and_Focal_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {
                    "alpha": [0.4305, 0.2518, 0.3177],
                    "gamma": 2,
                    "smooth": 1e-5,
                },
                weight_focal=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            # loss_seg = DC_and_CE_loss(
            #     {
            #         "batch_dice": self.configuration_manager.batch_dice,
            #         "smooth": 1e-5,
            #         "do_bg": False,
            #         "ddp": self.is_ddp,
            #     },
            #     {"label_smoothing": self.label_smooth},
            #     weight_ce=1,
            #     weight_dice=1,
            #     ignore_label=self.label_manager.ignore_label,
            #     dice_class=MemoryEfficientSoftDiceLoss,
            # )

        loss_class = RobustCrossEntropyLoss(
            weight=None,
            ignore_index=self.label_manager.ignore_label
            if self.label_manager.has_ignore_label
            else -100,
            label_smoothing=self.label_smooth,
        )

        if self._do_i_compile():
            loss_seg.dc = torch.compile(loss_seg.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales_seg = self._get_deep_supervision_scales()
            deep_supervision_scales_class = self._get_compute_deep_scale_class()
            weights_seg = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales_seg))]
            )
            weights_class = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales_class))]
            )
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights_seg[-1] = 1e-6
                weights_class[-1] = 1e-6
            else:
                weights_seg[-1] = 0
                weights_class[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights_seg = weights_seg / weights_seg.sum()
            weights_class = weights_class / weights_class.sum()
            # now wrap the loss
            loss_seg = DeepSupervisionWrapper(loss_seg, weights_seg)
            loss_class = DeepSupervisionWrapper(loss_class, weights_class)

        return loss_seg, loss_class

    def _get_deep_supervision_scales(self):
        return super()._get_deep_supervision_scales()

    def _get_compute_deep_scale_class(self):
        deep_supervision_scales = list(
            [i] for i in 1 / np.cumprod(np.array([1, 2, 2]), axis=0)
        )
        return deep_supervision_scales

    def configure_optimizers(self):
        combined_parameters = [
            {
                "params": self.network.parameters(),
                "lr": self.initial_lr,
            },
            {
                "params": self.class_decoder.parameters(),
                "lr": self.initial_lr / 10,
            },
        ]
        optimizer = torch.optim.Adam(
            combined_parameters,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        self.class_decoder.train()
        # self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()
        self.class_decoder.eval()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        subtype = batch["subtype"]
        # print(data.shape)
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if isinstance(subtype, list):
            subtype = [i.to(self.device, non_blocking=True) for i in subtype]
        else:
            subtype = subtype.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            encode_feature = self.network.encoder(data)
            output = self.network.decoder(encode_feature)
            output_class = self.class_decoder(encode_feature[-1])
            # del data
            l_seg = self.loss_seg(output, target)
            l_class = self.loss_class(output_class, subtype)
            l = self.lambda_seg * l_seg + self.lambda_class * l_class

        all_params = []
        for param_group in self.optimizer.param_groups:
            all_params.extend(param_group["params"])

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # Collect all parameters from all optimizer parameter groups
            torch.nn.utils.clip_grad_norm_(all_params, 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 12)
            self.optimizer.step()

        return {
            "loss": l.detach().cpu().numpy(),
            "seg_loss": l_seg.detach().cpu().numpy(),
            "class_loss": l_class.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs["loss"])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs["loss"])
            loss_seg_here = np.mean(outputs["seg_loss"])
            loss_class_here = np.mean(outputs["class_loss"])

        self.logger.log("train_losses", loss_here, self.current_epoch)
        self.logger.log("train_seg_losses", loss_seg_here, self.current_epoch)
        self.logger.log("train_class_losses", loss_class_here, self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        subtype = batch["subtype"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if isinstance(subtype, list):
            subtype = [i.to(self.device, non_blocking=True) for i in subtype]
        else:
            subtype = subtype.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            encode_feature = self.network.encoder(data)
            output = self.network.decoder(encode_feature)
            output_class = self.class_decoder(encode_feature[-1])
            del data
            l_seg = self.loss_seg(output, target)
            l_class = self.loss_class(output_class, subtype)
            l = self.lambda_seg * l_seg + self.lambda_class * l_class

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
            output_class = output_class[0]
            subtype = subtype[0]

        weighted_f1 = compute_weighted_f1_score(
            output_class, subtype, self.num_subtype_classes
        )
        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "seg_loss": l_seg.detach().cpu().numpy(),
            "class_loss": l_class.detach().cpu().numpy(),
            "weighted_f1": weighted_f1,
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])
            loss_seg_here = np.mean(outputs_collated["seg_loss"])
            loss_class_here = np.mean(outputs_collated["class_loss"])

        weighted_f1 = np.mean(outputs_collated["weighted_f1"])
        global_dc_per_class = [
            i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        ]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log("mean_fg_dice", mean_fg_dice + weighted_f1, self.current_epoch)
        self.logger.log(
            "dice_per_class_or_region", global_dc_per_class, self.current_epoch
        )
        self.logger.log("weighted_f1", weighted_f1, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)
        self.logger.log("val_seg_losses", loss_seg_here, self.current_epoch)
        self.logger.log("val_class_losses", loss_class_here, self.current_epoch)

    def on_epoch_end(self):
        self.lr_scheduler.step()
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        self.print_to_log_file(
            "train_loss",
            np.round(self.logger.my_fantastic_logging["train_losses"][-1], decimals=4),
        )
        self.print_to_log_file(
            "train_seg_loss",
            np.round(
                self.logger.my_fantastic_logging["train_seg_losses"][-1], decimals=4
            ),
        )
        self.print_to_log_file(
            "train_class_loss",
            np.round(
                self.logger.my_fantastic_logging["train_class_losses"][-1], decimals=4
            ),
        )
        self.print_to_log_file(
            "val_loss",
            np.round(self.logger.my_fantastic_logging["val_losses"][-1], decimals=4),
        )
        self.print_to_log_file(
            "val_seg_loss",
            np.round(
                self.logger.my_fantastic_logging["val_seg_losses"][-1], decimals=4
            ),
        )
        self.print_to_log_file(
            "val_class_loss",
            np.round(
                self.logger.my_fantastic_logging["val_class_losses"][-1], decimals=4
            ),
        )
        self.print_to_log_file(
            "Weighted F1 Score",
            np.round(self.logger.my_fantastic_logging["weighted_f1"][-1], decimals=4),
        )
        self.print_to_log_file(
            "Pseudo dice",
            [
                np.round(i, decimals=4)
                for i in self.logger.my_fantastic_logging["dice_per_class_or_region"][
                    -1
                ]
            ],
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if (
            self._best_ema is None
            or self.logger.my_fantastic_logging["ema_fg_dice"][-1] > self._best_ema
        ):
            self._best_ema = self.logger.my_fantastic_logging["ema_fg_dice"][-1]
            self.print_to_log_file(
                f"Yayy! New best EMA pseudo Dice and F1 Score: {np.round(self._best_ema, decimals=4)}"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                if isinstance(self.class_decoder, OptimizedModule):
                    class_decoder = self.class_decoder._orig_mod
                else:
                    class_decoder = self.class_decoder

                checkpoint = {
                    "network_weights": mod.state_dict(),
                    "class_decoder_weights": class_decoder.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "grad_scaler_state": self.grad_scaler.state_dict()
                    if self.grad_scaler is not None
                    else None,
                    "logging": self.logger.get_checkpoint(),
                    "_best_ema": self._best_ema,
                    "current_epoch": self.current_epoch + 1,
                    "init_args": self.my_init_kwargs,
                    "trainer_name": self.__class__.__name__,
                    "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file(
                    "No checkpoint written, checkpointing is disabled"
                )

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith(
                "module."
            ):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = (
            checkpoint["inference_allowed_mirroring_axes"]
            if "inference_allowed_mirroring_axes" in checkpoint.keys()
            else self.inference_allowed_mirroring_axes
        )

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        if isinstance(self.class_decoder, OptimizedModule):
            self.class_decoder._orig_mod.load_state_dict(
                checkpoint["class_decoder_weights"]
            )
        else:
            self.class_decoder.load_state_dict(checkpoint["class_decoder_weights"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])


def get_num_groups(num_channels, default_num_groups=32):
    num_groups = min(default_num_groups, num_channels)
    while num_groups > 0:
        if num_channels % num_groups == 0:
            return num_groups
        num_groups -= 1
    return 1  # Fallback to 1 if no divisor is found


# Memory-efficient Swish activation function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
        return grad_input


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class ResidualBlock(nn.Module):
    """
    3D residual block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout_ratio=0.3,
        type="3D",
    ):
        super().__init__()
        if type == "3D":
            self.conv1 = nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
            self.gn1 = nn.GroupNorm(
                get_num_groups(out_channels), out_channels, affine=True
            )
            self.swish = MemoryEfficientSwish()

            self.conv2 = nn.Conv3d(
                out_channels, out_channels, kernel_size, stride, padding, bias=False
            )
            self.gn2 = nn.GroupNorm(
                get_num_groups(out_channels), out_channels, affine=True
            )
            self.dropout = nn.Dropout3d(p=dropout_ratio)
            # If in/out mismatch, use 1x1 conv in skip
            self.downsample = None
            if in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv3d(
                        in_channels, out_channels, kernel_size=1, stride=1, bias=False
                    ),
                    nn.GroupNorm(
                        get_num_groups(out_channels), out_channels, affine=True
                    ),
                )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
            self.gn1 = nn.GroupNorm(
                get_num_groups(out_channels), out_channels, affine=True
            )
            self.swish = MemoryEfficientSwish()

            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size, stride, padding, bias=False
            )
            self.gn2 = nn.GroupNorm(
                get_num_groups(out_channels), out_channels, affine=True
            )
            self.dropout = nn.Dropout2d(p=dropout_ratio)
            # If in/out mismatch, use 1x1 conv in skip
            self.downsample = None
            if in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=1, stride=1, bias=False
                    ),
                    nn.GroupNorm(
                        get_num_groups(out_channels), out_channels, affine=True
                    ),
                )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.swish(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.swish(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.swish(out)
        return out


class SqueezeExcitation(nn.Module):
    """
    A Squeeze-and-Excitation block for channel attention in 3D.
    """

    def __init__(self, channels, reduction=16, type="3D"):
        super().__init__()
        if type == "3D":
            self.pool = nn.AdaptiveAvgPool3d(
                (1, 1, 1)
            )  # squeeze B x C x D x H x W -> B x C x 1 x 1 x 1
        else:
            self.pool = nn.AdaptiveAvgPool2d(
                (1, 1)
            )  # squeeze B x C x H x W -> B x C x 1 x 1
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        b, c = x.shape[:2]
        # Squeeze
        y = self.pool(x).view(b, c)
        # Excitation
        y = self.fc1(y)
        y = self.swish(y)
        y = self.fc2(y)
        y = torch.sigmoid(y).view((b, c) + (1,) * (x.ndim - 2))
        return x * y


class MultiScaleAttentiveClassificationDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=3,
        conv_name="Conv3d",
        res_channels=None,
        hidden_dim=256,
        dropout_p=0.3,
        deep_supervision=False,
    ):
        """
        Args:
            in_channels: number of channels from the encoder's output
            num_classes: number of classification classes
            pool_scales: list of spatial sizes for 3D adaptive pool
            res_channels: number of channels inside each residual block
                        (defaults to in_channels if None)
            hidden_dim: dimension for the FC middle layer
            dropout_p: dropout probability
        """
        super().__init__()
        self.conv_name = conv_name
        self.res_channels = res_channels if res_channels is not None else in_channels
        self.dropout_p = dropout_p
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        if conv_name == "Conv3d":
            self.pool_scales = [(1, 1, 1), (2, 2, 2)]
            # Create a small sub-module for each pooling scale
            self.aux_blocks = nn.ModuleList()
            self.scale_blocks = nn.ModuleList()
            for scale in self.pool_scales:
                block = nn.Sequential(
                    nn.AdaptiveAvgPool3d(scale),
                    ResidualBlock(
                        in_channels,
                        self.res_channels,
                        dropout_ratio=self.dropout_p,
                        type="3D",
                    ),
                    SqueezeExcitation(self.res_channels, reduction=16, type="3D"),
                )
                self.scale_blocks.append(block)
                self.aux_blocks.append(
                    nn.Sequential(
                        nn.Flatten(),
                        nn.Dropout(p=self.dropout_p),
                        nn.Linear(
                            self.res_channels * scale[0] * scale[1] * scale[2],
                            hidden_dim,
                        ),
                        nn.GroupNorm(
                            get_num_groups(hidden_dim), hidden_dim, affine=True
                        ),
                        MemoryEfficientSwish(),
                        nn.Linear(hidden_dim, num_classes),
                    )
                )
            total_flat_dim = sum(
                [
                    self.res_channels * scale[0] * scale[1] * scale[2]
                    for scale in self.pool_scales
                ]
            )
        else:
            self.pool_scales = [(1, 1), (2, 2)]
            self.aux_blocks = nn.ModuleList()
            self.scale_blocks = nn.ModuleList()
            for scale in self.pool_scales:
                block = nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ResidualBlock(
                        in_channels,
                        self.res_channels,
                        dropout_ratio=self.dropout_p,
                        type="2D",
                    ),
                    SqueezeExcitation(self.res_channels, reduction=16, type="2D"),
                )
                self.scale_blocks.append(block)
                self.aux_blocks.append(
                    nn.Sequential(
                        nn.Flatten(),
                        nn.Dropout(p=self.dropout_p),
                        nn.Linear(self.res_channels * scale[0] * scale[1], hidden_dim),
                        nn.GroupNorm(
                            get_num_groups(hidden_dim), hidden_dim, affine=True
                        ),
                        MemoryEfficientSwish(),
                        nn.Linear(hidden_dim, num_classes),
                    )
                )

            total_flat_dim = sum(
                [self.res_channels * scale[0] * scale[1] for scale in self.pool_scales]
            )
        # Final MLP
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.fc1 = nn.Linear(total_flat_dim, hidden_dim)
        self.gn_fc1 = nn.GroupNorm(get_num_groups(hidden_dim), hidden_dim, affine=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        """
        x: shape (batch_size, in_channels, D, H, W) or (batch_size, in_channels, H, W)
        """
        # 1) Multi-scale pooling + residual + SE
        scale_feats = []
        aux_feats = []
        for i, block in enumerate(self.scale_blocks):
            feat = block(x)  # shape: (B, res_channels, scale_d, scale_h, scale_w)
            feat = feat.view(feat.size(0), -1)  # flatten each scale
            scale_feats.append(feat)
            aux_feats.append(self.aux_blocks[i](feat))
        # 2) Concat across scales
        combined = torch.cat(scale_feats, dim=1)  # shape: (B, total_flat_dim)

        # 3) MLP
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.gn_fc1(x)
        x = self.swish(x)
        x = self.fc2(x)  # shape: (B, num_classes)
        aux_feats.append(x)
        class_outputs = aux_feats[::-1]
        if self.deep_supervision:
            return class_outputs
        else:
            return class_outputs[0]
