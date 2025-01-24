#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-01-20 16:16:33
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-23 19:49:26
FilePath     : /Documents/nnUNet/inference_class.py
Description  :
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
from torch._dynamo import OptimizedModule
import argparse
from acvl_utils.cropping_and_padding.padding import pad_nd_image
import numpy as np
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerSegAndClass import (
    MultiScaleAttentiveClassificationDecoder,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.inference.data_iterators import (
    preprocessing_iterator_fromfiles,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from typing import List, Union
import os
from torch.backends import cudnn
import pandas as pd


class ClassPredictor(object):
    def __init__(
        self, dataset_name_or_id, configuration_name_or_id, plan, trainer, folds, device
    ):
        super(ClassPredictor, self).__init__()
        # initialize nnunet trainer

        preprocessed_dataset_folder_base = join(
            nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id)
        )
        plans_file = join(preprocessed_dataset_folder_base, plan + ".json")
        plans = load_json(plans_file)
        self.dataset_json = load_json(
            join(preprocessed_dataset_folder_base, "dataset.json")
        )

        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(
            configuration_name_or_id
        )
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)

        self.output_folder_base = (
            join(
                nnUNet_results,
                self.plans_manager.dataset_name,
                trainer
                + "__"
                + self.plans_manager.plans_name
                + "__"
                + configuration_name_or_id,
            )
            if nnUNet_results is not None
            else None
        )

        self.num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )
        self.device = device
        self.folds = folds

    def init_network(self):
        self.seg_network = get_network_from_plans(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.label_manager.num_segmentation_heads,
            allow_init=True,
            deep_supervision=False,
        ).to(self.device)

        in_channel = self.seg_network.encoder.output_channels[-1]
        self.class_decoder = MultiScaleAttentiveClassificationDecoder(
            in_channels=in_channel,
            num_classes=3,
            conv_name=self.configuration_manager.network_arch_conv_name,
            res_channels=in_channel,
            deep_supervision=False,
        ).to(self.device)
        self.seg_network = torch.compile(self.seg_network)
        self.class_decoder = torch.compile(self.class_decoder)

    def _internal_get_data_iterator_from_lists_of_filenames(
        self,
        input_list_of_lists: List[List[str]],
        seg_from_prev_stage_files: Union[List[str], None],
        output_filenames_truncated: Union[List[str], None],
        num_processes: int,
    ):
        return preprocessing_iterator_fromfiles(
            input_list_of_lists,
            seg_from_prev_stage_files,
            output_filenames_truncated,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == "cuda",
        )

    def _manage_input_and_output_lists(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
        folder_with_segs_from_prev_stage: str = None,
        overwrite: bool = True,
        part_id: int = 0,
        num_parts: int = 1,
        save_probabilities: bool = False,
    ):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(
                list_of_lists_or_source_folder, self.dataset_json["file_ending"]
            )
        print(
            f"There are {len(list_of_lists_or_source_folder)} cases in the source folder"
        )
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[
            part_id::num_parts
        ]
        caseids = [
            os.path.basename(i[0])[: -(len(self.dataset_json["file_ending"]) + 5)]
            for i in list_of_lists_or_source_folder
        ]
        print(
            f"I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)"
        )
        print(f"There are {len(caseids)} cases that I would like to predict")

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [
                join(output_folder_or_list_of_truncated_output_files, i)
                for i in caseids
            ]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files

        seg_from_prev_stage_files = [
            join(folder_with_segs_from_prev_stage, i + self.dataset_json["file_ending"])
            if folder_with_segs_from_prev_stage is not None
            else None
            for i in caseids
        ]
        # remove already predicted files form the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [
                isfile(i + self.dataset_json["file_ending"])
                for i in output_filename_truncated
            ]
            if save_probabilities:
                tmp2 = [isfile(i + ".npz") for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [
                output_filename_truncated[i] for i in not_existing_indices
            ]
            list_of_lists_or_source_folder = [
                list_of_lists_or_source_folder[i] for i in not_existing_indices
            ]
            seg_from_prev_stage_files = [
                seg_from_prev_stage_files[i] for i in not_existing_indices
            ]
            print(
                f"overwrite was set to {overwrite}, so I am only working on cases that haven't been predicted yet. "
                f"That's {len(not_existing_indices)} cases."
            )
        return (
            list_of_lists_or_source_folder,
            output_filename_truncated,
            seg_from_prev_stage_files,
            len(caseids),
        )

    def center_crop_3d(self, volume: torch.Tensor, output_shape: tuple) -> torch.Tensor:
        """
        Crops the 3D/4D `volume` tensor to `output_shape` in the D, H, W dimensions from the center.

        Args:
            volume (torch.Tensor):
                - shape (D, H, W) or (C, D, H, W)
            output_shape (tuple):
                - desired (pD, pH, pW) for the spatial dimensions.
                Does NOT include channel dimension.
        Returns:
            torch.Tensor:
                - cropped volume with shape (pD, pH, pW) if input was 3D
                - or (C, pD, pH, pW) if input was 4D
        Raises:
            ValueError: If `output_shape` is bigger than the current volume size.
        """
        dim = volume.dim()
        if dim not in (3, 4):
            raise ValueError(
                f"volume must be 3D or 4D, but got shape {list(volume.shape)}"
            )

        # Spatial dimensions are always the last 3
        D = volume.size(-3)
        H = volume.size(-2)
        W = volume.size(-1)

        pD, pH, pW = output_shape

        # Check if output_shape fits
        if pD > D or pH > H or pW > W:
            raise ValueError(
                f"Requested crop size {output_shape} is bigger than volume spatial shape {(D, H, W)}. "
                "Please use smaller crop or pad the volume."
            )

        # Compute center start indices
        start_D = (D - pD) // 2
        start_H = (H - pH) // 2
        start_W = (W - pW) // 2

        end_D = start_D + pD
        end_H = start_H + pH
        end_W = start_W + pW

        if dim == 4:
            # shape: (C, D, H, W)
            cropped = volume[:, start_D:end_D, start_H:end_H, start_W:end_W]
        else:
            # shape: (D, H, W)
            cropped = volume[start_D:end_D, start_H:end_H, start_W:end_W]

        return cropped

    def predict_class(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
        save_probabilities: bool = False,
        overwrite: bool = True,
        num_processes_preprocessing: int = default_num_processes,
        num_processes_segmentation_export: int = default_num_processes,
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0,
    ):
        # sort out input and output filenames
        (
            list_of_lists_or_source_folder,
            output_filename_truncated,
            seg_from_prev_stage_files,
            num_cases,
        ) = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage,
            overwrite,
            part_id,
            num_parts,
            save_probabilities,
        )
        output_dir = os.path.dirname(output_filename_truncated[0])
        if len(list_of_lists_or_source_folder) == 0:
            return

        class_logits_all = np.zeros((len(self.folds), num_cases, 3))
        filenames = []
        for fold in self.folds:
            data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
                list_of_lists_or_source_folder,
                seg_from_prev_stage_files,
                output_filename_truncated,
                num_processes_preprocessing,
            )
            self.init_network()
            print(f"\nLoading fold {fold}...")
            checkpoint = torch.load(
                join(
                    self.output_folder_base,
                    f"fold_{fold}",
                    "checkpoint_best.pth",
                ),
                map_location=self.device,
            )
            if isinstance(self.seg_network, OptimizedModule):
                self.seg_network._orig_mod.load_state_dict(
                    checkpoint["network_weights"]
                )
            else:
                self.seg_network.load_state_dict(checkpoint["network_weights"])

            if isinstance(self.class_decoder, OptimizedModule):
                self.class_decoder._orig_mod.load_state_dict(
                    checkpoint["class_decoder_weights"]
                )
            else:
                self.class_decoder.load_state_dict(checkpoint["class_decoder_weights"])

            with torch.no_grad():
                self.seg_network.eval()
                self.class_decoder.eval()
                for case, preprocessed in enumerate(data_iterator):
                    data = preprocessed["data"]
                    if isinstance(data, str):
                        delfile = data
                        data = torch.from_numpy(np.load(data))
                        os.remove(delfile)

                    ofile = preprocessed["ofile"]
                    if ofile is not None:
                        print(f"\nPredicting {os.path.basename(ofile)}:")
                    else:
                        print(f"\nPredicting image of shape {data.shape}:")
                    data, _ = pad_nd_image(
                        data,
                        self.configuration_manager.patch_size,
                        "constant",
                        {"value": 0},
                        True,
                        None,
                    )
                    crop_data = self.center_crop_3d(data, (64, 128, 192))
                    data = crop_data.unsqueeze(0).to(self.device)
                    encode_feat = self.seg_network.encoder(data)
                    class_logits = self.class_decoder(encode_feat[-1])
                    numpy_logits = class_logits.detach().cpu().numpy()
                    print(f"Subtype: {torch.argmax(class_logits, dim=1).item()}")
                    if int(fold) == 0:
                        filenames.append(f"{os.path.basename(ofile)}.nii.gz")
                    class_logits_all[int(fold), case] = numpy_logits[0]

        agg_logits = np.argmax(np.mean(class_logits_all, axis=0), axis=1)
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(
            {
                "Names": filenames,
                "Subtype": agg_logits,
            }
        )

        df.to_csv(join(output_dir, "subtype_results.csv"), index=False)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", type=str, help="Dataset name or ID to train with")
    parser.add_argument("-c", type=str, help="Configuration that should be trained")
    parser.add_argument("-i", type=str, help="Input folder with the data")
    parser.add_argument(
        "-o", type=str, help="Output folder where the predictions should be saved"
    )
    parser.add_argument(
        "-f",
        nargs="+",
        type=str,
        required=False,
        default=(0, 1, 2, 3, 4),
        help="Specify the folds of the trained model that should be used for prediction. "
        "Default: (0, 1, 2, 3, 4)",
    )
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="nnUNetTrainer",
        help="[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnUNetPlans",
        help="[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the training should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!",
    )

    args = parser.parse_args()

    assert args.device in ["cpu", "cuda", "mps"], (
        f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."
    )

    if args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    print(f"Using device: {device}")
    predictor = ClassPredictor(args.id, args.c, args.p, args.tr, args.f, device)
    predictor.predict_class(
        args.i,
        args.o,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3,
        num_parts=1,
        part_id=0,
    )
