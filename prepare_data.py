#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-01-23 21:44:30
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-23 23:46:57
FilePath     : /nnUNet-Multi-Task/prepare_data.py
Description  :
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import os
import SimpleITK as sitk
import numpy as np
import shutil
import json
from typing import Tuple


def generate_dataset_json(
    output_folder: str,
    channel_names: dict,
    labels: dict,
    num_training_cases: int,
    file_ending: str,
    regions_class_order: Tuple[int, ...] = None,
    dataset_name: str = None,
    reference: str = None,
    release: str = None,
    license: str = None,
    description: str = None,
    overwrite_image_reader_writer: str = None,
    **kwargs,
):
    """
    Generates a dataset.json file in the output folder

    channel_names:
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }

        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

    num_training_cases: is used to double check all cases are there!

    file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
    segmentations!

    dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
    completeness and as a reminder that these would be great!

    overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
    BaseReaderWriter, place it into nnunet.imageio and reference it here by name

    kwargs: whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any(
        [isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()]
    )
    if has_regions:
        assert regions_class_order is not None, (
            f"You have defined regions but regions_class_order is not set. "
            f"You need that."
        )
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        "channel_names": channel_names,  # previously this was called 'modality'. I didn't like this so this is
        # channel_names now. Live with it.
        "labels": labels,
        "numTraining": num_training_cases,
        "file_ending": file_ending,
    }

    if dataset_name is not None:
        dataset_json["name"] = dataset_name
    if reference is not None:
        dataset_json["reference"] = reference
    if release is not None:
        dataset_json["release"] = release
    if license is not None:
        dataset_json["licence"] = license
    if description is not None:
        dataset_json["description"] = description
    if overwrite_image_reader_writer is not None:
        dataset_json["overwrite_image_reader_writer"] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json["regions_class_order"] = regions_class_order

    dataset_json.update(kwargs)

    with open(os.path.join(output_folder, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, sort_keys=False, indent=4)


def make_if_dont_exist(folder_path: str, overwrite: bool = False):
    if os.path.exists(folder_path):
        if not overwrite:
            print(f"{folder_path} exists, no overwrite here.")
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path, ignore_errors=True)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")


def convert_labels_to_integer(label_path, output_path, allowed_labels={0, 1, 2}):
    """
    Converts all label images in the specified directory to integer data types.

    :param label_dir: Path to the directory containing label images.
    :param allowed_labels: Set of allowed label integers.
    :param output_dir: Path to save converted label images. If None, overwrites originals.
    """
    print(f"Processing: {label_path}")

    # Read the label image
    label_image = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label_image)

    # Check data type
    if np.issubdtype(label_array.dtype, np.integer):
        print(" - Already an integer type. Skipping conversion.")
        # Optionally, verify label values
        unique_labels = np.unique(label_array)
        if not set(unique_labels).issubset(allowed_labels):
            print(f"   Warning: Unexpected labels found: {unique_labels}")

    # Round label values to nearest integer
    rounded_labels = np.round(label_array).astype(np.int32)

    # Verify that all labels are within the allowed set
    unique_labels_before = np.unique(label_array)
    unique_labels_after = np.unique(rounded_labels)
    print(f" - Unique labels before conversion: {unique_labels_before}")
    print(f" - Unique labels after rounding: {unique_labels_after}")

    unexpected_labels = set(unique_labels_after) - allowed_labels
    if unexpected_labels:
        print(
            f"   Warning: Found unexpected labels {unexpected_labels}. Handling them by setting to background (0)."
        )
        # Set unexpected labels to background (0)
        rounded_labels[np.isin(rounded_labels, list(unexpected_labels))] = 0
        unique_labels_after = np.unique(rounded_labels)
        print(
            f" - Unique labels after handling unexpected labels: {unique_labels_after}"
        )

    # Create a new SimpleITK image with integer type
    corrected_label_image = sitk.GetImageFromArray(rounded_labels)
    corrected_label_image.CopyInformation(label_image)

    # Save the corrected label image
    sitk.WriteImage(corrected_label_image, output_path)
    print(f" - Saved converted label image to {output_path}\n")


def extract_type_info(
    raw_dir,
    output_dir,
    dataset_id,
    dataset_name,
    modes,
    allowed_labels,
    allowed_label_names,
    num_subtypes,
):
    """
    Extracts the type information from the raw data directory.

    :param raw_dir: Path to the raw data directory.
    :param num_subtypes: Number of subtype classes.
    :return: Dictionary containing type information.
    """
    if os.path.exists(
        os.path.join(
            output_dir, f"Dataset{dataset_id}_{dataset_name}", "type_info.json"
        )
    ):
        with open(
            os.path.join(
                output_dir, f"Dataset{dataset_id}_{dataset_name}", "type_info.json"
            )
        ) as f:
            type_info = json.load(f)
    else:
        type_info = {}
    dataset_name = dataset_name.capitalize()
    output_path = os.path.join(output_dir, f"Dataset{dataset_id}_{dataset_name}")
    make_if_dont_exist(output_path, overwrite=False)
    assert os.path.exists(raw_dir), f"Directory not found: {raw_dir}"
    for mode in modes:
        assert mode in ["train", "validation", "test"], f"Invalid mode: {mode}"
        if mode in type_info.keys() and type_info[mode] != {}:
            print(f"Type info for {mode} already exists. Skipping...")
            continue
        output_mode_path = os.path.join(output_path, mode)
        output_mode_img_path = os.path.join(output_mode_path, "images")
        output_mode_seg_path = os.path.join(output_mode_path, "labels")
        make_if_dont_exist(output_mode_path, overwrite=True)
        make_if_dont_exist(output_mode_img_path, overwrite=True)
        make_if_dont_exist(output_mode_seg_path, overwrite=True)
        type_info[mode] = {}
        mode_dir = os.path.join(raw_dir, mode)
        assert os.path.exists(mode_dir), f"Directory not found: {mode_dir}"
        if mode != "test":
            for sub in range(num_subtypes):
                sub_dir = os.path.join(mode_dir, f"Subtype{sub}")
                assert os.path.exists(sub_dir), f"Directory not found: {sub_dir}"
                for filename in os.listdir(sub_dir):
                    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                        file = filename.split("_")
                        if len(file) > 3:
                            scan_file = os.path.join(
                                output_mode_img_path,
                                dataset_name + "_" + "_".join(file[2:]),
                            )
                            shutil.copy(os.path.join(sub_dir, filename), scan_file)
                            assert os.path.exists(
                                os.path.join(sub_dir, "_".join(file[:-1]) + ".nii.gz")
                            ), (
                                f"Label file not found: {os.path.join(sub_dir, '_'.join(file[:-1]) + '.nii.gz')}"
                            )

                            label_file = os.path.join(
                                output_mode_seg_path,
                                dataset_name + "_" + file[2] + ".nii.gz",
                            )
                            convert_labels_to_integer(
                                os.path.join(sub_dir, "_".join(file[:-1]) + ".nii.gz"),
                                label_file,
                                allowed_labels=allowed_labels,
                            )
                            print(f"Processed: {dataset_name + '_' + file[2]}")
                            type_info[mode][dataset_name + "_" + file[2]] = {
                                "image": scan_file,
                                "label": label_file,
                                "subtype": sub,
                            }
        else:
            for filename in os.listdir(mode_dir):
                if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                    file = filename.split("_")
                    scan_file = os.path.join(
                        output_mode_img_path,
                        dataset_name + "_" + "_".join(file[1:]),
                    )
                    shutil.copy(os.path.join(mode_dir, filename), scan_file)
                    type_info[mode][dataset_name + "_" + file[1]] = {"image": scan_file}

    with open(os.path.join(output_path, "type_info.json"), "w") as f:
        json.dump(type_info, f, indent=4)

    generate_dataset_json(
        output_folder=output_path,
        channel_names={0: "CT"},
        labels={label: i for i, label in enumerate(allowed_label_names)},
        num_training_cases=len(type_info["train"]),
        file_ending="." + ".".join(file[-1].split(".")[1:]),
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument(
        "-i",
        type=str,
        required=True,
        help="Path to the data directory containing labels and images.",
    )
    args.add_argument(
        "-o", type=str, required=True, help="Path to the output directory."
    )
    args.add_argument("-id", type=str, required=True, help="Dataset ID XXX.")
    args.add_argument("-n", type=str, required=True, help="Dataset Name.")
    args.add_argument(
        "-l",
        type=str,
        nargs="+",
        required=True,
        help="All Valid Label Names including background. e.g. background pancreas lesion",
    )
    args.add_argument("-ns", type=int, required=True, help="Number of subtype classes.")
    args.add_argument(
        "-m",
        type=str,
        nargs="+",
        help="Dataset part train, validation or test, could be multiple one time.",
    )
    args = args.parse_args()

    raw_dir = args.i
    output_dir = args.o
    dataset_id = args.id
    dataset_name = args.n
    allowed_labels = set([i for i in range(len(args.l))])
    allowed_label_names = [i for i in args.l]
    num_subtypes = args.ns
    modes = args.m

    type_info = extract_type_info(
        raw_dir,
        output_dir,
        dataset_id,
        dataset_name,
        modes,
        allowed_labels,
        allowed_label_names,
        num_subtypes,
    )
