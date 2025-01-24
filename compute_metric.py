#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-01-23 01:23:57
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-24 01:45:50
FilePath     : /nnUNet-Multi-Task/compute_metric.py
Description  :
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from batchgenerators.utilities.file_and_folder_operations import join, load_json
import pandas as pd
from sklearn.metrics import confusion_matrix
import nibabel as nib
import numpy as np
import os


def load_nifti(file_path):
    """
    Loads a NIfTI file and returns the data as a NumPy array.

    Parameters:
        file_path (str): Path to the NIfTI file.

    Returns:
        np.ndarray: 3D array of the image data.
    """
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    return data


def compute_macro_F1(dataset_id, class_output_dir):
    type_info = load_json(
        join(
            nnUNet_preprocessed,
            maybe_convert_to_dataset_name(dataset_id),
            "type_info.json",
        )
    )

    y_true = []
    y_pred = []
    filenames = []
    df = pd.read_csv(join(class_output_dir, "subtype_results.csv"))
    for mode in ["validation", "test"]:
        valid_info = type_info[mode]
        for file in valid_info.keys():
            filename = file + ".nii.gz"
            if "subtype" not in valid_info[file].keys():
                continue
            if filename not in df["Names"].values:
                continue
            y_pred.append(df.loc[df["Names"] == filename, "Subtype"].values[0])
            filenames.append(filename)
            y_true.append(valid_info[file]["subtype"])

    # Get unique classes
    classes = sorted(list(set(y_true + y_pred)))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print("Confusion Matrix:")
    print(cm_df)

    # Calculate F1 for each class
    f1_scores = {}
    for cls in classes:
        TP = cm_df.at[cls, cls]
        FP = cm_df[cls].sum() - TP
        FN = cm_df.loc[cls].sum() - TP
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = (
            2 * Precision * Recall / (Precision + Recall)
            if (Precision + Recall) > 0
            else 0
        )
        f1_scores[cls] = F1
        print(
            f"Class {cls}: Precision={Precision:.3f}, Recall={Recall:.3f}, F1={F1:.3f}"
        )

    # Compute Macro F1
    macro_f1 = sum(f1_scores.values()) / len(classes)
    print(f"Macro-Averaged F1 Score: {macro_f1:.3f}")


def dice_score_single_class(pred, target, class_label):
    pred_class = pred == class_label
    target_class = target == class_label

    intersection = np.logical_and(pred_class, target_class).sum()
    pred_sum = pred_class.sum()
    target_sum = target_class.sum()

    if pred_sum + target_sum == 0:
        return 1.0

    dice = (2.0 * intersection) / (pred_sum + target_sum)
    return dice


def dice_score_multiclass(pred, target, class_labels, average="macro"):
    dice_dict = {}
    dice_scores = []
    weights = []

    for cls in class_labels:
        dice = dice_score_single_class(pred, target, cls)
        dice_dict[cls] = dice
        dice_scores.append(dice)
        weights.append((target == cls).sum())

    if average == "macro":
        average_dice = np.mean(dice_scores)
    elif average == "weighted":
        weights = np.array(weights)
        if weights.sum() == 0:
            average_dice = np.mean(dice_scores)
        else:
            average_dice = np.sum(np.array(dice_scores) * weights) / np.sum(weights)
    else:
        raise ValueError("average must be 'macro' or 'weighted'")

    return dice_dict, average_dice


def compute_DSC(dataset_id, seg_output_dir):
    type_info = load_json(
        join(
            nnUNet_preprocessed,
            maybe_convert_to_dataset_name(dataset_id),
            "type_info.json",
        )
    )
    filenames = []
    dices = []
    for mode in ["validation", "test"]:
        valid_info = type_info[mode]
        for file in sorted(valid_info.keys()):
            if "label" not in valid_info[file].keys():
                continue
            filename = join(seg_output_dir, file + ".nii.gz")
            if not os.path.exists(filename):
                continue
            seg = load_nifti(filename)
            gt = load_nifti(valid_info[file]["label"])
            class_labels = list(set(gt.flatten()))
            class_labels.remove(0)
            dice_dict, average_dice = dice_score_multiclass(seg, gt, class_labels)
            filenames.append(os.path.basename(filename))
            dices.append([dice_dict[1], dice_dict[2], average_dice])
    dices = np.vstack(dices)
    df = pd.DataFrame(
        {
            "Names": filenames,
            "DSC_Pancreas": dices[:, 0],
            "DSC_Lesion": dices[:, 1],
            "Average_DSC": dices[:, 2],
        }
    )
    df.to_csv(join(seg_output_dir, "DSC.csv"), index=False)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("-id", type=str, required=True)
    args.add_argument("-oseg", type=str, required=True)
    args.add_argument("-ocls", type=str, required=True)

    args = args.parse_args()
    compute_DSC(args.id, args.oseg)
    compute_macro_F1(args.id, args.ocls)
