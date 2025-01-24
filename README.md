# nnUNet-Multi-Task

## Environment and Requirements

- OS: Ubuntu 22.04
- CPU: AMD Ryzen 9 7900X
- GPU: NVIDIA RTX 3090
- RAM: 48GB
- CUDA: 12.2
- Python: 3.9.21

To install requirements:

```bash
conda create -n nnunet python=3.9 -y
conda activate nnunet
git clone https://github.com/mikami520/nnUNet-Multi-Task.git
cd nnUNet-Multi-Task
pip install -e .
```

add the following to the shell profile:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## Dataset

Raw dataset in this task is:

```text
ML-Quiz-3DMedImg/
├── train
|   ├── subtype0
│   |   ├── quiz_0_001_0000.nii.gz
│   |   ├── quiz_0_001.nii.gz
│   |   ├── quiz_0_002_0000.nii.gz
│   |   ├── quiz_0_002.nii.gz
│   |   ├── quiz_0_003_0000.nii.gz
│   |   ├── quiz_0_003.nii.gz
│   |   ├── quiz_0_004_0000.nii.gz
│   |   ├── quiz_0_004.nii.gz
│   ├── ...
├── validation
|   ├── subtype0
│   |   ├── quiz_0_010_0000.nii.gz
│   |   ├── quiz_0_010.nii.gz
│   |   ├── ...
├── test
|   ├── quiz_252_0000.nii.gz
|   ├── quiz_253_0000.nii.gz
|   ├── ...
```

run the following command to generate the nnUNet-compatible dataset:

```bash
python prepare_data.py -i /path/to/raw/data -o /path/to/output_dir -id XXX -n DatasetName -l all_available_label_names -ns number_subtypes -m train validation test
```

The prepared dataset will be:

```text
Output_dir/
├── Dataset001_DatasetName
|   ├── dataset.json
|   ├── type_info.json
|   ├── train
│   |   ├── images
│   |   |   ├── DatasetName_001_0000.nii.gz
│   |   |   ├── DatasetName_002_0000.nii.gz
│   |   |   ├── DatasetName_003_0000.nii.gz
│   |   |   ├── DatasetName_004_0000.nii.gz
│   |   |   ├── DatasetName_005_0000.nii.gz
│   |   |   ├── ...
│   |   ├── labels
│   |   |   ├── DatasetName_001.nii.gz
│   |   |   ├── DatasetName_002.nii.gz
│   |   |   ├── DatasetName_003.nii.gz
│   |   |   ├── DatasetName_004.nii.gz
│   |   |   ├── DatasetName_005.nii.gz
│   |   |   ├── ...
|   ├── validation
│   |   ├── images
│   |   |   ├── ...
│   |   ├── labels
│   |   |   ├── ...
|   ├── test
│   |   ├── images
├── Dataset002_DatasetName
```

`type_info.json` contains the path information of the dataset for each paired images, labels and their corresponding subtype class. This will be used in the training to provide the ground truth labels for classification task and in the inference to compute metrics.

Finally you can copy train, validation and test folders to `nnUNet_raw` folder.

```bash
# prepare nnUNet_raw, nnUNet_preprocessed and nnUNet_results folder

mkdir -p /path/to/nnUNet_raw/DatasetXXX_DatasetName/imagesTr
mkdir -p /path/to/nnUNet_raw/DatasetXXX_DatasetName/labelsTr
mkdir -p /path/to/nnUNet_raw/DatasetXXX_DatasetName/imagesTs

mkdir -p /path/to/nnUNet_preprocessed 
mkdir -p /path/to/nnUNet_results

# copy files to nnUnet_raw
cp -r /path/to/output_dir/DatasetXXX_DatasetName/train/images/* /path/to/nnUNet_raw/DatasetXXX_DatasetName/imagesTr

cp -r /path/to/output_dir/DatasetXXX_DatasetName/train/labels/* /path/to/nnUNet_raw/DatasetXXX_DatasetName/labelsTr

cp -r /path/to/output_dir/DatasetXXX_DatasetName/validation/images/* /path/to/nnUNet_raw/DatasetXXX_DatasetName/imagesTs

cp -r /path/to/output_dir/DatasetXXX_DatasetName/test/images/* /path/to/nnUNet_raw/DatasetXXX_DatasetName/imagesTs

# copy json files to nnUnet_raw
cp /path/to/output_dir/DatasetXXX_DatasetName/dataset.json /path/to/nnUNet_raw/DatasetXXX_DatasetName/
cp /path/to/output_dir/DatasetXXX_DatasetName/type_info.json /path/to/nnUNet_raw/DatasetXXX_DatasetName/
```

## nnUNet Preprocessing

run nnUNet preprocessing:

```bash
nnUNetv2_plan_and_preprocess -d XXX -pl nnUNetPlannerResEncM --verify_dataset_integrity
```

After the preprocessing, remember to copy `type_info.json` to the `nnUNet_preprocessed/DatasetXXX_DatasetName/` folder:

```bash
cp /path/to/nnUNet_raw/DatasetXXX_DatasetName/type_info.json /path/to/nnUNet_preprocessed/DatasetXXX_DatasetName/
```

For more details, please refer to the [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)

## Training

run nnUNet 5-fold cross validation 3d_fullres training with the script file `run.sh`, remember to modify the `XXX` to the dataset ID:

```bash
bash run.sh
```

For more details, please refer to the [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)

## Find the Best Configuration

run the following command to find the best configuration:

```bash
nnUNetv2_find_best_configuration XXX -p nnUNetResEncUNetMPlans -c 3d_fullres -tr nnUNetTrainerSegAndClass 
```

See nnUNetv2_find_best_configuration -h for more options.

nnUNetv2_find_best_configuration will also automatically determine the postprocessing that should be used. Postprocessing in nnU-Net only considers the removal of all but the largest component in the prediction (once for foreground vs background and once for each label/region).

Once completed, the command will print to your console exactly what commands you need to run to make predictions. It will also create two files in the `nnUNet_results/DatasetName` folder for you to inspect:

`inference_instructions.txt` again contains the exact commands you need to use for predictions
`inference_information.json` can be inspected to see the performance of all configurations and ensembles, as well as the effect of the postprocessing plus some debug information.

## Inference

For each of the desired configurations, run the following command for segmentation task:

```bash
nnUNetv2_predict -d DatasetXXX_DatasetName -i INPUT_FOLDER -o OUTPUT_FOLDER_Seg -f 0 1 2 3 4 -tr nnUNetTrainerSegAndClass -c 3d_fullres -p nnUNetResEncUNetMPlans
```

For classification task, run the following command:

```bash
python inference_class.py -id XXX -c 3d_fullres -i INPUT_FOLDER -o OUTPUT_FOLDER_Class -f 0 1 2 3 4 -tr nnUNetTrainerSegAndClass -p nnUNetResEncUNetMPlans
```

The `subtype_results.csv` will be saved in the `OUTPUT_FOLDER_Class` folder.

## Postprocessing only for segmentation task

run the following command to apply postprocessing:

```bash
nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER_Seg -o OUTPUT_FOLDER_Seg_PP -pp_pkl_file /path/to/nnUNet_results/DatasetXXX_DatasetName/nnUNetTrainerSegAndClass__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /path/to/nnUNet_results/DatasetXXX_DatasetName/nnUNetTrainerSegAndClass__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json
```

## Evaluation if having ground truth labels

run the following command to evaluate the segmentation and classification tasks:

```bash
python compute_metric.py -id XXX -oseg OUTPUT_FOLDER_Seg_PP -ocls OUTPUT_FOLDER_Class
```

The `DSC.csv` will be saved in the `OUTPUT_FOLDER_Seg_PP` folder and the `macro-average f1 score` will be printed on the terminal.