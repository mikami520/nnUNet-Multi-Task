###
 # @Author       : Chris Xiao yl.xiao@mail.utoronto.ca
 # @Date         : 2025-01-20 01:28:03
 # @LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
 # @LastEditTime : 2025-01-24 01:53:07
 # @FilePath     : /nnUNet-Multi-Task/run.sh
 # @Description  : 
 # I Love IU
 # Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
### 

nnUNetv2_train XXX 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSegAndClass --npz -device cuda
nnUNetv2_train XXX 3d_fullres 1 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSegAndClass --npz -device cuda
nnUNetv2_train XXX 3d_fullres 2 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSegAndClass --npz -device cuda
nnUNetv2_train XXX 3d_fullres 3 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSegAndClass --npz -device cuda
nnUNetv2_train XXX 3d_fullres 4 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSegAndClass --npz -device cuda