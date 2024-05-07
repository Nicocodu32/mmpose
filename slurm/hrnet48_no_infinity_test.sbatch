#!/usr/bin/bash
#SBATCH --job-name=hrnet48_no_infinity_test
#SBATCH --output=hrnet48_no_infinity_test.out
#SBATCH --error=hrnet48_no_infinity_test.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1

python3 tools/test.py configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_no_infinity.py /scratch/users/yonigoz/mmpose_data/work_dirs/no_infinity/HRNet/w48_dark_pretrained/epoch_30.pth


