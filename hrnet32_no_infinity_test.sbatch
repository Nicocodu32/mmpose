#!/usr/bin/bash
#SBATCH --job-name=hrnet32_no_infinity_test
#SBATCH --output=hrnet32_no_infinity_test.out
#SBATCH --error=hrnet32_no_infinity_test.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1

ml python/3.9.0
source /scratch/users/yonigoz/mmvenv/bin/activate
python3 tools/test.py configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_hrnet-w32_dark-8xb64-210e_no_infinity.py /scratch/users/yonigoz/mmpose_data/work_dirs/no_infinity/HRNet/w32_dark_bedlam/best_infinity_AP_epoch_18.pth


