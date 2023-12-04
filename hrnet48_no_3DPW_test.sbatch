#!/usr/bin/bash
#SBATCH --job-name=hrnet48_no_3DPW_test
#SBATCH --output=hrnet48_no_3DPW_test.out
#SBATCH --error=hrnet48_no_3DPW_test.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1

ml python/3.9.0
source /scratch/users/yonigoz/mmvenv/bin/activate
python3 tools/test.py configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_no_3DPW.py hrnet48_no_3DPW_test/best_infinity_AP_epoch_27.pth


