#!/usr/bin/bash
#SBATCH --job-name=hrnet32_final
#SBATCH --output=hrnet32_final.out
#SBATCH --error=hrnet32_final.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1

ml python/3.9.0
source /scratch/users/yonigoz/mmvenv/bin/activate
python3 tools/test.py configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_no_pretrained.py /scratch/users/yonigoz/mmpose_data/work_dirs/no_pretrained/HRNet/w48_dark_pretrained/best_infinity_AP_epoch_18.pth


