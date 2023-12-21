#!/usr/bin/bash
#SBATCH --job-name=hrnet48_no_bedlam_test
#SBATCH --output=hrnet48_no_bedlam_test.out
#SBATCH --error=hrnet48_no_bedlam_test.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:16GB
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

ml python/3.9.0
source /scratch/users/yonigoz/mmvenv/bin/activate
python3 tools/train.py configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_no_bedlam.py --amp --resume


