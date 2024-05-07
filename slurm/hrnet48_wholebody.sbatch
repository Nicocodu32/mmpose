#!/usr/bin/bash
#SBATCH --job-name=hrnet48_wholebody
#SBATCH --output=hrnet48_wholebody.out
#SBATCH --error=hrnet48_wholebody.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:16GB
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

python3 tools/train.py configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_wholebody.py --amp --resume


