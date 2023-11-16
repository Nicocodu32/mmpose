#!/usr/bin/bash
#SBATCH --job-name=hrnet32_no_infinity
#SBATCH --output=hrnet32_no_infinity.out
#SBATCH --error=hrnet32_no_infinity.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:16GB
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

ml python/3.9.0
source /scratch/users/yonigoz/mmvenv/bin/activate
python3 tools/train.py configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_hrnet-w32_dark-8xb64-210e_no_infinity.py --amp


