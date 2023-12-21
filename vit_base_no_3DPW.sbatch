#!/usr/bin/bash
#SBATCH --job-name=vit_base_no_3DPW
#SBATCH --output=vit_base_no_3DPW.out
#SBATCH --error=vit_base_no_3DPW.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:16GB
#SBATCH --mem=32GB
#SBATCH --time=48:00:00

python3 tools/train.py configs/body_2d_keypoint/topdown_heatmap/infinity/vitbase/td-hm_ViTPose-base_8xb64-210e_no_3DPW.py --amp --resume


