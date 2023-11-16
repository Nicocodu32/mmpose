
#!/usr/bin/bash
#SBATCH --job-name=vit_base_no_3DPW
#SBATCH --output=vit_base_no_3DPW.out
#SBATCH --error=vit_base_no_3DPW.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:16GB
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

ml python/3.9.0
source /scratch/users/yonigoz/mmvenv/bin/activate
python3 tools/train.py configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_ViTPose-base_8xb64-210e_no_3DPW.py --amp


