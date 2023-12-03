#!/usr/bin/bash
#SBATCH --job-name=hrnet32_final
#SBATCH --output=hrnet32_final.out
#SBATCH --error=hrnet32_final.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:16GB
#SBATCH --mem=16GB
#SBATCH --time=48:00:00

ml python/3.9.0
source /scratch/users/yonigoz/mmvenv/bin/activate
python3 tools/test.py configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet32/td-hm_hrnet-w32_dark-8xb64-210e_merge_bedlam_infinity_coco_3DPW_eval_rich-384x288_pretrained.py /scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_coco_3DPW_eval_bedlam_final/HRNet/w32_dark_bedlam/best_infinity_AP_epoch_21.pth --amp


