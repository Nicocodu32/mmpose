#!/usr/bin/bash
#SBATCH --job-name=vit_huge_no_coco_test
#SBATCH --output=vit_huge_no_coco_test.out
#SBATCH --error=vit_huge_no_coco_test.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:16GB
#SBATCH --mem=32GB

python3 tools/test.py configs/body_2d_keypoint/topdown_heatmap/infinity/vithuge/td-hm_ViTPose-huge_8xb64-210e_no_coco.py /scratch/users/yonigoz/mmpose_data/work_dirs/no_coco/ViT/huge_pretrained/epoch_30.pth
