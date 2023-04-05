#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:a100:1
#SBATCH --time=167:59:00
#SBATCH --mem=64GB
#SBATCH --job-name=wise-ft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --output=wise-ft_vit-l-14-pretrained=laion2b_s32b_b82k_%j.out

module purge

singularity exec --nv \
  --overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate /scratch/mp5847/src/comp_exp; cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \
      python src/wise_ft.py   \
        --train-dataset=ImageNet  \
        --epochs=10  \
        --lr=0.00003  \
        --batch-size=32 \
        --cache-dir=cache  \
        --model="ViT-L-14"  \
        --pretrained=laion2b_s32b_b82k  \
        --eval-datasets=ImageNet  \
        --template=openai_imagenet_template  \
        --results-db=results.jsonl  \
        --save=/scratch/mp5847/wise-ft-ckpt/ViT-L-14-pretrained=laion2b_s32b_b82k  \
        --data-location=/ \
        --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'




