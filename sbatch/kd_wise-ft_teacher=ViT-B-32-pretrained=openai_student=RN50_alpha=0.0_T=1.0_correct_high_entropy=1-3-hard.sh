#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:a100:1
#SBATCH --time=47:59:00
#SBATCH --mem=200GB
#SBATCH --job-name=kd_wise-ft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --output=kd_wise-ft_teacher=ViT-B-32-pretrained=openai_student=RN50_alpha=0.0_T=1.0_correct_high_entropy=1-3-hard_%j.out

module purge

singularity exec --nv \
  --overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate /scratch/mp5847/conda_environments/conda_pkgs/wise_ft; cd /home/mp5847/src/wise-ft-selected-entropy; export PYTHONPATH="$PYTHONPATH:$PWD"; \
      torchrun src/train_kd_hard.py --name "teacher=ViT-B-32-pretrained=openai_student=RN50_alpha=0.0_T=1.0_correct_high_entropy=1-3-hard" \
                            --epochs 100 \
                            --learning_rate 0.1 \
                            --cuda \
                            --workers 5 \
                            --model "RN50" \
                            --pretrained "" \
                            --teacher-ckpt "/scratch/mp5847/wise-ft-ckpt/ViT-B-32-pretrained=openai/finetuned/wise_ft_alpha=0.000.pt" \
                            --alpha 0.0 \
                            --batch_size 512 \
                            --dist-url "env://" \
                            --dist-backend "nccl" \
                            --multiprocessing-distributed \
                            --world-size 1 \
                            --template openai_imagenet_template \
                            --train-dataset=ImageNet \
                            --data-location=/ \
                            --rank 0 \
                            --T 1.0 \
                            --teacher-logits-path /scratch/mp5847/wise-ft-precompute/ViT-B-32-openai-imagenet-logits.npy \
                            --teacher-index-path /scratch/mp5847/wise-ft-precompute/ViT-B-32-openai-imagenet-logits-correct-high-entropy-threshold-1-3.npy'
