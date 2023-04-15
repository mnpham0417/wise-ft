#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=47:59:00
#SBATCH --mem=120GB
#SBATCH --job-name=kd_wise-ft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --output=kd_wise-ft_teacher=RN101-pretrained=openai_student=RN50_alpha=0.0_T=1.0-entropy-upper_threshold=4-hard-finetune-entropy-upper_threshold=4_%j.out

module purge

# ip=$(ifconfig eno1np0 | grep "inet " | awk '{print $2}')
# echo "IP address: $ip"

singularity exec --nv \
  --overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate /scratch/mp5847/conda_environments/conda_pkgs/wise_ft; cd /home/mp5847/src/wise-ft-selected-entropy; export PYTHONPATH="$PYTHONPATH:$PWD"; \
      torchrun src/train_kd.py --name "teacher=RN101-pretrained=openai_student=RN50_alpha=0.0_T=1.0-entropy-upper_threshold=4-hard-finetune-entropy-upper_threshold=4" \
                            --epochs 100 \
                            --learning_rate 0.01 \
                            --cuda \
                            --workers 5 \
                            --model "RN50" \
                            --pretrained "" \
                            --teacher-ckpt "/scratch/mp5847/wise-ft-ckpt/RN101-pretrained=openai/finetuned/wise_ft_alpha=0.000.pt" \
                            --alpha 0.0 \
                            --batch_size 1024 \
                            --dist-url "env://" \
                            --dist-backend "nccl" \
                            --multiprocessing-distributed \
                            --world-size 1 \
                            --template openai_imagenet_template \
                            --train-dataset=ImageNet \
                            --data-location=/ \
                            --rank 0 \
                            --T 1.0 \
                            --teacher-logits-path /scratch/mp5847/wise-ft-precompute/RN101-openai/RN101-openai-imagenet-logits.npy \
                            --teacher-index-path /scratch/mp5847/wise-ft-precompute/RN101-openai/RN101-openai-imagenet-index-entropy-upper_threshold-4.npy \
                            --resume \
                            --model_ckpt "/scratch/mp5847/wise-ft-kd/teacher=RN101-pretrained=openai_student=RN50_alpha=0.0_T=1.0-entropy-upper_threshold=4-hard/model_99.pt" '