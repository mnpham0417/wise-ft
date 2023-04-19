#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=47:59:00
#SBATCH --mem=120GB
#SBATCH --job-name=kd_wise-ft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --output=kd_wise-ft_teacher=RN50-pretrained=ImageNet_student=RN50_alpha=0.0_T=1.0_%j.out

module purge

singularity exec --nv \
  --overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate /scratch/mp5847/conda_environments/conda_pkgs/wise_ft; cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \
      torchrun src/train_kd_ce_softlabels.py --name "teacher=RN50-pretrained=ImageNet_student=RN50_alpha=0.0_T=1.0" \
                            --epochs 100 \
                            --learning_rate 0.01 \
                            --cuda \
                            --workers 6 \
                            --model "RN50" \
                            --pretrained "" \
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
                            --teacher-logits-path /scratch/mp5847/wise-ft-precompute/rn50_scratch_1-ImageNet-imagenet-logits.npy \
                            --resume \
                            --pretrained-ckpt "/scratch/mp5847/wise-ft-kd/rn50_scratch_0/model_99.pt"'