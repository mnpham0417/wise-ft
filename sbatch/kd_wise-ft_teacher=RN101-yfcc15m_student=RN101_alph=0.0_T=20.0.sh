#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=47:59:00
#SBATCH --mem=200GB
#SBATCH --job-name=kd_wise-ft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --output=kd_wise-ft_RN101-pretrained=yfcc15m_student=RN101_%j.out

module purge

singularity exec --nv \
  --overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate /scratch/mp5847/src/comp_exp; cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \
      torchrun src/train_kd.py --name "teacher=RN101-yfcc15m-wiseft-alpha=0.000_student=RN101-yfcc15m_alpha=0.0_T=20.0" \
                            --epochs 100 \
                            --learning_rate 0.1 \
                            --cuda \
                            --workers 5 \
                            --model RN101 \
                            --pretrained "" \
                            --teacher-ckpt "/scratch/mp5847/wise-ft-ckpt/RN101-pretrained=yfcc15m/finetuned/wise_ft_alpha=0.000.pt" \
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
                            --T 20.0 \
                            --teacher-logits-path /scratch/mp5847/wise-ft-precompute/RN101-yfcc15m-imagenet-logits.npy'
