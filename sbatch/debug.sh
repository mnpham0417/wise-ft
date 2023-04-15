cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

torchrun src/train_kd_ce_softlabels.py --name "teacher=RN50-pretrained=ImageNet_student=RN50_alpha=0.0_T=1.0" \
                            --epochs 100 \
                            --learning_rate 0.1 \
                            --cuda \
                            --workers 5 \
                            --model "RN50" \
                            --pretrained "" \
                            --batch_size 32 \
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
                            --pretrained-ckpt "/scratch/mp5847/wise-ft-kd/rn50_scratch_0/model_99.pt"
