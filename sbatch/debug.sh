cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

torchrun src/train_kd.py --name "teacher=coca_ViT-L-14-pretrained=laion2B-s13B-b90k_student=RN50_timm_alpha=0.0_T=1.0" \
                            --epochs 100 \
                            --learning_rate 0.1 \
                            --cuda \
                            --workers 5 \
                            --model "resnet50" \
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
                            --teacher-logits-path /scratch/mp5847/wise-ft-precompute/coca_ViT-L-14-laion2B-s13B-b90k/coca_ViT-L-14-laion2B-s13B-b90k-imagenet-logits.npy \
                            --loss-type "kl_softlabels"
