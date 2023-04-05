cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

torchrun src/train_kd.py --name "teacher=RN101-yfcc15m-wiseft-alpha=0.200_student=RN101-yfcc15m_alpha=0.5" \
                            --epochs 120 \
                            --learning_rate 0.01 \
                            --cuda \
                            --workers 20 \
                            --model RN101 \
                            --pretrained "" \
                            --teacher-ckpt "/scratch/mp5847/wise-ft-ckpt/RN101-pretrained=yfcc15m/finetuned/wise_ft_alpha=0.200.pt" \
                            --alpha 0.5 \
                            --batch_size 128 \
                            --dist-url "env://" \
                            --dist-backend "nccl" \
                            --multiprocessing-distributed \
                            --world-size 1 \
                            --template openai_imagenet_template \
                            --train-dataset=ImageNet \
                            --data-location=/ \
                            --rank 0