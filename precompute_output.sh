cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

torchrun src/precompute_output.py  --num_epoch 400 \
                                    --learning_rate 0.01 \
                                    --cuda \
                                    --workers 20 \
                                    --model ViT-g-14 \
                                    --pretrained laion2b_s12b_b42k \
                                    --teacher-ckpt "/scratch/mp5847/wise-ft-ckpt/ViT-g-14-pretrained=laion2b_s12b_b42k/wise_ft_alpha=0.000.pt" \
                                    --batch_size 256 \
                                    --dist-url "env://" \
                                    --dist-backend "nccl" \
                                    --multiprocessing-distributed \
                                    --world-size 1 \
                                    --template openai_imagenet_template \
                                    --train-dataset=ImageNet \
                                    --data-location=/ \
                                    --rank 0