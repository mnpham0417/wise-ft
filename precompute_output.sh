cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

torchrun src/precompute_output.py  --num_epoch 400 \
                                    --learning_rate 0.01 \
                                    --cuda \
                                    --workers 20 \
                                    --model RN50 \
                                    --pretrained "" \
                                    --teacher-ckpt "/scratch/mp5847/wise-ft-kd/rn50_scratch_1/model_99.pt" \
                                    --batch_size 512 \
                                    --dist-url "env://" \
                                    --dist-backend "nccl" \
                                    --multiprocessing-distributed \
                                    --world-size 1 \
                                    --template openai_imagenet_template \
                                    --train-dataset=ImageNet \
                                    --data-location=/ \
                                    --rank 0