cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

python3 src/precompute_output.py --model "coca_ViT-L-14" \
                                    --pretrained "laion2B-s13B-b90k" \
                                    --teacher-ckpt "/scratch/mp5847/wise-ft-ckpt/coca_ViT-L-14-pretrained=laion2B-s13B-b90k/finetuned/wise_ft_alpha=0.000.pt" \
                                    --batch_size 512 \
                                    --cuda \
                                    --workers 20 \
                                    --data-location=/ 