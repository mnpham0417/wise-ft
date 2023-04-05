cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; 

torchrun src/precompute_output.py --batch_size 512 \
                                  --model "RN101" \
                                  --pretrained "openai" \
                                  --template "openai_imagenet_template" \
                                  --train-dataset "ImageNet" \
                                  --data-location "/" \
                                  --teacher-ckpt "/scratch/mp5847/wise-ft-ckpt/RN101-pretrained=openai/finetuned/wise_ft_alpha=0.000.pt" \
                                  --workers 4