cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

python src/test.py   \
    --eval-datasets=ImageNet,ImageNetV2  \
    --checkpoint "/scratch/mp5847/wise-ft-ckpt/ViT-B-32-pretrained=openai/finetuned/wise_ft_alpha=0.000.pt"  \
    --data-location "/" 