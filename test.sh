cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

python src/test.py   \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet  \
    --checkpoint "/scratch/mp5847/wise-ft-ckpt/coca_ViT-B-32-pretrained=laion2B-s13B-b90k/finetuned/wise_ft_alpha=0.000.pt"  \
    --data-location "/" \
    --batch-size 64