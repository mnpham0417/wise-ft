cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

python src/test.py   \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet  \
    --checkpoint "/scratch/mp5847/wise-ft-kd/teacher=coca_ViT-L-14-pretrained=laion2B-s13B-b90k_student=RN50_timm_alpha=0.0_T=1.0/optimizer_31.pt"  \
    --data-location "/" \
    --batch-size 64