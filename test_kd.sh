cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

python src/test_kd.py   \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet  \
    --model RN50 \
    --pretrained "" \
    --checkpoint "/scratch/mp5847/wise-ft-kd/teacher=coca_ViT-L-14-pretrained=laion2B-s13B-b90k_student=RN50_timm_alpha=0.0_T=1.0/model_33.pt"  \
    --data-location "/" \
    --save "./"