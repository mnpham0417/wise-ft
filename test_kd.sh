cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

python src/test_kd.py   \
    --eval-datasets=ImageNet,ImageNetV2  \
    --model RN50 \
    --pretrained "" \
    --checkpoint "/scratch/mp5847/wise-ft-kd/teacher=ViT-B-32-pretrained=openai_student=RN50_alpha=0.0_T=1.0_correct_low_entropy=1-1-hard/model_99.pt"  \
    --data-location "/" 