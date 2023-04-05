cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

python src/test_kd_yfcc15m.py   \
    --eval-datasets=ImageNet,ImageNetV2  \
    --model RN101 \
    --pretrained "yfcc15m" \
    --checkpoint "/scratch/mp5847/wise-ft-kd/teacher=RN101-yfcc15m-wiseft-alpha=0.000_student=RN101-yfcc15m_feature_matching_mse/model_10.pt"  \
    --data-location "/" \
    --template openai_imagenet_template \
    --train-dataset=ImageNet