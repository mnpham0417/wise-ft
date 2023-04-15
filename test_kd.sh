cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

python src/test_kd.py   \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet  \
    --model RN50 \
    --pretrained "" \
    --checkpoint "/scratch/mp5847/wise-ft-kd/rn50_scratch_0/model_99.pt"  \
    --data-location "/" \
    --alpha 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --save "./"