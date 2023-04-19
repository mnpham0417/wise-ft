cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

python src/test_kd.py   \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet  \
    --model RN50 \
    --pretrained "" \
    --data-location "/" \
    --alpha 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 \
    --save "./"