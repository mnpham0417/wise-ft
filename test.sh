cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

# python src/test.py   \
#     --eval-datasets=ImageNet,ImageNetV2  \
#     --checkpoint "/scratch/mp5847/wise-ft-ckpt/ViT-B-32-pretrained=openai/finetuned/wise_ft_alpha=0.000.pt"  \
#     --data-location "/" 

python src/tmp.py   \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet  \
    --results-db=results.jsonl  \
    --data-location=/ \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --model RN50 \
    --pretrained "" \
    --save ""