cd /home/mp5847/src/wise-ft; export PYTHONPATH="$PYTHONPATH:$PWD"; \

python src/wise_ft.py   \
        --train-dataset=ImageNet  \
        --epochs=1  \
        --lr=0.00003  \
        --batch-size=32  \
        --cache-dir=cache  \
        --model="xlm-roberta-large-ViT-H-14"  \
        --pretrained=frozen_laion5b_s13b_b90k  \
        --template=openai_imagenet_template  \
        --results-db=results.jsonl  \
        --save=/scratch/mp5847/wise-ft-ckpt/xlm-roberta-large-ViT-H-14-pretrained=frozen_laion5b_s13b_b90k  \
        --data-location=/ \
        --alpha 0 