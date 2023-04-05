import os

import numpy as np

import torch

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments

if __name__ == '__main__':
    args = parse_arguments()

    model = ImageClassifier.load(args.checkpoint)

    # evaluate
    evaluate(model, args)
