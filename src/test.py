import os
import numpy as np
import torch
import timm
from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
import torchvision.transforms as transforms


if __name__ == '__main__':
    args = parse_arguments()

    #Load a pre-trained model
    model = timm.create_model('resnet50', pretrained=True)

    # Set the model to evaluation mode
    model.eval()

    img_cls = ImageClassifier(None, model, process_images=False)
    img_cls.load_state_dict(torch.load(args.checkpoint))


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_cls.val_preprocess = transform

    # evaluate
    evaluate(img_cls, args)
