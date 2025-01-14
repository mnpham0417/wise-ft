import os

import numpy as np

import torch

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
import open_clip
import torch.nn as nn
import copy

if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     # create student model
    image_encoder_student_model, image_encoder_student_train_preprocess, image_encoder_student_val_preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrained, device=device, jit=False)
    image_encoder_student = ImageEncoder(args, keep_lang=True)
    image_encoder_student.model = image_encoder_student_model
    image_encoder_student.train_preprocess = image_encoder_student_train_preprocess
    image_encoder_student.val_preprocess = image_encoder_student_val_preprocess
    classification_head_student = get_zeroshot_classifier(args, image_encoder_student_model)
    # classification_head_student = nn.Linear(512, 1000)
    model_student =  ImageClassifier(image_encoder_student, classification_head_student, process_images=True)

    state_dict = torch.load(args.checkpoint)

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    #load teacher
    model_tmp = copy.deepcopy(model_student)
    model_tmp.load_state_dict(new_state_dict)

    # model_student.load_state_dict(new_state_dict)
    # model_student.classification_head = classification_head_student
    model_student.image_encoder = model_tmp.image_encoder

    # evaluate
    evaluate(model_student, args)
