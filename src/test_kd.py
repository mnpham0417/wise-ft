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
from collections import OrderedDict

def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta


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

    if(args.model == "ViT-B-32"):
        classification_head_student = nn.Linear(512, 1000)
    elif(args.model == "RN50"):
        classification_head_student = nn.Linear(1024, 1000)
    elif(args.model == "RN101"):
        classification_head_student = nn.Linear(512, 1000)
    elif(args.model == "ViT-B-16"):
        classification_head_student = nn.Linear(512, 1000)
    elif(args.model == "ViT-L-14"):
        classification_head_student = nn.Linear(768, 1000)
    model_pretrained =  ImageClassifier(image_encoder_student, classification_head_student, process_images=True)

    state_dict = torch.load("/scratch/mp5847/wise-ft-kd/rn50_scratch_0/model_99.pt")

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model_pretrained.load_state_dict(new_state_dict)
    model_pretrained.to(device)

    # create student model
    image_encoder_student_model, image_encoder_student_train_preprocess, image_encoder_student_val_preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrained, device=device, jit=False)
    image_encoder_student = ImageEncoder(args, keep_lang=True)
    image_encoder_student.model = image_encoder_student_model
    image_encoder_student.train_preprocess = image_encoder_student_train_preprocess
    image_encoder_student.val_preprocess = image_encoder_student_val_preprocess

    if(args.model == "ViT-B-32"):
        classification_head_student = nn.Linear(512, 1000)
    elif(args.model == "RN50"):
        classification_head_student = nn.Linear(1024, 1000)
    elif(args.model == "RN101"):
        classification_head_student = nn.Linear(512, 1000)
    elif(args.model == "ViT-B-16"):
        classification_head_student = nn.Linear(512, 1000)
    elif(args.model == "ViT-L-14"):
        classification_head_student = nn.Linear(768, 1000)
    model_finetuned =  ImageClassifier(image_encoder_student, classification_head_student, process_images=True)

    state_dict = torch.load("/scratch/mp5847/wise-ft-kd/teacher=RN50-pretrained=ImageNet_student=RN50_alpha=0.0_T=1.0/model_10.pt")

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model_finetuned.load_state_dict(new_state_dict)
    model_finetuned.to(device)

    theta_0 = {k: v.clone() for k, v in model_pretrained.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in model_finetuned.state_dict().items()}
    del model_pretrained

    if args.fisher is None:
        fishers = None
    else:
        fisher_0_file, fisher_1_file = args.fisher
        fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
        fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
        fishers = fisher_0, fisher_1

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = args.alpha
    for alpha in alphas:
        args.alpha = alpha

        theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)

        # update the model (in-place) acccording to the new weights
        model_finetuned.load_state_dict(theta)

        # save model
        model_finetuned.save(os.path.join(args.save, f'wise_ft_alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(model_finetuned, args)