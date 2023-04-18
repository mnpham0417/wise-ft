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
import os
from collections import OrderedDict
import numpy as np

import torch

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments


def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            # key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            key: theta_0[key] + alpha * (theta_1[key] - theta_0[key])
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #Load a pre-trained model
    model_pretrained = timm.create_model('resnet50', pretrained=True)
    model_pretrained.eval()
    img_cls_pretrained = ImageClassifier(None, model_pretrained, process_images=False)
    img_cls_pretrained.val_preprocess = transform

    #Load a fine-tuned model
    model_finetuned_statedict = torch.load("/scratch/mp5847/wise-ft-kd/teacher=coca_ViT-B-32-pretrained=laion2B-s13B-b90k_student=RN50_timm_alpha=0.0_T=1.0/model_1.pt")
    new_model_finetuned_statedict = OrderedDict()
    for k, v in model_finetuned_statedict.items():
        name = k[7:]
        new_model_finetuned_statedict[name] = v

    model_finetuned = timm.create_model('resnet50', pretrained=False)
    model_finetuned.eval()
    img_cls_finetuned = ImageClassifier(None, model_finetuned, process_images=False)
    img_cls_finetuned.val_preprocess = transform
    img_cls_finetuned.load_state_dict(new_model_finetuned_statedict)

    theta_0 = {k: v.clone() for k, v in img_cls_pretrained.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in img_cls_finetuned.state_dict().items()}
    del img_cls_pretrained

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = [1.0]
    for alpha in alphas:
        args.alpha = alpha

        theta = _merge(alpha, theta_0, theta_1, None, args.fisher_floor)

        # update the model (in-place) acccording to the new weights
        img_cls_finetuned.load_state_dict(theta)

        # save model
        img_cls_finetuned.save(os.path.join(args.save, f'wise_ft_alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(img_cls_finetuned, args)
