import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *
from itertools import cycle
import torchvision
import copy
import open_clip
import clip
import time
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset
from torch.distributed.elastic.multiprocessing.errors import record

import numpy as np

import torch

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments

from src.datasets.imagenet import ImageNet
from src.datasets.common import get_dataloader, maybe_dictionarize
import torch.nn.functional as F


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="training and testing batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for models")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of epochs")
    parser.add_argument("--cuda", action='store_true', help="use gpu or not")

    #student model
    parser.add_argument("--model", type=str, default="RN50")
    parser.add_argument("--pretrained", type=str, default="")  

    #teacher model
    parser.add_argument("--teacher-ckpt", type=str, default="")

    #disresnet34riburesnet34ed training
    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    
    #wise-ft
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )

    #teacher logits
    parser.add_argument(
        "--teacher-logits-path",
        type=str,
        default=None,
        help="Path to teacher logits",
    )
    return parser

def main(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    print("Device:")
    print(device)

    # create student model
    image_encoder_student_model, image_encoder_student_train_preprocess, image_encoder_student_val_preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrained, device=device, jit=False)


    # create teacher model
    model_teacher = ImageClassifier.load(args.teacher_ckpt)
    model_teacher = model_teacher.to(device)
    model_teacher.eval()

    #Data loading code
    imagenet_data = ImageNet(image_encoder_student_train_preprocess, args.data_location, args.batch_size)
    trainset = imagenet_data.train_dataset
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


    out_all = []
    targets_all = []
    for i, data in enumerate(tqdm(trainloader)):

        images = data['images']
        target = data['labels']

        images = images.to(device)
        
        with torch.no_grad():
            out = model_teacher(images)

        out = out.detach().cpu().numpy()
        target = target.numpy()

        #make sure out_old and out are different
        # assert not np.array_equal(out_old, out)

        out_all.extend(out)
        targets_all.extend(target)

    out_all = np.array(out_all)
    targets_all = np.array(targets_all)

    #save
    np.save(f"/scratch/mp5847/wise-ft-precompute/{args.model}-{args.pretrained}-imagenet-logits.npy", out_all)
    np.save(f"/scratch/mp5847/wise-ft-precompute/{args.model}-{args.pretrained}-imagenet-labels.npy", targets_all)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    
    main(args)