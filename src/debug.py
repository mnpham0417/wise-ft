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
    parser.add_argument("--name", type=str, default="test", help="name for experiment to save")
    parser.add_argument("--num_step", type=int, default=40)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="SGD")

    #student model
    parser.add_argument("--model", type=str, default="RN50")  
    parser.add_argument("--pretrained", type=str, default="")  

    #teacher model
    parser.add_argument("--teacher-ckpt", type=str, default="")

    #disresnet34riburesnet34ed training
    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    
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


def get_flatten_weights(model):
    weights = []
    for name, param in model.named_parameters():
        weights.append(param.clone().reshape(-1))

    shapes = []
    for p in model.parameters():
        if p.requires_grad:
            if len(p.shape) == 0:
                shapes.append([1])
            else:
                shapes.append(p.shape)

    return torch.cat(weights, dim=0), shapes

def load_weights(model, flat_w, shapes):
    '''
    Function that takes in a model, a flattened vector of weights, and a list of shapes, and loads the weights into the model
    @param model: model to load weights into
    @param flat_w: flattened vector of weights
    @param shapes: list of shapes
    @return: None
    '''

    index = 0
    shape_index = 0
    for name, param in model.named_parameters():
        if shape_index >= len(shapes):
            raise IndexError("Not enough shapes to match all parameters")
        shape = shapes[shape_index]
        if(len(shape) == 0):
            continue
        size = int(torch.tensor(shape).prod().item())

        # create a new tensor from the flattened vector of weights, with the correct shape and data type
        weight_tensor = flat_w[index:index + size].reshape(*shape).to(param.device).type(param.dtype)

        # use nested attribute names with setattr() to set the value of the parameter
        keys = name.split('.')
        obj = model
        for key in keys[:-1]:
            obj = getattr(obj, key)
        setattr(obj, keys[-1], torch.nn.Parameter(weight_tensor, requires_grad=param.requires_grad))

        index += size
        shape_index += 1


def create_model(args, num_classes=10):
    '''
    Get a randomly initialized model
    '''
    image_encoder = ImageEncoder(args, keep_lang=False)
    
    if(args.model == "ViT-B-32"):
        classification_head = nn.Linear(512, num_classes)
    elif(args.model == "RN50"):
        classification_head = nn.Linear(1024, num_classes)
    elif(args.model == "RN101"):
        classification_head = nn.Linear(512, num_classes)
    elif(args.model == "ViT-B-16"):
        classification_head = nn.Linear(512, num_classes)
    elif(args.model == "ViT-L-14"):
        classification_head = nn.Linear(768, num_classes)
    model =  ImageClassifier(image_encoder, classification_head, process_images=True)

    # change_model_name(model)

    return model, image_encoder.train_preprocess, image_encoder.val_preprocess

def change_model_name(model):
    #iterate through all model parameters name, change . to _ and set as attribute
    for name, param in model.named_parameters():
        name = name.replace(".", "_")
        setattr(model, name, param)

def main(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    # device = torch.device("cpu")
    args.device = device

    print("Device:")
    print(device)

    # create student model
    image_encoder_student_model, image_encoder_student_train_preprocess, image_encoder_student_val_preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrained, device=device, jit=False)

    # create teacher model
    # model_teacher = ImageClassifier.load(args.teacher_ckpt)
    # model_teacher = model_teacher.to(device)
    # model_teacher.eval()

    # change_model_name(model_teacher)

    #Data loading code

    #cifar10
    trainset = torchvision.datasets.CIFAR10(root='/scratch/mp5847/cifar', train=True,
                                            download=True, transform=image_encoder_student_train_preprocess)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.workers, pin_memory=True)
    
    model_random1, _, _ = create_model(args)
    flat_w1 = model_random1.get_param()
    model_random2, _, _ = create_model(args)
    flat_w2 = model_random2.get_param()

    data = trainset[0][0].unsqueeze(0).to(device)

    out1 = model_random1.forward_with_param(data, flat_w1)

    loss = torch.sum(out1)
    # model_random = model_random.to(device)
    print(out1.shape)
    gw1, = torch.autograd.grad(loss, flat_w1, create_graph=True, allow_unused=True)
    print(gw1.shape)
    # flat_w_random, shape_w_random = get_flatten_weights(model_random)
    # flat_w, shape_w = get_flatten_weights(model_teacher)

    # out = model_teacher(trainset[0][0].unsqueeze(0).to(device))

    # weights = get_flatten_weights(model_teacher)[0]

    # loss = torch.sum(out)

    # print(weights.shape)

    # gw = torch.autograd.grad(loss, weights[100], create_graph=True, allow_unused=True)

    # print(gw)

    # print(flat_w_random.shape, flat_w.shape)

    # print("Distance of original weights and random model:")
    # print(torch.dist(flat_w, flat_w_random))

    # #load random model the weights of teacher
    # load_weights(model_random, flat_w, shape_w)

    # flat_w_random, shape_w_random = get_flatten_weights(model_random)

    # print(flat_w_random.shape, flat_w.shape)
    # print("Distance of original weights and random model (loaded weights):")
    # print(torch.dist(flat_w, flat_w_random))




if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    
    main(args)
