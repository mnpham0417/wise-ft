import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *
from itertools import cycle
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
    parser.add_argument("--cuda", action='store_true', help="use gpu or not")
    parser.add_argument("--name", type=str, default="test", help="name for experiment to save")
    parser.add_argument("--num_step", type=int, default=40)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="SGD")

    #student model
    parser.add_argument("--model", type=str, default="RN50")  
    parser.add_argument("--pretrained", type=str, default="")  

    

    #kd hyperparameters
    parser.add_argument("--T", type=float, default=4.0)

    #distributed training
    
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

    #teacher 
    parser.add_argument(
        "--teacher-logits-path",
        type=str,
        default=None,
        help="Path to teacher logits",
    )

    parser.add_argument(
        "--teacher-index-path",
        type=str,
        default=None,
        help="Path to teacher logits index",
    )

    #pretrained ckpt for resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from a checkpoint",
    )

    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default=None,
        help="Path to pretrained ckpt",
    )

    parser.add_argument(
        "--optimizer-ckpt",
        type=str,
        default=None,
        help="Path to optimizer ckpt",
    )

    parser.add_argument(
        "--scheduler-ckpt",
        type=str,
        default=None,
        help="Path to scheduler ckpt",
    )

    #loss type
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce_softlabels",
        help="type of loss",
    )
    return parser

class ImageNetDatasetOverrideLen(datasets.ImageFolder):
    def __init__(self, dataset, index_path):
        self.dataset = dataset
        self.index_list = np.load(index_path)
    def __getitem__(self, index):
        selected_index = self.index_list[index]
        return self.dataset[selected_index]
    def __len__(self):
        return len(self.index_list)

def test(model, test_dataloader):
    #test
    model.eval()
    total_acc = 0
    total = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_dataloader)):
            if(args.cuda):
                images = images.cuda()
                labels = labels.cuda()
            labels = labels.squeeze()
            
            pred = model(images)
            _, pred = torch.max(pred, 1)
            total_acc += (pred == labels).sum().item()
            total += labels.size(0)

    return total_acc/total

def main(args):
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

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
    model_student =  ImageClassifier(image_encoder_student, classification_head_student, process_images=True)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_student.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model_student = torch.nn.parallel.DistributedDataParallel(model_student, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model_student.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model_student = torch.nn.parallel.DistributedDataParallel(model_student, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model_student = model_student.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model_student.features = torch.nn.DataParallel(model_student.features)
            model_student.cuda()
        else:
            model_student = torch.nn.DataParallel(model_student).cuda()

    optimizer = torch.optim.SGD(model_student.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    cudnn.benchmark = True

    # Data loading code
    imagenet_data = ImageNet(image_encoder_student_train_preprocess, args.data_location, args.batch_size)
    trainset = imagenet_data.train_dataset
    if args.loss_type in ["ce_softlabels", "kl_softlabels", "mse_softlabels"]:
        trainset.set_teacher_logits(args.teacher_logits_path)
        if(args.teacher_index_path is not None):
            trainset = ImageNetDatasetOverrideLen(trainset, args.teacher_index_path)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None
        # val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     testset_cifar10, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # if args.evaluate:
    #     validate(val_loader, model, criterion, args)
    #     return

    if(args.loss_type == "ce_softlabels"):
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif(args.loss_type == "kl_softlabels"):
        criterion = nn.KLDivLoss(reduction="batchmean").cuda(args.gpu)
    elif(args.loss_type == "mse_softlabels"):
        criterion = nn.MSELoss().cuda(args.gpu)

    for epoch in range(args.start_epoch, args.epochs):
        # val_acc = validate(val_loader, student, nn.CrossEntropyLoss().cuda(args.gpu), args)

        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model_student, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            os.makedirs(f"/scratch/mp5847/wise-ft-kd/{args.name}", exist_ok=True)
            
            #save model
            torch.save(model_student.state_dict(), f"/scratch/mp5847/wise-ft-kd/{args.name}/model_{epoch}.pt")

            #save optimizer
            torch.save(optimizer.state_dict(), f"/scratch/mp5847/wise-ft-kd/{args.name}/optimizer_{epoch}.pt")

            #save scheduler
            torch.save(scheduler.state_dict(), f"/scratch/mp5847/wise-ft-kd/{args.name}/scheduler_{epoch}.pt")
        scheduler.step()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

@record
def train(train_loader, criterion, student, teacher, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')   
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    student.train()
    
    end = time.time()
    for idx, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        images = data['images']
        target = data['labels']
        
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            if(args.loss_type in ["ce_softlabels", "kl_softlabels", "mse_softlabels"]):
                teacher_out = data['teacher_logits']
                teacher_out = teacher_out.cuda(args.gpu, non_blocking=True)
        
        # compute output
        # with torch.no_grad():
        #     teacher_out = teacher(images)

        # if args.gpu is not None:
        #     teacher_out = teacher_out.cuda(args.gpu, non_blocking=True)

        student_out = student(images)

        if args.loss_type == "ce_softlabels":
            
            teacher_softmax = F.softmax(teacher_out, dim=1)
            loss = criterion(student_out, teacher_softmax)

        elif args.loss_type == "ce_hardlabels":
            loss = criterion(student_out, target)

        elif args.loss_type == "kl_softlabels":
            loss = criterion(F.log_softmax(student_out/args.T, dim=1), F.softmax(teacher_out/args.T, dim=1))*args.T*args.T

        elif args.loss_type == "mse_softlabels":
            student_out = F.softmax(student_out, dim=1)
            teacher_out = F.softmax(teacher_out, dim=1)
            loss = criterion(student_out, teacher_out)

        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx + 1)
    
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count]).cuda()
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)
                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    
    main(args)
