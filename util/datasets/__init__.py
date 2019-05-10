from torchvision.datasets import *
from .base import *
from .coco import COCOSegmentation
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .minc import MINCDataset
from .ultrasound_nerve import UltraNerve
from .bladder import Bladder
from .chaos import CHAOS
from .promise12 import Promise12
from .camvid import  CamVid

datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'minc': MINCDataset,
    'cifar10': CIFAR10,
    'ultrasound_nerve': UltraNerve,
    'bladder': Bladder,
    'chaos' : CHAOS,
    'promise12': Promise12,
    'camvid': CamVid
}

acronyms = {
    'coco': 'coco',
    'pascal_voc': 'voc',
    'pascal_aug': 'voc',
    'pcontext': 'pcontext',
    'ade20k': 'ade',
    'citys': 'citys',
    'minc': 'minc',
    'cifar10': 'cifar10',
    'ultrasound_nerve': 'ultrasound_nerve',
    'bladder': 'bladder',
    'chaos': 'chaos',
    'promise12': 'promise12',
    'camvid': 'camvid'
}

dir = '/train_tiny_data/imgseg/'
#=
#dir = '../../../training_data/imageSeg/'

def get_dataset(name, path=dir, **kwargs):
    return datasets[name.lower()](root = path, **kwargs)

