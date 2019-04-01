import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

import torch
from .base import BaseDataset

class VOCSegmentation(BaseDataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]
    NUM_CLASS = 21
    IN_CHANNELS = 3
    CROP_SIZE = 256
    BASE_DIR = 'VOCdevkit/VOC2012'
    CLASS_WEIGHTS = None
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train', mode=None):

        super(VOCSegmentation, self).__init__(root, split, mode)
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        self.joint_transform = None
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if self.mode == 'train':
            _split_f = os.path.join(_splits_dir, 'trainval.txt')
        elif self.mode == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif self.mode == 'test':
            self.images = []
            return
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])

        # 1. do crop transform
        if self.mode == 'train':
            img, target = self.random_crop(img, target)
        elif self.mode == 'val':
            img, target = self.random_center_crop(img, target)

        # 2. do joint transform
        if  self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        # 3. to tensor
        img, target = self.to_tensor(img, target)

        # 4. normalize for img
        img = self.img_normalize(img)

        target[target == 255] = 0

        return img, target

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0
