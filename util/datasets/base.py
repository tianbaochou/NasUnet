###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from util.augmentations import *
import torch
import torch.utils.data as data

__all__ = ['BaseDataset']

class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, norm={'mu':[.485, .456, .406], 'std':[.229, .224, .225]}):
        self.root = root
        self.random_crop = RandomSizedCrop(size=self.crop_size)
        self.random_center_crop = CenterCrop(size=self.crop_size)
        self.img_normalize = transforms.Normalize(norm['mu'], norm['std'])
        self.test_resize = transforms.Resize(size=self.crop_size)
        self.to_tensor = ToTensor()
        self.img2tensor = tf.to_tensor
        self.split = split
        self.mode = mode
        if self.mode == 'train':
            print('BaseDataset: crop_size {}'. format(self.CROP_SIZE))

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def class_weight(self):
        return self.CLASS_WEIGHTS

    @property
    def base_dir(self):
        return self.BASE_DIR

    @property
    def root_dir(self):
        return self.root

    @property
    def in_channels(self):
        return self.IN_CHANNELS

    @property
    def crop_size(self):
        return self.CROP_SIZE

    @property
    def pred_offset(self):
        raise NotImplemented

    def make_pred(self, x):
        return x + self.pred_offset


