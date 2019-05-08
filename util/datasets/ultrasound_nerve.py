from __future__ import print_function
import os
import random
import numpy as np
from .base import BaseDataset
from PIL import Image
from util.augmentations import *

def make_dataset(root, dirname):
    base_path = os.path.join(root, dirname)
    images = os.listdir(base_path)
    images_list = []

    for image_name in images:
        if 'mask' in image_name:
            continue
        img_path = os.path.join(base_path, image_name)
        if dirname != 'test':
            image_mask_name = image_name.split('.')[0] + '_mask.png'
            img_mask_path = os.path.join(base_path, image_mask_name)
            images_list.append((img_path, img_mask_path))
        else:
            images_list.append((img_path, None))

    return images_list

class UltraNerve(BaseDataset):
    BASE_DIR = 'ultrasound-nerve'
    NUM_CLASS = 2
    IN_CHANNELS = 1
    CROP_SIZE = 256
    CLASS_WEIGHTS = None
    mean = [0.3919]
    std = [0.2212]
    def __init__(self, root,  split='train', mode=None, ft=False):
        super(UltraNerve, self).__init__(root, split, mode, norm = {'mu': self.mean, 'std': self.std})
        self.root = os.path.expanduser(root)
        self.ft = ft
        self.joint_transform = Compose([
            RandomTranslate(offset=(0.2, 0.2)),
            RandomVerticallyFlip(),
            RandomHorizontallyFlip(),
            RandomElasticTransform(alpha = 1.5, sigma = 0.07),
            ])
        base_path = os.path.join(self.root, self.BASE_DIR)

        if mode in ['train', 'val']:
            self.data_info = make_dataset(base_path, 'data_clean') # 'data_clean': after clean, before: 'train'
        else:
            self.data_info = make_dataset(base_path, 'test')

        if len(self.data_info) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
            "Supported image extensions are: " + ",".join('tif')))

    def __getitem__(self, index):
        img_path, target_path = self.data_info[index][0], self.data_info[index][1]

        img = Image.open(img_path).convert('L')
        if target_path != None:
            target = Image.open(target_path).convert('L')
        elif self.mode == 'test':
            target = img_path

        if not self.ft:
            # 1. do crop transform
            if self.mode == 'train':
                img, target = self.random_crop(img, target)
            elif self.mode == 'val':
                img, target = self.random_center_crop(img, target)

        # 2. do joint transform
        if self.mode != 'test' and self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        # 3. to tensor
        if self.mode != 'test':
            img, target = self.to_tensor(img, target)
        else:
            img = self.img2tensor(img)

        # 4. normalize for img
        img = self.img_normalize(img)

        if target is not None and not isinstance(target, str) :
            target[target == 255] = 1

        return img, target

    def __len__(self):
        return len(self.data_info)
