from __future__ import print_function
import os
from .base import BaseDataset
from PIL import Image
import re

def make_dataset(root, dirname):
    base_path = os.path.join(root, dirname)
    ids = (re.findall(r"\d+", f[:-4])[0] for f in os.listdir(base_path))
    images_list = []

    for id in ids:
        img_path = os.path.join(base_path, 'IM'+id+'.png')
        img_mask_path = os.path.join(base_path+'/../Label', 'Label'+id+'.png')
        images_list.append((img_path, img_mask_path))

    return images_list

class Bladder(BaseDataset):
    BASE_DIR = 'bladder'
    NUM_CLASS = 3
    IN_CHANNELS = 1
    CROP_SIZE = 512
    mean = [0.1355]
    std = [0.1348]
    CLASS_WEIGHTS = None
    def __init__(self, root,  split='train', mode=None):
        super(Bladder, self).__init__(root, split, mode, norm = {'mu': self.mean, 'std': self.std})
        self.root = os.path.expanduser(root)
        self.joint_transform = None
        base_path = os.path.join(self.root, self.BASE_DIR)

        if mode in ['train', 'val']:
            self.data_info = make_dataset(base_path, 'Train')
            if len(self.data_info) == 0:
                raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                "Supported image extensions are: " + ",".join('png')))
        else:
            self.data_info = []

    def __getitem__(self, index):

        if len(self.data_info) == 0: # empty test dataset
            return None,None

        img_path, target_path = self.data_info[index][0], self.data_info[index][1]

        img = Image.open(img_path)
        target = Image.open(target_path)

        # 1. do crop transform
        if self.mode == 'train':
            img, target = self.random_crop(img, target)
        elif self.mode == 'val':
            img, target = self.random_center_crop(img, target)

        # 2. do joint transform
        if self.mode != 'test' and self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        # 3. to tensor
        img, target = self.to_tensor(img, target)

        # 4. normalize for img
        img = self.img_normalize(img)

        # convert label to 0, 1 and 2
        target[target == 128] = 1
        target[target == 255] = 2

        return img, target

    def __len__(self):
        return len(self.data_info)
