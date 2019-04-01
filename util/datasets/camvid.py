import os
import torch
from .base import BaseDataset
from util.augmentations import *

classes = ['Sky', 'Building', 'Column-Pole', 'Road',
           'Sidewalk', 'Tree', 'Sign-Symbol', 'Fence', 'Car', 'Pedestrain',
           'Bicyclist', 'Void']


class_color = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
    (0, 0, 0),
]

def _make_dataset(dir):
    images_list = []
    if not os.path.exists(os.path.join(dir)):
        print('The dir {} is not exists'.format(dir))
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if os.path.exists(os.path.join(root, fname)):
                img_path = os.path.join(root, fname)
                img_mask_path = os.path.join(root+'annot', fname)
                images_list.append((img_path, img_mask_path))
    return images_list


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().long()
        return label


class LabelTensorToPILImage(object):
    def __call__(self, label):
        label = label.unsqueeze(0)
        colored_label = torch.zeros(3, label.size(1), label.size(2)).byte()
        for i, color in enumerate(class_color):
            mask = label.eq(i)
            for j in range(3):
                colored_label[j].masked_fill_(mask, color[j])
        npimg = colored_label.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        mode = None
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]
            mode = "L"

        return Image.fromarray(npimg, mode=mode)

class CamVid(BaseDataset):
    BASE_DIR = 'CamVid'
    NUM_CLASS = 12
    IN_CHANNELS = 3
    CROP_SIZE = 256
    # https://github.com/yandex/segnet-torch/blob/master/datasets/camvid-gen.lua
    CLASS_WEIGHTS = torch.tensor([
        0.58872014284134, 0.51052379608154, 2.6966278553009,
        0.45021694898605, 1.1785038709641, 0.77028578519821,
        2.4782588481903, 2.5273461341858, 1.0122526884079,
        3.2375309467316, 4.1312313079834, 0.0])

    mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    std = [0.27413549931506, 0.28506257482912, 0.28284674400252]
    def __init__(self, root, split='train', mode=None, ft=False):

        super(CamVid, self).__init__(root, split, mode, norm={'mu':self.mean,'std':self.std})

        self.root = os.path.expanduser(root)
        base_path = os.path.join(self.root, self.BASE_DIR)
        self.ft = ft
        assert split in ('train', 'val', 'test')
        self.split = split

        self.joint_transform = Compose([
                RandomHorizontallyFlip(),
            ])

        self.classes = classes

        self.imgs = _make_dataset(os.path.join(base_path, self.split))

    def __getitem__(self, index):
        img_path, target_path = self.imgs[index][0], self.imgs[index][1]
        img = Image.open(img_path)
        target = Image.open(target_path)

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
        img, target = self.to_tensor(img, target)

        # 4. normalize for img
        img = self.img_normalize(img)

        return img, target

    def __len__(self):
        return len(self.imgs)



