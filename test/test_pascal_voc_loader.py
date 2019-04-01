import os
import sys
import numpy as np
import collections
import matplotlib.pyplot as plt
import util.augmentations as aug
from util.loader import get_loader
from torch.utils import data
import torchvision.transforms as transforms

if __name__ == '__main__':
    local_path = '../../../training_data/imageSeg/VOCdevkit/VOC2012/'
    bs = 4
    if not os.path.exists(local_path):
        print('path not exist!!!')
        sys.exit(-1)
    augs = transforms.Compose([transforms.RandomRotation(10),
                               transforms.RandomHorizontalFlip()])
    pasvoc_loader = get_loader('pascal')
    dst = pasvoc_loader(root=local_path, is_transform=True, augmentations=augs)
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs): # show a batch images
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()
