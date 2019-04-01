import test
import sys
import torch.utils.data as tdata
sys.path.append('..')
from util.prim_ops_set import *
from util.datasets import get_dataset
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from util.encoder_colors import get_mask_pallete

class TestOperation(test.TestCase):
    def setUp(self):
        self.data_transform = transform.Compose([
            transform.ToTensor(),
            #transform.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    def test_op0(self):
        dst = get_dataset('promise12', split='train', mode='train', transform=self.data_transform)
        bs = 4
        trainloader = tdata.DataLoader(dst, batch_size=bs, num_workers=0)
        for i, data in enumerate(trainloader):
            imgs, labels = data
            imgs = imgs.numpy()[:, ::-1, :, :]
            imgs = np.transpose(imgs, [0, 2, 3, 1])
            if imgs.shape[-1] ==  1:
                imgs = imgs.squeeze(axis=-1)
            f, axarr = plt.subplots(bs, 2)
            for j in range(bs):
                axarr[j][0].imshow(imgs[j])
                axarr[j][1].imshow(get_mask_pallete(labels.numpy()[j]))
            plt.show()
            a = input()
            if a == "ex":
                break
            else:
                plt.close()

if __name__ == '__main__':
    test.main()