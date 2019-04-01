import unittest
from util.utils import *
from util.datasets import get_dataset
from torch.utils import data
import matplotlib.pyplot as plt
class TestUtils(unittest.TestCase):
    def setUp(self):
        kwargs = {'num_workers': 2, 'pin_memory': True}
        self.trainset = get_dataset('ultrasound_nerve', split='train', mode='train')
        self.train_queue = data.DataLoader(self.trainset, batch_size=2,
                                          drop_last=False, shuffle=True, **kwargs)

    def show(self, imgs, cls=1):
        # convert img to PIL Image firstly
        label, label_new = imgs[0], imgs[1]
        N = label.size()[0]
        f, axarr = plt.subplots(N, cls+1)
        for i in range (N):
            label_i = label[i]
            label_new_i = label_new[i]
            axarr[i][0].imshow(label_i.numpy(), cmap='gray')
            for j in range(cls):
                label_new_ij = label_new_i[j]
                axarr[i][j+1].imshow(label_new_ij.numpy(), cmap='gray')
        plt.show()

    def test_onehot_encoder(self):
        # Note: For a binary label, one hot encoding : first channels, the
        # background is 255, target is 0 and second channels, the background is 0
        # target is 255
        for step, (_, labels) in enumerate(self.train_queue):
            label_new = one_hot_encoding(labels, self.trainset.num_class)
            self.show([labels, label_new], self.trainset.num_class)
            a = input()
            if a == "ex":
                plt.close()
                break
            else:
                plt.close()
if __name__ == '__main__':
    unittest.main()