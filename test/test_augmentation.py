import os
import sys
import unittest
import torchvision.transforms as transform
sys.path.append('..')
from torch.utils import data
from util.augmentations import *
from util.datasets import promise12
from util.datasets import get_dataset
import matplotlib.pyplot as plt

class TestAugmentation(unittest.TestCase):
    def setUp(self):
        # test augmentation use promise12 dataset
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

        # root = '../../../training_data/imageSeg/PROMISE2012'
        # data_path = os.path.join(root, 'npy_image')
        # self.X_train, self.y_train = promise12.load_val_data(data_path)
        # self.data = [(x,y) for x in self.X_train for y in self.y_train]

        kwargs = {'num_workers': 2, 'pin_memory': True}
        self.trainset = get_dataset('chaos', split='train', mode='train')
        self.train_queue = data.DataLoader(self.trainset, batch_size=2,
                                          drop_last=False, shuffle=True, **kwargs)

    def show(self):
        for step, (imgs, labels) in enumerate(self.train_queue):
            f, axarr = plt.subplots(2, 2)
            for i in range(2): # batch == 2
                axarr[i][0].imshow(imgs[0].numpy().squeeze(0), cmap='gray')
                axarr[i][1].imshow(labels[0].numpy(), cmap='gray')
            plt.show()
            a = input()
            if a == "ex":
                plt.close()
                break
            else:
                plt.close()

    def show1(self):
        for img, target in self.data:
            # convert img to PIL Image firstly
            img_tran = Image.fromarray(img, mode='F')
            target_tran = Image.fromarray(target, mode='L')
            img_orig = Image.fromarray(img, mode='F')
            target_orig = Image.fromarray(target, mode='L')

            if self.joint_transform is not None:
                img_tran, target_tran = self.joint_transform(img_tran, target_tran)
                target_tran = torch.from_numpy(np.array(target_tran)).long()

            if self.transform is not None:
                img_tran = self.transform(img_tran)

            img_orig, target_orig = tf.to_tensor(img_orig), tf.to_tensor(target_orig)

            f, axarr = plt.subplots(2, 2)
            axarr[0][0].imshow((img_orig.numpy().squeeze(axis=0)), cmap='gray')
            axarr[0][1].imshow(target_orig.numpy().squeeze(axis=0), cmap='gray')
            axarr[1][0].imshow((img_tran.numpy().squeeze(axis=0)), cmap='gray')
            axarr[1][1].imshow(target_tran.numpy(), cmap='gray')
            plt.show()
            a = input()
            if a == "ex":
                plt.close()
                break
            else:
                plt.close()

    def test_Pad(self):
        self.joint_transform = Compose([
            Pad(padding=20)
        ])
        self.show()

    def test_RandomRotate(self):
        self.joint_transform = Compose([
            RandomRotate(degree=10.0)
        ])
        self.show()

    def test_RandomTranslate(self):
        self.joint_transform = Compose([
            RandomTranslate(offset=(0.1, 0.1))
        ])
        self.show()

    def test_RandomVerticallyFlip(self):
        self.joint_transform = Compose([
            RandomVerticallyFlip()
        ])
        self.show()

    def test_RandomHorizontallyFlip(self):
        self.joint_transform = Compose([
            RandomHorizontallyFlip()
        ])
        self.show()

    def test_RandomElasticTransform(self):
        self.joint_transform = Compose([
            RandomElasticTransform(alpha = 3, sigma = 0.07)
        ])
        self.show()

    def test_RandomCrop(self):
        self.joint_transform = Compose([
            RandomCrop(224, 224)
        ])
        self.show()

    def test_AdjustGamma(self):
        self.joint_transform = Compose([
            AdjustGamma(224, 224)
        ])
        self.show()

    def test_AdjustSaturation(self):
        self.joint_transform = Compose([
            AdjustSaturation(saturation=0.2)
        ])
        self.show()

    def test_AdjustContrast(self):
        self.joint_transform = Compose([
            AdjustContrast(cf=0.2)
        ])
        self.show()


    def test_CenterCrop(self):
        self.joint_transform = Compose([
            CenterCrop(size=(224,224))
        ])
        self.show()

    def test_FreeScale(self):
        self.joint_transform = Compose([
            FreeScale(size=(224,224))
        ])
        self.show()

    def test_Scale(self):
        self.joint_transform = Compose([
            Scale(size=(224,224))
        ])
        self.show()

    def test_RandomSizedCrop(self):
        self.joint_transform = Compose([
            RandomSizedCrop(size=(224,224))
        ])
        self.show()

    def test_combined(self):
        self.show()

if __name__ == '__main__':
    unittest.main()