'''
An tool for computing mean and std for torch.Normalization
zbabby: 2019/05/08

Usage:
    # the root path for segmentation dataset
    dir = '../../../../training_data/imageSeg/'

    # Example1:
    # For ultrasound chaos dataset
    # dataset = DicomFormatDataset(root=dir)

    # Example 2:
    # For ultrasound nerve dataset or bladder dataset
    dataset = NormalFormatDataset(root=dir)
    loader = DataLoader(
        dataset,
        batch_size=256, # Make sure you have adequate cpu memory! otherwise, set down the batch size
        num_workers=4,
        shuffle=False
    )
    mean, std = CalcMeanAndStd(loader, n_channels=1)

Calculate your dataset is much easier when you look at our example!
'''
import os
import torch
import re
import pydicom
from PIL import Image
import numpy as np
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader
import torch.utils.data as data

class DicomFormatDataset(data.Dataset):
    def __init__(self, root):
        self.root = os.path.expanduser(root)

        BASE_DIR1 = 'CHAOS/CT_data_batch/'
        BASE_DIR2 = 'CHAOS/MR_data_batch1/'
        base_path1 = os.path.join(self.root, BASE_DIR1)
        base_path2 = os.path.join(self.root, BASE_DIR2)
        self.data_info = []
        dirnames = os.listdir(base_path2)
        for dir in dirnames: # MR
            if dir == 'notes.txt':
                continue
            self.data_info += self.make_dataset_chaos(base_path2, dir + '/T1DUAL', type='MR', is_dup=True)
            self.data_info += self.make_dataset_chaos(base_path2, dir + '/T2SPIR', type='MR')
        dirnames = os.listdir(base_path1)
        for dir in dirnames: # CT
            if dir == 'notes.txt':
                continue
            self.data_info += self.make_dataset_chaos(base_path1, dir)

    """
    For dcm format images
    """
    def __getitem__(self, index):
        if len(self.data_info) == 0:
            return None, None
        img_path = self.data_info[index][0]
        # Read image
        dataset = pydicom.dcmread(img_path)

        if 'PixelData' in dataset:
            if dataset.Modality == "CT":  # size: 512x512
                img, itercept = dataset.RescaleSlope * dataset.pixel_array + dataset.RescaleIntercept, dataset.RescaleIntercept
                img[img >= 4000] = itercept  # remove abnormal pixel
            else:
                img = self.extract_grayscale_image(dataset)
                img = self.auto_contrast(img)

            img = Image.fromarray(img).convert('L')
        img = tf.center_crop(img, 256)
        img = tf.to_tensor(img)
        return img

    def histogram(self, image):
        hist = dict()
        # Initialize dict
        for shade in range(0, 256):
            hist[shade] = 0
        for index, val in np.ndenumerate(image):
            hist[val] += 1
        return hist

    def shade_at_percentile(self, hist, percentile):
        n = sum(hist.values())
        cumulative_sum = 0.0
        for shade in range(0, 256):
            cumulative_sum += hist[shade]
            if cumulative_sum / n >= percentile:
                return shade
        return None

    def auto_contrast(self, image):
        """ Apply auto contrast to an image using
            https://stackoverflow.com/questions/9744255/instagram-lux-effect/9761841#9761841
        """
        hist = self.histogram(image)
        p5 = self.shade_at_percentile(hist, .01)
        p95 = self.shade_at_percentile(hist, .99)
        a = 255.0 / (p95 + p5)
        b = -1.0 * a * p5

        result = (image.astype(float) * a) + b
        result = result.clip(0, 255.0)

        return image

    def extract_grayscale_image(self, dicom_data):
        # Extracting data from the mri file
        plan = dicom_data
        shape = plan.pixel_array.shape
        # Convert to float to avoid overflow or underflow losses.
        image_2d = plan.pixel_array.astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)

        return image_2d_scaled

    def make_dataset_chaos(self, root, dirname, type='CT', is_dup=False):
        base_path = os.path.join(root, dirname)
        images_path = os.path.join(root, dirname, 'DICOM_anon')
        mask_path = os.path.join(root, dirname, 'Ground')

        images = os.listdir(images_path)
        images_list = []
        for image_name in images:
            if type == 'CT': # two types batch
                if 'IMG' in image_name:
                    image_mask_name = 'liver_GT_' + image_name[:-4].split('-')[-1][2:] + '.png'
                else:
                    image_mask_name = 'liver_GT_' + image_name[:-4].split(',')[0][2:] + '.png'
            else:
                M = image_name[:-4].split('-')[-1]
                id = "%03d" % ((int(M)+1) // 2) if is_dup else M[2:]
                image_mask_name = 'liver_' + id + '.png'
            img_path = os.path.join(images_path, image_name)
            img_mask_path = os.path.join(mask_path, image_mask_name)
            images_list.append((img_path, img_mask_path))

        return images_list

    def __len__(self):
        return len(self.data_info)

class NormalFormatDataset(data.Dataset):
    def __init__(self, root):
        self.root = os.path.expanduser(root)
        self.joint_transform = None

        # ultrasound nerve
        BASE_DIR = 'ultrasound-nerve'
        base_path = os.path.join(self.root, BASE_DIR)
        self.data_info = self.make_dataset_nerve(base_path, 'data_clean')
        self.data_info += self.make_dataset_nerve(base_path, 'test')
        if len(self.data_info) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
            "Supported image extensions are: " + ",".join('png')))

        # bladder
        # BASE_DIR = 'bladder'
        # base_path = os.path.join(self.root, BASE_DIR)
        # self.data_info = self.make_dataset_bladder(base_path, 'data_clean')

    """
    For simple jpg, png format images
    """
    def __getitem__(self, index):
        if len(self.data_info) == 0:
            return None

        img_path = self.data_info[index]
        img = Image.open(img_path)
        img = tf.to_tensor(img)
        return img

    def make_dataset_nerve(self, root, dirname):
        base_path = os.path.join(root, dirname)
        images = os.listdir(base_path)
        images_list = []

        for image_name in images:
            if 'mask' in image_name:
                continue
            img_path = os.path.join(base_path, image_name)
            if dirname != 'test':
                images_list.append(img_path)
            else:
                images_list.append(img_path)

        return images_list

    def make_dataset_bladder(self, root, dirname):
        base_path = os.path.join(root, dirname)
        ids = (re.findall(r"\d+", f[:-4])[0] for f in os.listdir(base_path))
        images_list = []
        for id in ids:
            img_path = os.path.join(base_path, 'IM'+id+'.png')
            images_list.append(img_path)
        return images_list

    def __len__(self):
        return len(self.data_info)

def CalcMeanAndStd(loader, n_channels):
    '''
    Var[x] = E[x^2] - E^2[x]
    :param loader:
    :return:
    '''
    cnt = 0
    ford_moment = torch.empty(n_channels)
    sord_moment = torch.empty(n_channels)

    for data in loader:
        print('process: {} pixels'.format(cnt))
        b, c, h, w = data.shape
        nb_pixels = b*h*w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data**2, dim = [0, 2, 3])
        ford_moment = (cnt * ford_moment + sum_) / (cnt + nb_pixels)
        sord_moment = (cnt * sord_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    return ford_moment, torch.sqrt(sord_moment - ford_moment ** 2)

if __name__ == '__main__':
    dir = '../../../../training_data/imageSeg/'

    # Example1:
    # For ultrasound chaos dataset
    dataset = DicomFormatDataset(root=dir)

    # Example 2:
    # For ultrasound nerve dataset or bladder dataset
    #dataset = NormalFormatDataset(root=dir)

    loader = DataLoader(
        dataset,
        batch_size=256, # Make sure you have adequate cpu memory! otherwise, set down the batch size
        num_workers=4,
        shuffle=False
    )
    mean, std = CalcMeanAndStd(loader, n_channels=1)
    print('mean: {}, std: {}'.format(mean, std))
