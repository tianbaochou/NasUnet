from __future__ import print_function
import os
from .base import BaseDataset
import pydicom
from util.augmentations import *
from util.utils import create_class_weight

def make_dataset(root, dirname, type='CT', is_dup=False):
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

def histogram(image):
    hist = dict()

    # Initialize dict
    for shade in range(0, 256):
        hist[shade] = 0

    for index, val in np.ndenumerate(image):
        hist[val] += 1

    return hist

def shade_at_percentile(hist, percentile):
    n = sum(hist.values())
    cumulative_sum = 0.0
    for shade in range(0, 256):
        cumulative_sum += hist[shade]
        if cumulative_sum / n >= percentile:
            return shade

    return None

def auto_contrast(image):
    """ Apply auto contrast to an image using
        https://stackoverflow.com/questions/9744255/instagram-lux-effect/9761841#9761841
    """
    hist = histogram(image)
    p5 = shade_at_percentile(hist, .01)
    p95 = shade_at_percentile(hist, .99)
    a = 255.0 / (p95 + p5)
    b = -1.0 * a * p5

    result = (image.astype(float) * a) + b
    result = result.clip(0, 255.0)

    return image

def extract_grayscale_image(dicom_data):
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


class CHAOS(BaseDataset):
    TYPE = 'CT'
    BASE_DIR, NUM_CLASS, CROP_SIZE = ('CHAOS/CT_data_batch/', 2, 256) if TYPE=='CT'\
        else ('CHAOS/MR_data_batch1/', 5, 256)
    IN_CHANNELS = 1
    CLASS_WEIGHTS = None
    mean = [0.2389]
    std = [0.2801]
    def __init__(self, root,  split='train', mode=None):
        super(CHAOS, self).__init__(root, split, mode, norm = {'mu': self.mean, 'std': self.std})
        self.root = os.path.expanduser(root)
        self.joint_transform = None
        # The MR image is too challenge!
        self.joint_transform = Compose([
            RandomTranslate(offset=(0.3, 0.3)),
            RandomVerticallyFlip(),
            RandomHorizontallyFlip(),
            RandomElasticTransform(alpha=1.5, sigma=0.07),
        ])
        base_path = os.path.join(self.root, self.BASE_DIR)
        self.data_info = []

        if mode in ['train', 'val']:
            dirnames = os.listdir(base_path)
            if self.TYPE == 'MR':
                for dir in dirnames:
                    if dir == 'notes.txt':
                        continue
                    self.data_info += make_dataset(base_path, dir + '/T1DUAL', type='MR', is_dup=True)
                    self.data_info += make_dataset(base_path, dir + '/T2SPIR', type='MR')
            else:
                for dir in dirnames:
                    if dir == 'notes.txt':
                        continue
                    self.data_info += make_dataset(base_path, dir)

            if len(self.data_info) == 0:
                raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                "Supported image extensions are: " + ",".join('png')))
        else:
            self.data_info = []
        if self.TYPE == 'MR':
            self.CLASS_WEIGHTS = torch.tensor(self._generate_weight())

    def _generate_weight(self):
        weight_dict = {0:0., 80:0., 160:0., 240:0, 255:0.}
        areas = 256.*256.
        for index in range(len(self.data_info)):
            img_path, target_path = self.data_info[index][0], self.data_info[index][1]
            target = Image.open(target_path).convert('L')
            target_np = np.array(target)
            unique, counts = np.unique(target_np, return_counts=True)
            dict_c = dict(zip(unique, counts))
            for id, val in dict_c.items():
                weight_dict[id] += val / areas

        weight_list = [weight_dict[k] for k in sorted(weight_dict.keys())]
        return create_class_weight(weight_list)

    def __getitem__(self, index):
        if len(self.data_info) == 0:
            return None, None

        img_path, target_path = self.data_info[index][0], self.data_info[index][1]

        # Read image
        dataset = pydicom.dcmread(img_path)
        target = Image.open(target_path).convert('L')
        if 'PixelData' in dataset:
            if dataset.Modality == "CT":  # size: 512x512
                img, itercept = dataset.RescaleSlope * dataset.pixel_array + dataset.RescaleIntercept, dataset.RescaleIntercept
                img[img >= 4000] = itercept  # remove abnormal pixel
            else:  # size: 256x256
                img = extract_grayscale_image(dataset)
                img = auto_contrast(img)

            img = Image.fromarray(img).convert('L')

        # 1. do crop transform
        if self.mode == 'train':
            img, target = self.random_crop(img, target)
        elif self.mode == 'val':
            img, target = self.random_center_crop(img, target)

        # 2. do joint transform
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        ## 3.to tensor
        img, target = self.to_tensor(img, target)

        # 4. normalize for img
        img = self.img_normalize(img)

        if self.TYPE == 'CT':
            # Convert label to 0, 1
            target[target == 255] = 1
        else:
            target[target == 80] = 1
            target[target == 160] = 2
            target[target == 240] = 3
            target[target == 255] = 4

        return img, target

    def __len__(self):
        return len(self.data_info)
