import logging
import torchvision.transforms as transforms
from util.augmentations.augmentation import *

logger = logging.getLogger('nas_seg')

key2aug = {'gamma': AdjustGamma,
           'hue': AdjustHue,
           'brightness': AdjustBrightness,
           'saturation': AdjustSaturation,
           'contrast': AdjustContrast,
           'translate': RandomTranslate,
           'rcrop': transforms.RandomCrop,
           'hflip': transforms.RandomHorizontalFlip,
           'vflip': transforms.RandomVerticalFlip,
           'scale': transforms.Scale,
           'rsize': transforms.Resize,
           'rsizecrop': transforms.RandomSizedCrop,
           'rotate': transforms.RandomRotation,
           'ccrop': transforms.CenterCrop,}

def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return transforms.Compose(augmentations)


