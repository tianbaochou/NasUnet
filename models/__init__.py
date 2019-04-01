from .fcn import *
from .fc_densenet import get_fc_densenet
from .psp import *
from .deeplab import *
from .unet import *
from .nas_unet import *
from .linknet import *
from .segnet import *


def get_segmentation_model(name, **kwargs):
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'deeplab': get_deeplab,
        'linknet': get_linknet,
        'unet': get_unet,
        'nasunet': get_nas_unet,
        'segnet': get_segnet,
        'fc_densenet': get_fc_densenet,
    }
    return models[name.lower()](**kwargs)
