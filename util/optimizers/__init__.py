import copy
import logging
import functools

from torch.optim import SGD
from torch.optim import Adam
from torch.optim import ASGD
from torch.optim import Adamax
from torch.optim import Adadelta
from torch.optim import Adagrad
from torch.optim import RMSprop
from adabound import AdaBound

logger = logging.getLogger('nas_seg')

key2opt =  {'sgd': SGD,
            'adam': Adam,
            'asgd': ASGD,
            'adamax': Adamax,
            'adadelta': Adadelta,
            'adagrad': Adagrad,
            'rmsprop': RMSprop,
            'adabound': AdaBound}

def get_optimizer(cfg, phase='searching', optimizer_type='optimizer_model'):

    if cfg[phase][optimizer_type] is None:
        logger.info("Using SGD optimizer")
        return SGD
    else:
        opt_name = cfg[phase][optimizer_type]['name']
        if opt_name not in key2opt:
            raise NotImplementedError('Optimizer {} not implemented'.format(opt_name))

        logger.info('Using {} optimizer'.format(opt_name))
        return key2opt[opt_name]
