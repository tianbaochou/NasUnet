import os
import sys
import yaml
import time
import argparse

from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.utils import data
import torch.backends.cudnn as cudnn

sys.path.append('..')
from util.loss.loss import SegmentationLosses
from util.datasets import get_dataset
from util.utils import get_logger, average_meter, store_images, resize_pred_to_val
from util.utils import get_gpus_memory_info, calc_parameters_count, create_exp_dir
from util.challenge.promise12.store_test_seg import predict_test
from models import get_segmentation_model
from util.challenge.promise12.metrics import biomedical_image_metric, numpy_dice
from util.metrics import *
from PIL import Image
from models import geno_searched
from util.crf import dense_crf

class TestNetwork(object):
    def __init__(self):
        self._init_configure()
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()
        if not self._check_resume():
            self.logger.error('The pre-trained model not exist!!!')
            exit(-1)

    def _init_configure(self):
        parser = argparse.ArgumentParser(description='config')

        # Add default argument
        parser.add_argument('--config',nargs='?',type=str,default='../configs/nas_unet/nas_unet_nerve.yml',
                            help='Configuration file to use')
        parser.add_argument('--model',nargs='?',type=str,default='nasunet',
                            help='Model to test')
        parser.add_argument('--crf',action='store_true', default= False,
                            help='Model to test')


        self.args = parser.parse_args()

        with open(self.args.config) as fp:
            self.cfg = yaml.load(fp)
            print('load configure file at {}'.format(self.args.config))
        self.model_name = self.args.model
        print('Usage model :{}'.format(self.model_name))

    def _init_logger(self):
        log_dir = '../logs/'+ self.model_name + '/test' + '/{}'.format(self.cfg['data']['dataset']) \
                  +'/{}'.format(time.strftime('%Y%m%d-%H%M'))
        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        self.logger.info('{}-Train'.format(self.model_name))
        self.save_path = log_dir
        self.save_image_path = os.path.join(self.save_path, 'saved_val_images')

    def _init_device(self):
        if not torch.cuda.is_available():
            self.logger.info('no gpu device available')
            sys.exit(1)

        np.random.seed(self.cfg.get('seed', 1337))
        torch.manual_seed(self.cfg.get('seed', 1337))
        torch.cuda.manual_seed(self.cfg.get('seed', 1337))
        cudnn.enabled = True
        cudnn.benchmark = True
        self.device_id, self.gpus_info = get_gpus_memory_info()
        self.device = torch.device('cuda:{}'.format(0 if self.cfg['training']['multi_gpus'] else self.device_id))

    def _init_dataset(self):
        self.trainset = get_dataset(self.cfg['data']['dataset'], split='train', mode='train')
        self.valset = get_dataset(self.cfg['data']['dataset'], split='val', mode ='val')
        self.testset = get_dataset(self.cfg['data']['dataset'], split='test', mode='test')
        self.nweight = self.trainset.class_weight
        self.n_classes = self.trainset.num_class
        self.batch_size = self.cfg['training']['batch_size']
        kwargs = {'num_workers': self.cfg['training']['n_workers'], 'pin_memory': True}

        # Original data no split val dataset
        if self.cfg['data']['dataset'] in ['bladder', 'chaos', 'ultrasound_nerve']:
            num_train = len(self.trainset)
            indices = list(range(num_train))
            split = int(np.floor(0.8 * num_train))

            self.valid_queue = data.DataLoader(self.trainset, batch_size=self.batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                   indices[split:num_train]),
                                               **kwargs)
            self.test_queue = data.DataLoader(self.testset, batch_size=self.batch_size,
                                               drop_last=False, shuffle=False, **kwargs)
        else:

            self.valid_queue = data.DataLoader(self.valset, batch_size=self.batch_size,
                                               drop_last=False, shuffle=False, **kwargs)

            self.test_queue = data.DataLoader(self.testset, batch_size=self.batch_size,
                                               drop_last=False, shuffle=False, **kwargs)

    def _init_model(self):
        # Setup loss function
        criterion = SegmentationLosses(name=self.cfg['training']['loss']['name'],
                                       aux_weight=self.cfg['training']['loss']['aux_weight'],
                                       weight=self.nweight
                                       )
        self.criterion = criterion.to(self.device)
        self.logger.info("Using loss {}".format(self.cfg['training']['loss']['name']))

        # Setup Model
        try:
            genotype = eval('geno_types.%s' % self.cfg['training']['geno_type'])
            init_channels = self.cfg['training']['init_channels']
            depth = self.cfg['training']['depth']
        except:
            genotype = None
            init_channels = 0
            depth = 0
        model = get_segmentation_model(self.model_name,
                                       dataset = self.cfg['data']['dataset'],
                                       backbone=self.cfg['training']['backbone'],
                                       aux = False,
                                       c = init_channels,
                                       depth = depth,
                                       # the below two are special for nasunet
                                       genotype=genotype,
                                       double_down_channel=self.cfg['training']['double_down_channel']
                                       )

        if torch.cuda.device_count() > 1 and self.cfg['training']['multi_gpus']:
            self.logger.info('use: %d gpus', torch.cuda.device_count())
            model = nn.DataParallel(model)
        else:
            self.logger.info('gpu device = %d' % self.device_id)
            torch.cuda.set_device(self.device_id)
        self.model = model.to(self.device)
        self.logger.info('param size = %fMB', calc_parameters_count(model))

    def _check_resume(self):
        resume = self.cfg['training']['resume'] if self.cfg['training']['resume'] is not None else None
        if resume is not None:
            if os.path.isfile(resume):
                self.logger.info("Loading model and optimizer from checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state'])
                return True
            else:
                self.logger.info("No checkpoint found at '{}'".format(resume))
                return False
        return False

    def test(self, img_queue, split='val', desc=''):
        self.model.eval()
        predict_list = []
        accuracy = 0
        tbar = tqdm(img_queue)
        create_exp_dir(desc, desc='=>Save prediction image on')
        with torch.no_grad():
            for step, (input, target) in enumerate(tbar):
                input = input.cuda(self.device)
                if not isinstance(target, list):
                    target = target.cuda(self.device)

                predicts = self.model(input)

                # for cityscapes, voc, camvid, test have label
                if not isinstance(target, list):
                    test_loss = self.criterion(predicts[0], target)
                    self.loss_meter.update(test_loss.item())
                    self.metric.update(target, predicts[0])
                    if step % self.cfg['training']['report_freq'] == 0:
                        pixAcc, mIoU = self.metric.get()
                        self.logger.info('{} loss: {}, pixAcc: {}, mIoU: {}'.format(
                            split, self.loss_meter.mloss, pixAcc, mIoU))
                        tbar.set_description('loss: %.6f, pixAcc: %.3f, mIoU: %.6f'
                                             % (self.loss_meter.mloss, pixAcc, mIoU))
                    accuracy += dice_coefficient(predicts[0].cpu(), target.cpu())
                else:
                    N =  predicts[0].shape[0]
                    for i in range(N):
                        if self.args.crf: # use crf
                            predict = torch.argmax(predicts[0].cpu(), 1)[i]
                            predict = dense_crf(np.array(input[i].cpu()).astype(np.uint8), predict) > 0.5
                            img = Image.fromarray((predict * 255).astype(np.uint8))
                            file_name = os.path.split(target[i])[1]
                            file_name = file_name.split('.')[0]+'_mask.tif'
                            img.save(os.path.join(desc,file_name))
                        else:
                            img = Image.fromarray((torch.argmax(predicts[0].cpu(),1)[i] * 255).numpy().astype(np.uint8))
                            file_name = os.path.split(target[i])[1]
                            file_name = file_name.split('.')[0]+'_mask.tif'
                            img.save(os.path.join(desc,file_name))

                if desc=='promise12': # for promise12, test have not label or have label to calc extra metric
                    predict_list += [torch.argmax(predicts[0], dim=1).cpu().numpy()]

        print('==> accuracy: {}'.format(accuracy/len(img_queue)))

        # cause the predicts is a list [pred, aux_pred(may not)]
        if len(predicts[0].shape) == 4:  #
            pred = predicts[0]
        else:
            pred = predicts

        # save images
        if not isinstance(target, list) and not isinstance(target, str): #
            grid_image = store_images(input, pred, target)
            pixAcc, mIoU = self.metric.get()
            self.logger.info('{}/loss: {}, pixAcc: {}, mIoU: {}'.format(
                split, self.loss_meter.mloss, pixAcc, mIoU))
        elif desc == 'promise12':  # for promise12, test have not label
            predict_test(predict_list, target, self.save_path + '/{}_rst'.format(split))

        # for promise12 metirc
        if desc == 'promise12' and split=='val':
            val_list = [5, 15, 25, 35, 45]
            dir = os.path.join(self.trainset.root_dir, self.trainset.base_dir, 'TrainingData')
            biomedical_image_metric(predict_list, val_list, dir+'/')

    def run(self):
        self.logger.info('args = %s', self.cfg)
        # Setup Metrics
        self.metric = SegmentationMetric(self.n_classes)
        self.loss_meter = average_meter()
        run_start = time.time()
        # Set up results folder
        if not os.path.exists(self.save_image_path):
            os.makedirs(self.save_image_path)

        # if len(self.valid_queue) != 0:
        #     self.logger.info('Begin valid set evaluation')
        #     self.test(self.valid_queue, split='val', desc='promise12')
        if len(self.test_queue) != 0:
            self.logger.info('Begin test set evaluation')
            self.test(self.test_queue, split='test', desc='../predictions/nerve_rst')
        self.logger.info('Evaluation done!')


if __name__  == '__main__':
    testNetwork = TestNetwork()
    testNetwork.run()





