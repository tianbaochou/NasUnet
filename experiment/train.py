import os
import sys
import yaml
import time
import shutil
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn

sys.path.append('..')
from util.loss.loss import SegmentationLosses
from util.datasets import get_dataset
from util.utils import get_logger, save_checkpoint, calc_time, store_images
from util.utils import  average_meter, weights_init
from util.utils import get_gpus_memory_info, calc_parameters_count
from util.schedulers import get_scheduler
from util.optimizers import get_optimizer
from util.challenge.promise12.store_test_seg import predict_test
from util.metrics import *
from models import get_segmentation_model
import models.geno_searched as geno_types
from tensorboardX import SummaryWriter

class Network(object):

    def __init__(self):
        self._init_configure()
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()
        self._check_resume()

    def _init_configure(self):
        parser = argparse.ArgumentParser(description='config')

        # Add default argument
        parser.add_argument('--config',nargs='?',type=str,default='../configs/nas_unet/nas_unet_chaos.yml',
                            help='Configuration file to use')
        parser.add_argument('--model',nargs='?',type=str,default='nasunet',
                            help='Model to train and evaluation')
        parser.add_argument('--ft', action='store_true', default= False,
                            help='finetuning on a different dataset')
        parser.add_argument('--warm',nargs='?',type=int,default=0,
                            help='warm up from pre epoch')

        self.args = parser.parse_args()

        with open(self.args.config) as fp:
            self.cfg = yaml.load(fp)
            print('load configure file at {}'.format(self.args.config))
        self.model_name = self.args.model
        print('Usage model :{}'.format(self.model_name))

    def _init_logger(self):
        log_dir = '../logs/'+ self.model_name + '/train' + '/{}'.format(self.cfg['data']['dataset']) \
                  +'/{}'.format(time.strftime('%Y%m%d-%H%M%S'))
        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        self.logger.info('{}-Train'.format(self.model_name))
        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.save_image_path = os.path.join(self.save_path, 'saved_val_images')
        self.writer = SummaryWriter(self.save_tbx_log)
        shutil.copy(self.args.config, self.save_path)

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
        trainset = get_dataset(self.cfg['data']['dataset'], split='train', mode='train')
        valset = get_dataset(self.cfg['data']['dataset'], split='val', mode ='val')
        # testset = get_dataset(self.cfg['data']['dataset'], split='test', mode='test')
        self.nweight = trainset.class_weight
        print('dataset weights: {}'.format(self.nweight))
        self.n_classes = trainset.num_class
        self.batch_size = self.cfg['training']['batch_size']
        kwargs = {'num_workers': self.cfg['training']['n_workers'], 'pin_memory': True}

        # Split val dataset
        if self.cfg['data']['dataset'] in ['bladder', 'chaos', 'ultrasound_nerve']:
            num_train = len(trainset)
            indices = list(range(num_train))
            split = int(np.floor(0.8 * num_train))
            self.logger.info('split training data : 0.8')
            self.train_queue = data.DataLoader(trainset, batch_size=self.batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                               **kwargs)

            self.valid_queue = data.DataLoader(trainset, batch_size=self.batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                   indices[split:num_train]),
                                               **kwargs)
        else:
            self.train_queue = data.DataLoader(trainset, batch_size=self.batch_size,
                                               drop_last=True, shuffle=True, **kwargs)

            self.valid_queue = data.DataLoader(valset, batch_size=self.batch_size,
                                               drop_last=False, shuffle=False, **kwargs)
    def _init_model(self):

        # Setup loss function
        criterion = SegmentationLosses(name=self.cfg['training']['loss']['name'],
                                       aux_weight = self.cfg['training']['loss']['aux_weight'],
                                       weight = self.nweight,
                                       ignore_index=-1 # ignore background
                                       )
        self.criterion = criterion.to(self.device)

        self.show_dice_coeff = False
        if self.cfg['data']['dataset'] in ['bladder', 'chaos', 'ultrasound_nerve', 'promise12']:
            self.show_dice_coeff = True

        self.logger.info("Using loss {}".format(self.criterion))

        # Setup Model
        try:
            genotype = eval('geno_types.%s' % self.cfg['training']['geno_type'])
            init_channels = self.cfg['training']['init_channels']
            depth = self.cfg['training']['depth']
        except:
            genotype = None
            init_channels = 0
            depth = 0
        # aux_weight > 0 and the loss is cross_entropy, we will use FCN header for auxiliary layer. and the aux set to True
        # aux_weight > 0 and the loss is cross_entropy_with_dice, we will combine cross entropy loss with dice loss
        self.aux = True if self.cfg['training']['loss']['aux_weight'] > 0  \
                    and self.cfg['training']['loss']['name'] != 'cross_entropy_with_dice' else False
        model = get_segmentation_model(self.model_name,
                                       dataset = self.cfg['data']['dataset'],
                                       backbone=self.cfg['training']['backbone'],
                                       aux = self.aux,
                                       c = init_channels,
                                       depth = depth,
                                       # the below two are special for nasunet
                                       genotype=genotype,
                                       double_down_channel=self.cfg['training']['double_down_channel']
                                       )

        # init weight using hekming methods
        model.apply(weights_init)
        self.logger.info('Initialize the model weights: kaiming_uniform')

        if torch.cuda.device_count() > 1 and self.cfg['training']['multi_gpus']:
            self.logger.info('use: %d gpus', torch.cuda.device_count())
            model = nn.DataParallel(model)
        else:
            self.logger.info('gpu device = %d' % self.device_id)
            torch.cuda.set_device(self.device_id)
        self.model = model.to(self.device)
        self.logger.info('param size = %fMB', calc_parameters_count(model))

        # Setup optimizer, lr_scheduler for model
        optimizer_cls = get_optimizer(self.cfg, phase='training', optimizer_type='model_optimizer')
        optimizer_params = {k: v for k, v in self.cfg['training']['model_optimizer'].items()
                            if k != 'name'}

        self.model_optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        self.logger.info("Using model optimizer {}".format(self.model_optimizer))


    def _check_resume(self):
        self.dur_time = 0
        self.start_epoch = 0
        self.best_mIoU, self.best_loss, self.best_pixAcc, self.best_dice_coeff = 0, 1.0, 0, 0
        # optionally resume from a checkpoint for model
        resume = self.cfg['training']['resume'] if self.cfg['training']['resume'] is not None else None
        if resume is not None:
            if os.path.isfile(resume):
                self.logger.info("Loading model and optimizer from checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume, map_location=self.device)
                if not self.args.ft: # no fine-tuning
                    self.start_epoch = checkpoint['epoch']
                    self.dur_time = checkpoint['dur_time']
                    self.best_mIoU = checkpoint[ 'best_mIoU']
                    self.best_pixAcc = checkpoint[ 'best_pixAcc']
                    self.best_loss = checkpoint['best_loss']
                    self.best_dice_coeff = checkpoint['best_dice_coeff']
                    self.model_optimizer.load_state_dict(checkpoint['model_optimizer'])
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.logger.info("No checkpoint found at '{}'".format(resume))


        # init LR_scheduler
        scheduler_params = {k: v for k, v in self.cfg['training']['lr_schedule'].items()}
        if 'max_iter' in self.cfg['training']['lr_schedule']:
            scheduler_params['max_iter'] = self.cfg['training']['epoch']
            # Note: For step in train epoch !!!!  must use the value below
            # scheduler_params['max_iter'] = len(self.train_queue) * self.cfg['training']['epoch'] \
            #                                // self.cfg['training']['batch_size']
        if 'T_max' in self.cfg['training']['lr_schedule']:
            scheduler_params['T_max'] = self.cfg['training']['epoch']
            # Note: For step in train epoch !!!!  must use the value below
            # scheduler_params['T_max'] = len(self.train_queue) * self.cfg['training']['epoch'] \
            #                                // self.cfg['training']['batch_size']

        scheduler_params['last_epoch'] = -1 if self.start_epoch == 0 else self.start_epoch
        self.scheduler = get_scheduler(self.model_optimizer, scheduler_params)

    def run(self):
        self.logger.info('args = %s', self.cfg)
        # Setup Metrics
        self.metric_train = SegmentationMetric(self.n_classes)
        self.metric_val = SegmentationMetric(self.n_classes)
        self.metric_test = SegmentationMetric(self.n_classes)
        self.val_loss_meter = average_meter()
        self.test_loss_meter = average_meter()
        self.train_loss_meter = average_meter()
        self.train_dice_coeff_meter = average_meter()
        self.val_dice_coeff_meter = average_meter()
        self.patience = 0
        self.save_best = True
        run_start = time.time()

        # Set up results folder
        if not os.path.exists(self.save_image_path):
            os.makedirs(self.save_image_path)

        for epoch in range(self.start_epoch, self.cfg['training']['epoch']):
            self.epoch = epoch

            self.scheduler.step()

            self.logger.info('=> Epoch {}, lr {}'.format(self.epoch, self.scheduler.get_lr()[-1]))

            # train and search the model
            self.train()

            # valid the model
            self.val()

            self.logger.info('current best loss {}, pixAcc {}, mIoU {}'.format(
                self.best_loss, self.best_pixAcc, self.best_mIoU,
            ))

            if  self.show_dice_coeff:
                self.logger.info('current best DSC {}'.format(self.best_dice_coeff))

            if self.save_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'dur_time': self.dur_time + time.time() - run_start,
                    'model_state': self.model.state_dict(),
                    'model_optimizer': self.model_optimizer.state_dict(),
                    'best_pixAcc': self.best_pixAcc,
                    'best_mIoU': self.best_mIoU,
                    'best_dice_coeff': self.best_dice_coeff,
                    'best_loss': self.best_loss,
                }, True, self.save_path)
                self.logger.info('save checkpoint (epoch %d) in %s  dur_time: %s',
                        epoch, self.save_path, calc_time(self.dur_time + time.time() - run_start))
                self.save_best = False

            if self.patience == self.cfg['training']['max_patience'] or epoch == self.cfg['training']['epoch']-1:
                # load best model weights
                # self._check_resume(os.path.join(self.save_path, 'checkpint.pth.tar'))
                # # Test
                # if len(self.test_queue) > 0:
                #     self.logger.info('Training ends \n Test')
                #     self.test()
                # else:
                #     self.logger.info('Training ends!')
                print('Early stopping')
                break
            else:
                self.logger.info('current patience :{}'.format(self.patience))

            self.val_loss_meter.reset()
            self.train_loss_meter.reset()
            self.train_dice_coeff_meter.reset()
            self.val_dice_coeff_meter.reset()
            self.metric_train.reset()
            self.metric_val.reset()
            self.logger.info('cost time: {}'.format(calc_time(self.dur_time + time.time() - run_start)))

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.save_tbx_log + "/all_scalars.json")
        self.writer.close()
        self.logger.info('cost time: {}'.format(calc_time(self.dur_time + time.time() - run_start)))
        self.logger.info('log dir in : {}'.format(self.save_path))


    def train(self):
        self.model.train()
        tbar = tqdm(self.train_queue)
        for step, (input, target) in enumerate(tbar):

            self.model_optimizer.zero_grad()

            input = input.cuda(self.device)
            target = target.cuda(self.device)

            # Note: if aux is True, predicts will be a list predicts = [pred, aux_pred], otherwise [pred]
            # so predicts[0] is original pred
            predicts = self.model(input)

            train_loss = self.criterion(predicts if self.aux else predicts[0], target)

            self.train_loss_meter.update(train_loss.item())

            self.metric_train.update(target, predicts[0])

            train_loss.backward()

            if self.show_dice_coeff:
                with torch.no_grad():
                    dice_coeff = dice_coefficient(predicts[0], target)
                self.train_dice_coeff_meter.update(dice_coeff)

            if self.cfg['training']['grad_clip']:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.cfg['training']['grad_clip'])

            if step % self.cfg['training']['report_freq'] == 0:
                if self.show_dice_coeff:
                    mdice_coeff = self.train_dice_coeff_meter.mloss
                    self.logger.info('dice coeff: {}'.format(mdice_coeff))

                self.logger.info('train loss %03d %e | epoch [%d] / [%d]', step,
                                 self.train_loss_meter.mloss, self.epoch, self.cfg['training']['epoch'])
                pixAcc, mIoU = self.metric_train.get()
                self.logger.info('pixAcc: {},  mIoU: {}'.format(pixAcc, mIoU))
                tbar.set_description('train loss: %.6f; pixAcc: %.3f; mIoU %.6f' % (self.train_loss_meter.mloss, pixAcc, mIoU))

            # Update the network parameters
            self.model_optimizer.step()

        # save in tensorboard scalars
        self.writer.add_scalar('Train/loss', self.train_loss_meter.mloss, self.epoch)

    def val(self):
        self.model.eval()
        tbar = tqdm(self.valid_queue)
        with torch.no_grad():
            for step, (input, target) in enumerate(tbar):
                input = input.cuda(self.device)
                target = target.cuda(self.device)
                predicts = self.model(input)

                val_loss = self.criterion(predicts if self.aux else predicts[0], target)

                self.val_loss_meter.update(val_loss.item())

                self.metric_val.update(target, predicts[0])

                # calc dice coeff
                if self.show_dice_coeff:
                    dice_coeff = dice_coefficient(predicts[0], target)
                    self.val_dice_coeff_meter.update(dice_coeff)

                if step % self.cfg['training']['report_freq'] == 0:
                    pixAcc, mIoU = self.metric_val.get()

                    if self.show_dice_coeff:
                        mdice_coeff = self.val_dice_coeff_meter.mloss
                        self.logger.info('dice coeff: {}'.format(mdice_coeff))

                    self.logger.info('val loss: {}, pixAcc: {}, mIoU: {}'.format(
                        self.val_loss_meter.mloss, pixAcc, mIoU))
                    tbar.set_description('val loss: %.6f, pixAcc: %.3f, mIoU: %.6f'
                                         % (self.val_loss_meter.mloss, pixAcc, mIoU))

        # save images
        # cause the predicts is a list [pred, aux_pred(may not)]
        if len(predicts[0].shape) == 4: #
            pred = predicts[0]
        else:
            pred = predicts

        grid_image = store_images(input, pred, target)
        self.writer.add_image('Val', grid_image, self.epoch)

        # save in tensorboard scalars
        pixAcc, mIoU = self.metric_val.get()
        cur_loss = self.val_loss_meter.mloss
        self.writer.add_scalar('Val/Acc', pixAcc, self.epoch)
        self.writer.add_scalar('Val/mIoU', mIoU, self.epoch)
        self.writer.add_scalar('Val/loss', self.val_loss_meter.mloss, self.epoch)
        if self.show_dice_coeff:
            mdice_coeff = self.val_dice_coeff_meter.mloss
            self.writer.add_scalar('Val/dice_coeff', mdice_coeff, self.epoch)

        # for early-stopping
        if self.best_loss > cur_loss or self.best_mIoU < mIoU:
            self.patience = 0
        else:
            self.patience += 1

        # Store best score
        self.best_pixAcc = pixAcc if self.best_pixAcc < pixAcc else self.best_pixAcc
        self.best_loss = cur_loss if self.best_loss > cur_loss else self.best_loss

        if self.show_dice_coeff: # DSC first
            if self.best_dice_coeff < mdice_coeff:
                self.best_dice_coeff = mdice_coeff
                self.best_mIoU =  mIoU if self.best_mIoU < mIoU else self.best_mIoU
                self.save_best = True
        elif self.best_mIoU < mIoU: # mIoU is the major metric if no use dice loss
            self.best_mIoU = mIoU
            self.save_best = True

    def test(self):
        self.model.eval()
        predict_list = []
        tbar = tqdm(self.test_queue)
        with torch.no_grad():
            for step, (input, target) in enumerate(tbar):
                input = input.cuda(self.device)
                if not isinstance(target, list) and not isinstance(target, str):
                    target = target.cuda(self.device)
                elif isinstance(target, str):
                    target = target.split('.')[0] + '_mask.tiff'

                predicts = self.model(input)

                # for cityscapes, voc, camvid
                if not isinstance(target, list):
                    test_loss = self.criterion(predicts if self.aux else predicts[0], target)
                    self.test_loss_meter.update(test_loss.item())
                    self.metric_test.update(target, predicts[0])
                else: # for promise12
                    N = predicts[0].shape[0]
                    for i in range(N):
                        predict_list += [torch.argmax(predicts[0], 1).cpu().numpy()[i]]

        # cause the predicts is a list [pred, aux_pred(may not)]
        if len(predicts[0].shape) == 4: #
            pred = predicts[0]
        else:
            pred = predicts

        # save images
        if not isinstance(target, list):
            grid_image = store_images(input, pred, target)
            self.writer.add_image('Test', grid_image, self.epoch)
            pixAcc, mIoU = self.metric_test.get()
            self.logger.info('Test/loss: {}, pixAcc: {}, mIoU: {}'.format(
                self.test_loss_meter.mloss, pixAcc, mIoU))
        else:
            predict_test(predict_list, target, self.save_path+'/test_rst')

if __name__ == '__main__':
    train_network = Network()
    train_network.run()
