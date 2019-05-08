import os
import sys
import cv2
import torch
import subprocess
import numpy as np
import SimpleITK as sitk
from os import listdir
from os.path import isfile, join, splitext
from skimage.exposure import equalize_adapthist
from .base import  BaseDataset
from util.utils import create_exp_dir
from util.augmentations import smooth_images
from util.augmentations import *

class DataManager(object):
    # dataset isotropically scaled to 1x1x1.5mm, volume resized to 128x128x64 (64 is the number of slices)
    # The datasets were first normalised using the N4 bias filed correction function of the ANTs framework
    # params=None
    # srcFolder=None
    # resultsDir=None
    #
    # fileList=None
    # gtList=None
    #
    # sitkImages=None
    # sitkGT=None
    # meanIntensityTrain = None

    def __init__(self,imageFolder, GTFolder, resultsDir, parameters):
        self.params = parameters
        self.imageFolder = imageFolder
        self.GTFolder = GTFolder
        self.resultsDir = resultsDir

    def createImageFileList(self):
        self.imageFileList = [f for f in listdir(self.imageFolder) if isfile(join(self.imageFolder, f)) and '_seg' not in f and '.raw' not in f]
        print('imageFileList: ' + str(self.imageFileList))

    def createGTFileList(self):
        self.GTFileList = [f for f in listdir(self.GTFolder) if isfile(
            join(self.GTFolder, f)) and '_seg' in f and '.raw' not in f]
        print('GTFileList: ' + str(self.GTFileList))

    def loadImages(self):
        self.sitkImages=dict()
        rescalFilt=sitk.RescaleIntensityImageFilter()
        rescalFilt.SetOutputMaximum(1)
        rescalFilt.SetOutputMinimum(0)

        stats = sitk.StatisticsImageFilter()
        m = 0.

        for f in self.imageFileList:
            id = f.split('.')[0]
            self.sitkImages[id]=rescalFilt.Execute(sitk.Cast(sitk.ReadImage(join(self.imageFolder, f)),sitk.sitkFloat32))
            stats.Execute(self.sitkImages[id])
            m += stats.GetMean()

        self.meanIntensityTrain=m/len(self.sitkImages)


    def loadGT(self):
        self.sitkGT=dict()

        for f in self.GTFileList:
            id = f.split('.')[0]
            self.sitkGT[id]=sitk.Cast(sitk.ReadImage(join(self.GTFolder, f))>0.5,sitk.sitkFloat32)

    def loadTrainingData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()

    def loadTestingData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()

    def loadInferData(self):
        self.createImageFileList()
        self.loadImages()

    def getNumpyImages(self):
        dat = self.getNumpyData(self.sitkImages,sitk.sitkLinear)

        for key in dat.keys(): # https://github.com/faustomilletari/VNet/blob/master/VNet.py, line 147. For standardization?
            mean = np.mean(dat[key][dat[key]>0]) # why restrict to >0? By Chao.
            std = np.std(dat[key][dat[key]>0])

            dat[key] -= mean
            dat[key] /=std

        return dat


    def getNumpyGT(self):
        dat = self.getNumpyData(self.sitkGT,sitk.sitkLinear)

        for key in dat:
            dat[key] = (dat[key]>0.5).astype(dtype=np.float32)

        return dat


    def getNumpyData(self,dat, method):
        ret=dict()
        for key in dat:
            ret[key] = np.zeros([self.params['VolSize'][0], self.params['VolSize'][1], self.params['VolSize'][2]], dtype=np.float32)

            img=dat[key]

            # we rotate the image according to its transformation using the direction and according to the final spacing we want
            factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                     self.params['dstRes'][2]]

            factorSize = np.asarray(img.GetSize() * factor, dtype=float)

            newSize = np.max([factorSize, self.params['VolSize']], axis=0)

            newSize = newSize.astype(dtype='int')

            T=sitk.AffineTransform(3)
            T.SetMatrix(img.GetDirection())

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
            resampler.SetSize(newSize.tolist())
            resampler.SetInterpolator(method)
            if self.params['normDir']:
                resampler.SetTransform(T.GetInverse())

            imgResampled = resampler.Execute(img)


            imgCentroid = np.asarray(newSize, dtype=float) / 2.0

            imgStartPx = (imgCentroid - self.params['VolSize'] / 2.0).astype(dtype='int')

            regionExtractor = sitk.RegionOfInterestImageFilter()
            size_2_set = self.params['VolSize'].astype(dtype='int')
            regionExtractor.SetSize(size_2_set.tolist())
            regionExtractor.SetIndex(imgStartPx.tolist())

            imgResampledCropped = regionExtractor.Execute(imgResampled)

            ret[key] = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [2, 1, 0])

        return ret


    def writeResultsFromNumpyLabel(self,result,key, resultTag, ext, resultDir):
        '''
        :param result: predicted mask
        :param key: sample id
        :return: register predicted mask (e.g. binary mask of size 96x96x48) to original image (e.g. CT volume of size 320x320x20), output the final mask of the same size as original image.
        '''
        img = self.sitkImages[key] # original image
        print("original img shape{}".format(img.GetSize()))

        toWrite = sitk.Image(img.GetSize()[0],img.GetSize()[1],img.GetSize()[2],sitk.sitkFloat32)

        factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                     self.params['dstRes'][2]]

        factorSize = np.asarray(img.GetSize() * factor, dtype=float)

        newSize = np.max([factorSize, self.params['VolSize']], axis=0)

        newSize = newSize.astype(dtype=int)

        T = sitk.AffineTransform(3)
        T.SetMatrix(img.GetDirection())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
        resampler.SetSize(newSize.tolist())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        if self.params['normDir']:
            resampler.SetTransform(T.GetInverse())

        toWrite = resampler.Execute(toWrite)

        imgCentroid = np.asarray(newSize, dtype=float) / 2.0

        imgStartPx = (imgCentroid - self.params['VolSize'] / 2.0).astype(dtype=int)

        for dstX, srcX in zip(range(0, result.shape[0]), range(imgStartPx[0],int(imgStartPx[0]+self.params['VolSize'][0]))):
            for dstY, srcY in zip(range(0, result.shape[1]), range(imgStartPx[1], int(imgStartPx[1]+self.params['VolSize'][1]))):
                for dstZ, srcZ in zip(range(0, result.shape[2]), range(imgStartPx[2], int(imgStartPx[2]+self.params['VolSize'][2]))):
                    try:
                        toWrite.SetPixel(int(srcX),int(srcY),int(srcZ),float(result[dstX,dstY,dstZ]))
                    except:
                        pass

        resampler.SetOutputSpacing([img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]])
        resampler.SetSize(img.GetSize())

        if self.params['normDir']:
            resampler.SetTransform(T)

        toWrite = resampler.Execute(toWrite)

        thfilter=sitk.BinaryThresholdImageFilter()
        thfilter.SetInsideValue(1)
        thfilter.SetOutsideValue(0)
        thfilter.SetLowerThreshold(0.5)
        toWrite = thfilter.Execute(toWrite)

        #connected component analysis (better safe than sorry)

        cc = sitk.ConnectedComponentImageFilter()
        toWritecc = cc.Execute(sitk.Cast(toWrite,sitk.sitkUInt8))

        arrCC=np.transpose(sitk.GetArrayFromImage(toWritecc).astype(dtype=float), [2, 1, 0])

        lab=np.zeros(int(np.max(arrCC)+1),dtype=float)

        for i in range(1,int(np.max(arrCC)+1)):
            lab[i]=np.sum(arrCC==i)

        activeLab=np.argmax(lab)

        toWrite = (toWritecc==activeLab)

        toWrite = sitk.Cast(toWrite,sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()

        writer.SetFileName(join(resultDir, key + resultTag + ext))
        writer.Execute(toWrite)


def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)

        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def data_to_array(base_path, store_path, img_rows, img_cols):

    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(int(img_rows/8),int(img_cols/8)))

    fileList =  os.listdir(os.path.join(base_path, 'TrainingData'))

    fileList = sorted((x for x in fileList if '.mhd' in x))

    val_list = [5, 15, 25, 35, 45]
    train_list = list(set(range(50)) - set(val_list) )
    count = 0
    for the_list in [train_list,  val_list]:
        images = []
        masks = []

        filtered = [file for file in fileList for ff in the_list if str(ff).zfill(2) in file ]

        for filename in filtered:

            itkimage = sitk.ReadImage(os.path.join(base_path, 'TrainingData', filename))
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append( imgs )
            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
                images.append(imgs)

        # images: slices x w x h ==> total number x w x h
        images = np.concatenate(images , axis=0 ).reshape(-1, img_rows, img_cols)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols)
        masks = masks.astype(np.uint8)

        # Smooth images using CurvatureFlow
        images = smooth_images(images)
        images = images.astype(np.float32)

        if count==0: # no normalize
            mu = np.mean(images)
            sigma = np.std(images)
            images = (images - mu)/sigma

            np.save(os.path.join(store_path, 'X_train.npy'), images)
            np.save(os.path.join(store_path,'y_train.npy'), masks)
        elif count==1:
            images = (images - mu)/sigma
            np.save(os.path.join(store_path, 'X_val.npy'), images)
            np.save(os.path.join(store_path,'y_val.npy'), masks)
        count+=1

    fileList =  os.listdir(os.path.join(base_path, 'TestData'))
    fileList = sorted([x for x in fileList if '.mhd' in x])
    n_imgs=[]
    images=[]
    for filename in fileList:
        itkimage = sitk.ReadImage(os.path.join(base_path, 'TestData', filename))
        imgs = sitk.GetArrayFromImage(itkimage)
        imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
        images.append(imgs)
        n_imgs.append(len(imgs))

    images = np.concatenate(images , axis=0).reshape(-1, img_rows, img_cols)
    images = smooth_images(images)
    images = images.astype(np.float32)
    images = (images - mu)/sigma

    np.save(os.path.join(store_path,'X_test.npy'), images)
    np.save(os.path.join(store_path, 'test_n_imgs.npy'), np.array(n_imgs))
    print('save file in {}'.format(store_path))

def load_train_data(store_path):

    X_train = np.load(os.path.join(store_path, 'X_train.npy'))
    y_train = np.load(os.path.join(store_path, 'y_train.npy'))

    return X_train, y_train

def load_val_data(store_path):

    X_val = np.load(os.path.join(store_path, 'X_val.npy'))
    y_val = np.load(os.path.join(store_path, 'y_val.npy'))
    return X_val, y_val

def load_test_data(store_path):
    X_test = np.load(os.path.join(store_path, 'X_test.npy'))
    x_slice_array = np.load(os.path.join(store_path, 'y_val.npy'))
    return X_test, x_slice_array

def get_test_list(base_path):
    fileList = os.listdir(os.path.join(base_path, 'TestData'))
    fileList = sorted([os.path.join(base_path, 'TestData',x) for x in fileList if '.mhd' in x])
    return fileList


class Promise12(BaseDataset):
    IN_CHANNELS = 1
    BASE_DIR = 'PROMISE2012'
    TRAIN_IMAGE_DIR = 'TrainingData'
    VAL_IMAGE_DIR = 'TestData'
    NUM_CLASS = 2
    CROP_SIZE = 256
    CLASS_WEIGHTS = None

    def __init__(self, root, split='train', mode=None):
        super(Promise12, self).__init__(root, split=split, mode=mode)
        self.mode = mode
        #self.joint_transform = joint_transform
        root = root + '/' + self.BASE_DIR
        self.joint_transform = Compose([
            RandomTranslate(offset=(0.2, 0.1)),
            RandomVerticallyFlip(),
            RandomHorizontallyFlip(),
            RandomElasticTransform(alpha = 1.5, sigma = 0.07, img_type='F'),
            ])

        self.img_normalize = None

        # SECOND
        # store data in the npy file
        data_path = os.path.join(root, 'npy_image')
        if not os.path.exists(data_path):
            create_exp_dir(data_path, 'Create augmentation data at {}')
            data_to_array(root, data_path, self.CROP_SIZE, self.CROP_SIZE)
        else:
            print('read the data from: {}'.format(data_path))

        self.test_file_list = get_test_list(root)

        # read the data from npy
        if mode == 'train':
            self.X_train, self.y_train = load_train_data(data_path)
            self.size = self.X_train.shape[0]
        elif mode == 'val':
            self.X_val, self.y_val = load_val_data(data_path)
            self.size = self.X_val.shape[0]
        elif mode == 'test':
            self.X_test, self.x_slice_array = load_test_data(data_path)
            self.size = self.X_test.shape[0]

    def __getitem__(self, index):
        # 1. the image already crop
        if self.mode == "train":
            img, target = self.X_train[index], self.y_train[index]
        elif self.mode == 'val':
            img, target = self.X_val[index], self.y_val[index]
        elif self.mode == 'test': # the test target indicate the number of slice for each case
            img, target = self.X_test[index], self.test_file_list
        img = Image.fromarray(img, mode='F')

        if self.mode != 'test':
            target = Image.fromarray(target, mode='L')
            # 2. do joint transform
            if self.joint_transform is not None:
                img, target = self.joint_transform(img, target)
                # 3. to tensor
                img, target = self.to_tensor(img, target)
        else:
            # 3. img to tensor
            img = self.img2tensor(img)

        # 4. normalize for img
        if self.img_normalize != None:
            img = self.img_normalize(img)

        return img, target

    def __len__(self):
        return self.size

import torch.utils.data as data
class customDataset(data.Dataset):
    def __init__(self, mode, images, GT, transform=None, GT_transform=None):
        if images is None:
            raise(RuntimeError("images must be set"))
        # read the images
        #
        self.mode = mode
        self.images = images
        self.GT = GT
        self.transform = transform
        self.GT_transform = GT_transform

    def __getitem__(self, index):
        """
        Args:
            index(int): Index
        Returns:
            tuple: (image, GT) where GT is index of the
        """
        if self.mode == "train":
            # because of data augmentation, train images are stored in a 4-d array, with first d as sample index.
            image = self.images[index]
            # print("image shape from DataManager shown in PROMISE12:" + str(image.shape)) # e.g. 96,96,48.
            image = np.transpose(image,[2,1,0])     # x,y,z  => z,x,y
            image = np.expand_dims(image, axis=0)   # z,x,y  => 1,z,x,y
            # print("expanded image dims:{}".format(str(image.shape)))
            # pdb.set_trace()
            image = image.astype(np.float32)
            if self.transform is not None:
                image = torch.from_numpy(image)
                # image = self.transform(image)

            GT = self.GT[index]
            GT = np.transpose(GT, [2, 1, 0])
            if self.GT_transform is not None:
                GT = self.GT_transform(GT)
            return image, GT
        elif self.mode == "test":
            keys = list(self.images.keys())
            id = keys[index]
            image = self.images[id]
            image = np.transpose(image, [2, 1, 0])  # added by Chao
            image = np.expand_dims(image, axis=0)
            # print("expanded image dims:{}".format(str(image.shape)))
            # pdb.set_trace()
            image = image.astype(np.float32)
            if self.transform is not None:
                image = torch.from_numpy(image)
                # image = self.transform(image)

            GT = self.GT[id+'_segmentation'] # require customization
            GT = np.transpose(GT, [2, 1, 0]) # Batch
            if self.GT_transform is not None:
                GT = self.GT_transform(GT)
            return image, GT, id
        elif self.mode == "infer":# added by Chao
            keys = list(self.images.keys())
            id = keys[index]
            image = self.images[id]
            # print("image shape from DataManager shown in PROMISE12:" + str(image.shape))
            image = np.transpose(image,[2,1,0]) # added by Chao
            image = np.expand_dims(image, axis=0)
            image = image.astype(np.float32)
            return image, id

    def __len__(self):
        return len(self.images)
