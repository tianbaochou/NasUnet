import os
import argparse
import numpy as np
import visdom
import pickle
import time

def main(args):
    vis = visdom.Visdom(port=9000)

    losses = pickle.load(open(os.path.join(args.path, 'saved_loss.p'),'rb'))
    x = np.squeeze(np.asarray(losses['X']))
    y = np.squeeze(np.asarray(losses['Y']))
    yval = np.squeeze(np.asarray(losses['Y_val']))
    vis.line(np.vstack((y, yval)).T, x, env='loss_acc', opts=dict(title='Loss'))

    accuracy = pickle.load(open(os.path.join(args.path, 'saved_accuracy.p'),'rb'))
    x = np.squeeze(np.asarray(accuracy['X']))
    pixAcc_train = accuracy['pixAcc_train']
    mIoU_train =accuracy['mIoU_train']
    pixAcc_val = accuracy['pixAcc_val']
    mIoU_val =accuracy['mIoU_val']

    vis.line(np.vstack((np.array(pixAcc_train).T, np.array(pixAcc_val).T)).T, x, env='pix_acc', opts=dict(title='Pixel Accuary'))
    vis.line(np.vstack((np.array(mIoU_train).T, np.array(mIoU_val).T)).T, x, env='mean_acc', opts=dict(title='Mean Accuracy'))

    if args.images:
        files = [f for f in os.listdir(os.path.join(args.path, 'saved_val_images'))]
        files.sort()

        for f in files:
            if f.endswith('.p'):
                image = pickle.load(open(os.path.join(args.path, 'saved_val_images', f), 'rb'))
                if len(image.shape) == 4:
                    vis.image(image[0], env='images', opts=dict(title=f))
                else:
                    if image.shape[0] != 3 and image.shape[0] != 1:
                        image = image.transpose((2, 0, 1))
                    vis.image(image, env='images', opts=dict(title=f))
                time.sleep(0.25)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display param')
    parser.add_argument('--images', help='Also displays images from validation set', action='store_true',
                        default=True)
    parser.add_argument('--path', nargs='?', type=str, default='../logs/nas-unet/train/train-20190104-1915/',
                        help='Configuration file to use')
    args = parser.parse_args()
    main(args)