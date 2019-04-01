# There are a huge number of similar examples in the training set and this puts a (somewhat low)
# upper bound on the best result you can achieve, regardless of the model.
# From https://github.com/julienr/kaggle_uns/blob/master/13_clean/0_filter_incoherent_images.ipynb
import os
import numpy as np
import glob
import cv2
import pylab as pl
import skimage.util
import skimage
import skimage.io
import matplotlib.cm as cm
import scipy.spatial.distance as spdist

def create_exp_dir(path, desc='Experiment dir: {}'):
    if not os.path.exists(path):
        os.makedirs(path)
    print(desc.format(path))

# Hard-Dice
def dice_coefficient(Y_pred, Y):
    """
    This works for one image
    http://stackoverflow.com/a/31275008/116067
    """
    denom = (np.sum(Y_pred == 1) + np.sum(Y == 1))
    if denom == 0:
        # By definition, see https://www.kaggle.com/c/ultrasound-nerve-segmentation/details/evaluation
        return 1
    else:
        return 2 * np.sum(Y[Y_pred == 1]) / float(denom)

class FilerImages(object):
    """
    Filter incoherent images
    """
    def __init__(self, train_path, clean_dir):
        self.train_path = train_path
        self.clean_dir = clean_dir


    def load_and_preprocess(self, imgname):
        img_fname = imgname
        mask_fname = os.path.splitext(imgname)[0] + "_mask.tif"
        img = cv2.imread(os.path.join(TRAIN_PATH, img_fname), cv2.IMREAD_GRAYSCALE)
        assert img is not None
        mask = cv2.imread(os.path.join(TRAIN_PATH, mask_fname), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        mask = (mask > 128).astype(np.float32)

        # TODO: Could subtract mean as on fimg above
        img = img.astype(np.float32) / 255.0
        np.ascontiguousarray(img)
        return img, mask

    def load_patient(self, pid):
        fnames = [os.path.basename(fname) for fname in glob.glob(self.train_path  \
                                        + "/%d_*.tif" % pid) if 'mask' not in fname]
        imgs, masks = zip(*map(self.load_and_preprocess, fnames))
        return imgs, masks, fnames

    def show(self,i):
        pl.figure(figsize=(10, 4))
        pl.suptitle(self.fnames[i])
        pl.subplot(121)
        pl.imshow(self.imgs[i], cmap=cm.gray)

        pl.subplot(122)
        pl.imshow(self.imgs[i], cmap=cm.gray)
        h, w = self.imgs[i].shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[self.masks[i] > 0, :] = (200, 30, 30, 255)
        pl.imshow(overlay, alpha=1)

    def compute_img_hist(self, img):
        # Divide the image in blocks and compute per-block histogram
        blocks = skimage.util.view_as_blocks(img, block_shape=(20, 20))
        img_hists = [np.histogram(block, bins=np.linspace(0, 1, 10))[0] for block in blocks]
        return np.concatenate(img_hists)

    def compute_patience(self, id=6):
        imgs, masks, fnames = self.load_patient(id)
        hists = np.array([*map(self.compute_img_hist, imgs)])
        D = spdist.squareform(spdist.pdist(hists, metric='cosine'))

        close_pairs = D + np.eye(D.shape[0]) < 0.008
        farthest = np.argmax(D[close_pairs])
        close_ij = np.transpose(np.nonzero(close_pairs))
        incoherent_ij = [(i, j) for i, j in close_ij if dice_coefficient(masks[i], masks[j]) < 0.2]
        incoherent_ij = np.array(incoherent_ij)

        i, j = incoherent_ij[np.random.randint(incoherent_ij.shape[0])]
        print(dice_coefficient(masks[i], masks[j]))
        print("D : ", D[i, j])
        self.show(i)
        self.show(j)
        # pl.imshow(imgs[close_ij[farthest, 0]])
        # pl.figure()
        # pl.imshow(imgs[close_ij[farthest, 1]])
        # pl.imshow(close_pairs)
        pl.show()

    def filter_images_for_patient(self, pid):
        imgs, masks, fnames = self.load_patient(pid)
        hists = np.array([*map(self.compute_img_hist, imgs)])
        D = spdist.squareform(spdist.pdist(hists, metric='cosine'))

        # Used 0.005 to train at 0.67
        close_pairs = D + np.eye(D.shape[0]) < 0.005
        close_ij = np.transpose(np.nonzero(close_pairs))

        incoherent_ij = [(i, j) for i, j in close_ij if dice_coefficient(masks[i], masks[j]) < 0.2]
        incoherent_ij = np.array(incoherent_ij)

        valids = np.ones(len(imgs), dtype=np.bool)
        for i, j in incoherent_ij:
            if np.sum(masks[i]) == 0:
                valids[i] = False
            if np.sum(masks[j]) == 0:
                valids[i] = False

        for i in np.flatnonzero(valids):
            imgname = os.path.splitext(fnames[i])[0] + ".png"
            mask_fname = os.path.splitext(imgname)[0] + "_mask.png"
            img = skimage.img_as_ubyte(imgs[i])
            cv2.imwrite(os.path.join(OUTDIR, imgname), img)
            mask = skimage.img_as_ubyte(masks[i])
            cv2.imwrite(os.path.join(OUTDIR, mask_fname), mask)
        print ('Discarded ', np.count_nonzero(~valids), " images for patient %d" % pid)
        return np.count_nonzero(~valids)

    def run_filter(self):
        create_exp_dir(self.clean_dir)
        removed = 0
        for pid in range(1, 48):
            removed += self.filter_images_for_patient(pid)
        print('Total removed {}'.format(removed))

if __name__ == '__main__':
    TRAIN_PATH = '/train_tiny_data/imgseg/ultrasound-nerve/train'
    OUTDIR = '/train_tiny_data/imgseg/ultrasound-nerve/data_clean'
    fi = FilerImages(train_path=TRAIN_PATH, clean_dir=OUTDIR)
    fi.run_filter()