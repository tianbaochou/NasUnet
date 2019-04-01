import numpy as np
from scipy.ndimage import morphology
import os
import SimpleITK as sitk
from skimage.measure import find_contours
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from util.utils import resize_pred_to_val

def biomedical_image_metric(y_pred, image_nums, image_path):
    """
    :param y_pred: numpy array
    :param image_nums: the image id in case
    :param image_path: the case file path
    :return:
    """
    vol_scores = []
    ravd = []
    scores = []
    hauss_dist = []
    mean_surf_dist = []

    # concate image
    img_rows, img_cols = y_pred[0].shape[0], y_pred[0].shape[1]
    y_pred = np.concatenate(y_pred, axis=0).reshape(-1, img_rows, img_cols, 1)

    start_ind = 0
    end_ind  = 0
    for y_true, spacing in read_cases(imgn = image_nums, folder = image_path):

        start_ind = end_ind
        end_ind +=len(y_true)

        y_pred_up = resize_pred_to_val(y_pred[start_ind:end_ind], y_true.shape)

        ravd.append( rel_abs_vol_diff( y_true , y_pred_up ) )
        vol_scores.append( numpy_dice( y_true , y_pred_up , axis=None) )
        surfd = surface_dist(y_true , y_pred_up, sampling=spacing)
        hauss_dist.append( surfd.max() )     # max hausdorff
        mean_surf_dist.append(surfd.mean())  # mean hausdorff
        axis = tuple( range(1, y_true.ndim ) )
        scores.append( numpy_dice( y_true, y_pred_up , axis=axis) )

    ravd = np.array(ravd)
    vol_scores = np.array(vol_scores)
    scores = np.concatenate(scores, axis=0)

    print('Mean volumetric DSC:', vol_scores.mean() )
    print('Median volumetric DSC:', np.median(vol_scores) )
    print('Std volumetric DSC:', vol_scores.std() )
    print('Mean Hausdorff Dist:', np.mean(hauss_dist) )
    print('Mean Mean Hausdorff Dist:', np.mean(mean_surf_dist) )
    print('Mean Rel. Abs. Vol. Diff:', ravd.mean() )

def read_cases(imgn=None, folder='../data/train/', masks=True):
    """
    :param imgn: (int) image id
    :param folder: (string) the case file path
    :param mask: (bool) whether load mask or not
    :return: iteration object
    """
    fileList =  os.listdir(folder)
    fileList = [x for x in fileList if '.mhd' in x]
    if masks:
        fileList = [x for x in fileList if 'segm' in x.lower()]
    sorted(fileList)
    if imgn is not None:
        fileList = [file for file in fileList for ff in imgn if str(ff).zfill(2) in file]

    for filename in fileList:
        itkimage = sitk.ReadImage(folder+filename)
        imgs = sitk.GetArrayFromImage(itkimage)
        yield imgs, itkimage.GetSpacing()[::-1]

def make_plots(X, y, y_pred, n_best=20, n_worst=20):
    #PLotting the results'
    img_rows = X.shape[1]
    img_cols = img_rows
    axis =  tuple( range(1, X.ndim ) )
    scores = numpy_dice(y, y_pred, axis=axis )
    sort_ind = np.argsort( scores )[::-1]
    indice = np.nonzero( y.sum(axis=axis) )[0]
    #Add some best and worst predictions
    img_list = []
    count = 1
    for ind in sort_ind:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_best:
            break

    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X[img_list].reshape(-1,img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    img_list = []
    count = 1
    for ind in sort_ind[::-1]:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_worst:
            break

    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X[img_list].reshape(-1,img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    n_cols= 4
    n_rows = int( np.ceil(len(img)/n_cols) )

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    for mm in range( len(img) ):
        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm] )
        contours = find_contours(segm[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')

        contours = find_contours(segm_pred[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)  # aspect ratio of 1

    fig.savefig('../images/worst_predictions.png', bbox_inches='tight', dpi=300 )

def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):
    intersection = y_true*y_pred
    return ( 2. * intersection.sum(axis=axis) +smooth)/ (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) +smooth )

def rel_abs_vol_diff(y_true, y_pred):
    return np.abs( (y_pred.sum()/y_true.sum() - 1)*100)

def get_boundary(data, img_dim=2, shift = -1):
    data  = data>0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data,shift=shift,axis=nn))
    return edge.astype(int)

def surface_dist(input1, input2, sampling=1, connectivity=1):
    # Hausdorff distance
    # HD(A,B) = max(h(A,B), h(B,A)
    input1 = np.squeeze(input1)
    input2 = np.squeeze(input2)

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
    S = np.logical_xor(input_1, morphology.binary_erosion(input_1, conn))
    Sprime = np.logical_or(input_2, morphology.binary_erosion(input_2, conn))

    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)

    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

    return sds

