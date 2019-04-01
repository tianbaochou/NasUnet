import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import SimpleITK as sitk
from util.utils import resize_pred_to_val

def predict_test(y_pred , X_test_list, dest='../data/predictions'):
    """
    :param y_pred: all of pred image
    :param X_test_list: the test image path list
    :param dest: the path to store predictions
    :return:
    """
    import re
    if not os.path.isdir(dest):
        os.mkdir(dest)

    # concate image
    img_rows, img_cols = y_pred[0].shape[0], y_pred[0].shape[1]
    y_pred = np.concatenate(y_pred, axis=0).reshape(-1, img_rows, img_cols, 1)

    # write the result
    start_ind=0
    end_ind=0
    for file_path in X_test_list:
        itkimage = sitk.ReadImage(file_path[0])
        img = sitk.GetArrayFromImage(itkimage)
        start_ind = end_ind
        end_ind +=len(img)
        pred = resize_pred_to_val(y_pred[start_ind:end_ind], img.shape )
        pred = np.squeeze(pred)
        mask = sitk.GetImageFromArray(pred)
        mask.SetOrigin( itkimage.GetOrigin() )
        mask.SetDirection( itkimage.GetDirection() )
        mask.SetSpacing( itkimage.GetSpacing() )
        _, filename = os.path.split(file_path[0])
        sitk.WriteImage(mask, dest+'/'+filename[:-4]+'_segmentation.mhd')
