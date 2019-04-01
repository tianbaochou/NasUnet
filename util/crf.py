import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral

def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2) # width, height, nlabels
    U = -np.log(output_probs) # U should be negative log-probabilities
    U = U.reshape((2, -1))    # Needs to be flat
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U) # 设置一元势函数

    ## This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=20, compat=3)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b). no use for this data !!!
    #d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)#仅支持RGB

    # Create the pairwise bilateral term from the above image.
    # The two `s{dims,chan}` parameters are model hyper-parameters defining
    # the strength of the location and image content bilaterals, respectively.
    # chdim代表channels通道在哪个维度
    pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01), img=img, chdim=2)

    #d.addPairwiseEnergy(pairwise_energy, compat=10) # 'compat' is the "strength" of this potential

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q
