import numpy as np
from astropy.io import fits
import patchify as pat


#####################################
#  LOAD IMAGES WE WANT TO LOCALIZE  #
#####################################

def load_pred_images(DATA_PATH):

    ###load fits file
    fit = fits.open(DATA_PATH)[0].data

    ###normalization
    fit = (fit - np.mean(fit)) / np.sqrt(np.var(fit))

    ###ensure that the image is divisible by 256
    dim1 = fit.shape[0] + (256 - fit.shape[0] % 256)
    dim2 = fit.shape[1] + (256 - fit.shape[0] % 256)
    images = np.zeros((dim1, dim2), np.float32)
    images[0:fit.shape[0],0:fit.shape[1]]= fit

    ###patchify in 256x256
    patches = pat.patchify(images, (256, 256), step=256)
    patches = patches.reshape(int(dim1*dim2/256/256), 256, 256, 1)

    return patches, dim1, dim2  ###final dim: (..., 256, 256, 1) 1, 1
