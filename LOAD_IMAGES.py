import numpy as np
from astropy.io import fits
import cv2
import patchify as pat


##LOAD 3 FIELDS AND CREATE MASKS

def load_pred_images(DATA_PATH):
    images = np.zeros((10560, 10560), np.float32)
    fit = fits.open(DATA_PATH)
    images[:, :] = fit[0].data

    images = images[32:10528, 32:10528]
    images = (images - np.mean(images)) / np.sqrt(np.var(images))

    patches = pat.patchify(images, (256, 256), step=256)
    patches = patches.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)

    return patches
