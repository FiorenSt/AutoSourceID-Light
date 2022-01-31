import numpy as np
from astropy.io import fits
import cv2
import patchify as pat
import random


#######################################################################
#  LOADS THE 3 FIELDS AND CREATES TRAINING, TEST AND VALIDATION SETS  #
#######################################################################

def load_train_images(snr_threshold):
    DATA_PATH = 'TrainingSet/'

    ###load image
    images = np.zeros((10560, 10560), np.float32)
    fit = fits.open(DATA_PATH + 'ML1_20200601_191800_red_cosmics_nobkgsub.fits')
    images[:, :] = fit[0].data

    ###cut edges
    images = images[32:10528, 32:10528]

    ###normalization
    images = (images - np.mean(images)) / np.sqrt(np.var(images))

    ###make patches
    patches = pat.patchify(images, (256, 256), step=256)
    patches200136 = patches.reshape(1681, 256, 256, 1)  #dim: (1681, 256, 256, 1)

    ###load locations
    red_cat = fits.open(DATA_PATH + 'ML1_20200601_191800_GaiaEDR3_cat_SNR_redimage.fits')

    x_pos = red_cat[1].data['X_POS'] - 1
    y_pos = red_cat[1].data['Y_POS'] - 1

    ###save snr
    snr = red_cat[1].data['snr']
    x_pos = x_pos[snr >= snr_threshold]
    y_pos = y_pos[snr >= snr_threshold]

    positions = np.vstack((x_pos, y_pos)).T

    mask = np.zeros((10560, 10560), np.uint8)

    array_of_tuples = map(tuple, np.around(positions).astype(int))
    locations = tuple(array_of_tuples)

    ###create mask
    for location in locations:
        cv2.circle(mask, location, 2, (1, 1, 1), -1)

    mask = mask[32:10528, 32:10528]

    ###make patches for the masks
    mask = pat.patchify(mask, (256, 256), step=256)
    mask200136 = mask.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)


    ########################################################################################################################

    images = np.zeros((10560, 10560), np.float32)
    fit = fits.open(DATA_PATH + 'ML1_20210401_173445_red_cosmics_nobkgsub.fits')
    images[:, :] = fit[0].data

    images = images[32:10528, 32:10528]
    images = (images - np.mean(images)) / np.sqrt(np.var(images))

    patches = pat.patchify(images, (256, 256), step=256)

    patches175442 = patches.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)

    ### MASK
    red_cat = fits.open(DATA_PATH + 'ML1_20210401_173445_GaiaEDR3_cat_SNR_redimage.fits')

    x_pos = red_cat[1].data['X_POS'] - 1
    y_pos = red_cat[1].data['Y_POS'] - 1

    snr = red_cat[1].data['snr']

    x_pos = x_pos[snr >= snr_threshold]
    y_pos = y_pos[snr >= snr_threshold]

    positions = np.vstack((x_pos, y_pos)).T

    mask = np.zeros((10560, 10560), np.uint8)

    array_of_tuples = map(tuple, np.around(positions).astype(int))
    locations = tuple(array_of_tuples)

    for location in locations:
        cv2.circle(mask, location, 2, (1, 1, 1), -1)

    mask = mask[32:10528, 32:10528]

    mask = pat.patchify(mask, (256, 256), step=256)
    mask175442 = mask.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)


    ########################################################################################################################

    images = np.zeros((10560, 10560), np.float32)
    fit = fits.open(DATA_PATH + 'ML1_20210910_022724_red_cosmics_nobkgsub.fits')
    images[:, :] = fit[0].data

    images = images[32:10528, 32:10528]
    images = (images - np.mean(images)) / np.sqrt(np.var(images))

    patches = pat.patchify(images, (256, 256), step=256)

    patches174042 = patches.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)

    #### MASK
    red_cat = fits.open(DATA_PATH + 'ML1_20210910_022724_GaiaEDR3_cat_SNR_redimage.fits')

    x_pos = red_cat[1].data['X_POS'] - 1
    y_pos = red_cat[1].data['Y_POS'] - 1

    snr = red_cat[1].data['snr']

    x_pos = x_pos[snr >= snr_threshold]
    y_pos = y_pos[snr >= snr_threshold]

    positions = np.vstack((x_pos, y_pos)).T

    mask = np.zeros((10560, 10560), np.uint8)

    array_of_tuples = map(tuple, np.around(positions).astype(int))
    locations = tuple(array_of_tuples)

    for location in locations:
        cv2.circle(mask, location, 2, (1, 1, 1), -1)

    mask = mask[32:10528, 32:10528]

    mask = pat.patchify(mask, (256, 256), step=256)

    mask174042 = mask.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)

    #################FINAL PATCHES AND MASKS
    training = np.row_stack((patches200136, patches174042, patches175442))

    training_mask = np.row_stack((mask200136, mask174042, mask175442))

    #############################################################   TEST SET   #####################################################################################
    random.seed(2)

    index1 = [118, 800, 1000]
    index2 = random.sample(range(0, 5043), 502)
    index = index1 + index2

    test = training[index, :, :, :]
    test_mask = training_mask[index, :, :, :]

    ###REMOVE IMAGES FROM TRANING
    training = np.delete(training, (index), axis=0)
    training_mask = np.delete(training_mask, (index), axis=0)

    training.shape

    #############################################################  VALIDATION SET ######################################################

    random.seed(1)
    index = random.sample(range(0, 4539), 506)

    validation = training[index, :, :, :]
    validation_mask = training_mask[index, :, :, :]

    ###REMOVE IMAGES FROM TRANING
    training = np.delete(training, (index), axis=0)
    training_mask = np.delete(training_mask, (index), axis=0)

    return training, training_mask, test, test_mask, validation, validation_mask
