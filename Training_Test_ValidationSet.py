import numpy as np
from astropy.io import fits
import cv2
import patchify as pat
import random


##LOAD 3 FIELDS AND CREATE MASKS

def load_train_images(snr_threshold):
    DATA_PATH = 'GITHUB_PROVVISORY/Training Set/'

    images = np.zeros((10560, 10560), np.float32)
    fit = fits.open(DATA_PATH + 'ML1_20200530_200136_red.fits')
    images[:, :] = fit[0].data

    images = images[32:10528, 32:10528]
    images = (images - np.mean(images)) / np.sqrt(np.var(images))

    patches = pat.patchify(images, (256, 256), step=256)
    patches200136 = patches.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)

    ##################REAL MASK
    red_cat = fits.open(DATA_PATH + 'ML1_20200530_200136_GaiaEDR3_cat_SNR_redimage.fits')

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

    mask200136 = mask.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)

    ########################################################################################################################

    images = np.zeros((10560, 10560), np.float32)
    fit = fits.open(DATA_PATH + 'ML1_20210312_175442_red.fits')
    images[:, :] = fit[0].data

    images = images[32:10528, 32:10528]
    images = (images - np.mean(images)) / np.sqrt(np.var(images))

    patches = pat.patchify(images, (256, 256), step=256)

    patches175442 = patches.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)

    ##################REAL MASK
    red_cat = fits.open(DATA_PATH + 'ML1_20210312_175442_GaiaEDR3_cat_SNR_redimage.fits')

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

    ###################################################################################################

    #images = np.zeros((10560, 10560), np.float32)
    #fit = fits.open(DATA_PATH + 'ML1_20210330_183742_red.fits')
    #images[:, :] = fit[0].data

    #images = images[32:10528, 32:10528]
    #images = (images - np.mean(images)) / np.sqrt(np.var(images))

    #patches = pat.patchify(images, (256, 256), step=256)
    #patches183742 = patches.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)

    ##################REAL MASK
    #red_cat = fits.open(DATA_PATH + 'ML1_20210330_183742_GaiaEDR3_cat_SNR_redimage.fits')

    #x_pos = red_cat[1].data['X_POS'] - 1
    #y_pos = red_cat[1].data['Y_POS'] - 1

    #snr = red_cat[1].data['snr']

    #x_pos = x_pos[snr >= snr_threshold]
    #y_pos = y_pos[snr >= snr_threshold]

    #positions = np.vstack((x_pos, y_pos)).T

    #mask = np.zeros((10560, 10560), np.uint8)

    #array_of_tuples = map(tuple, np.around(positions).astype(int))
    #locations = tuple(array_of_tuples)

    #for location in locations:
    #    cv2.circle(mask, location, 2, (1, 1, 1), -1)

    #mask = mask[32:10528, 32:10528]

    #mask = pat.patchify(mask, (256, 256), step=256)
    #mask183742 = mask.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)

    ###############################################################

    images = np.zeros((10560, 10560), np.float32)
    fit = fits.open(DATA_PATH + 'ML1_20210331_174042_red.fits')
    images[:, :] = fit[0].data

    images = images[32:10528, 32:10528]
    images = (images - np.mean(images)) / np.sqrt(np.var(images))

    patches = pat.patchify(images, (256, 256), step=256)

    patches174042 = patches.reshape(1681, 256, 256, 1)  # .transpose(1,2,0)

    ##################REAL MASK
    red_cat = fits.open(DATA_PATH + 'ML1_20210331_174042_GaiaEDR3_cat_SNR_redimage.fits')

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
    training = np.row_stack((patches200136, patches174042, patches175442))  ## REMOVED

    training_mask = np.row_stack((mask200136, mask174042, mask175442))  ## REMOVED

    #############################################################   TEST SET   #####################################################################################

    random.seed(2)

    index1 = [80, 800, 1000]
    index2 = random.sample(range(0, 5043), 1005)
    index = index1 + index2

    test = training[index, :, :, :]
    test_mask = training_mask[index, :, :, :]

    ###REMOVE IMAGES FROM TRANING
    training = np.delete(training, (index), axis=0)
    training_mask = np.delete(training_mask, (index), axis=0)

    #############################################################  VALIDATION SET ######################################################

    random.seed(1)
    index = random.sample(range(0, 4035), 1007)

    validation = training[index, :, :, :]
    validation_mask = training_mask[index, :, :, :]

    ###REMOVE IMAGES FROM TRANING
    training = np.delete(training, (index), axis=0)
    training_mask = np.delete(training_mask, (index), axis=0)

    return training, training_mask, test, test_mask, validation, validation_mask
