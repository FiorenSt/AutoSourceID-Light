
#
#                          ###           ######       ###      ### ##                ##
#                         #####         ##            ###      ###  ###              ##
#                        ### ###        ##            ###      ###   ###             ##
#                       ###   ###         #####       ###      ###     ###    ###    ##
#                      ### ### ###            ##      ###      ###   ###             ##
#                     ###       ###           ##      ###      ###  ###              ##
#                    ###         ###    #######       ###      ### ##                ###########
#


import numpy as np
from skimage.feature import blob_log
from joblib import Parallel, delayed
from astropy.io import fits
import patchify as pat
import sys
import tensorflow as tf



def ASID_L_loc(DATA_PATH='./TrainingSet/ML1_20200601_191800_red_cosmics_nobkgsub.fits',
           MODEL_PATH='./MODELS/TrainedModel.h5',
           train_model=False,
           demo_plot=False,
           epochs=10,
           snr_threshold=3,
           CPUs=1,
           ):

    ###load image to predict on
    image_in_patches, dim1, dim2 = load_pred_images(DATA_PATH)  ### dim.: (..., 256, 256, 1) ,  1,  1

    ###load model
    if train_model:
        training, training_mask, test, test_mask, validation, validation_mask = load_train_images(snr_threshold=snr_threshold)   ###FOR TRAINING FROM SCRATCH AND/OR PREDICTING ON TEST SET
        RUN_UNET(training,training_mask,validation,validation_mask,epochs=epochs)
        exit()
    else:
        U_net = LOAD_UNET(MODEL_PATH)

    ###U-Net prediction
    pred = U_net.predict(x=image_in_patches)  ### dim.: (..., 256, 256, 1)

    n_patches_x=int(dim1/256)
    n_patches_y=int(dim2/256)

    ###Laplacian of Gaussian
    d=joblib_loop(pred=pred,n_patches_x=n_patches_x,n_patches_y=n_patches_y,CPUs=CPUs)
    list = np.array([item for sublist in d for item in sublist]) 
    return(list)


    ###DEMO PLOT TO CHECK THE RESULTS
    if demo_plot:
        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval as zscale

        blobs_log = blob_log(pred[0,:,:,0],min_sigma=1.43, max_sigma=1.43, num_sigma=1, threshold=.2,  exclude_border=False, overlap=0.8)

        color = 'red'

        fig,ax =plt.subplots(1,1, figsize=(8,8), sharex=True, sharey=True)
        vmin, vmax = zscale().get_limits(image_in_patches[0,:,:,0])
        plt.imshow(image_in_patches[0,:,:,0], vmin=vmin, vmax=vmax,origin='lower')

        for blob in blobs_log:
            y, x, r = blob
            c = plt.Circle((x, y), 2.5, color=color, linewidth=3, fill=False)
            ax.add_patch(c)
        plt.tight_layout()
        plt.draw()
        plt.pause(10)




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





def unet(inputs):
    s = inputs

    ### Encoder path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    ### Decoder path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model


################################
#  Dice coefficient as metric  #
################################

def dice_coeff(y_true, y_pred, smooth=1e-4):
    y_true = tf.cast(y_true, tf.float32)
    #y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=(0, 1))
    union = tf.keras.backend.sum(y_true, axis=(0, 1)) + tf.keras.backend.sum(y_pred, axis=(0, 1))
    dice = (2. * intersection + smooth)/(union + smooth)
    return dice


#########################################################
#  Loss functionL: BinaryCrossentropy loss + Dice loss  #
#########################################################

def dice_BCE_loss(y_true, y_pred, smooth=1e-4):
    y_true = tf.cast(y_true, tf.float32)
    #y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2. * tf.keras.backend.sum(y_true * y_pred, axis=(0, 1)) + smooth
    denominator = tf.keras.backend.sum(y_true**2, axis=(0, 1)) + tf.keras.backend.sum(y_pred**2, axis=(0, 1)) + smooth
    dice_loss = 1 - tf.keras.backend.mean(numerator/denominator)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    BCE = bce(y_true, y_pred)
    Dice_BCE = BCE + dice_loss
    return Dice_BCE




#####################################
#  LOAD U_NET MODULE FROM .h5 FILE  #
#####################################

def LOAD_UNET(filepath):
    return tf.keras.models.load_model(filepath=filepath,custom_objects={'dice_coeff':dice_coeff},compile=False)



###############################
#  Loop for parallelization   #
###############################

def task(image, i, n_patches_x, n_patches_y):

    ###LoG step
    blobs_log = blob_log(image, min_sigma=1.43, max_sigma=1.43, num_sigma=1, threshold=.2,
                         exclude_border=False, overlap=0.8)

    ###from patch coordinates to full image coordinates
    x_idx = (i % n_patches_x) * 256
    y_idx = int(i / n_patches_y) * 256
    x_coord = x_idx + blobs_log[:, 1]
    y_coord = y_idx + blobs_log[:, 0]
    return np.column_stack((x_coord,y_coord))


######################
#  Parallelization   #
######################

def joblib_loop(pred,n_patches_x,n_patches_y,CPUs=1):
    return Parallel(n_jobs=CPUs)(delayed(task)(pred[i,:,:,0],i,n_patches_x,n_patches_y) for i in range(0,pred.shape[0]))




###############
# BUILD U-NET #
###############

def RUN_UNET(training,training_mask,validation,validation_mask,epochs=2):

    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 1

    ###Inputs
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    ###Call Model #
    U_net=unet(inputs)

    ###Compile Model

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6, verbose=1)
    save_checkpoint3 = tf.keras.callbacks.ModelCheckpoint('./MODELS/FROM_SCRATCH/TrainedModel_from_scratch.h5', verbose=1, save_best_only=True, monitor='val_loss')

    U_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss=dice_BCE_loss, metrics=['accuracy' , dice_coeff])
    U_net.summary()

    ### Fit the Model
    history=U_net.fit(x=training, y=training_mask,validation_data= (validation,validation_mask), batch_size=32,epochs=epochs, verbose=1, shuffle=True,
                                    callbacks=[save_checkpoint3, reduce_lr])





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

    ###load image
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

    ###load image
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













if __name__ == "__main__":
   # DATA_PATH = sys.argv[1]
   # MODEL_PATH = sys.argv[2]   # './MODELS/TrainedModel.h5'
    ASID_L()
