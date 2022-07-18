import sys
from U_Net import *
from Training_Test_ValidationSet import *


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



if __name__ == "__main__":
    snr_threshold = int(sys.argv[1])
    epochs = int(sys.argv[2])
    RUN_UNET(snr_threshold=snr_threshold,epochs=epochs)