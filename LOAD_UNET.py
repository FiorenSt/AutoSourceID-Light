import sys
from U_Net import *


def LOAD_UNET():
    ###########LOAD MODELS
    return tf.keras.models.load_model(filepath='C:/Users/fiore/Desktop/UNI/Projects/Project7-PointSourcesIdentification/GITHUB_PROVISORY/MODELS/trained_snr0.h5', custom_objects={'dice_coeff': dice_coeff}, compile=False)


if __name__ == "__main__":
    #file_name = sys.argv[1]
    U_net = LOAD_UNET()
    #pred = U_net.predict(x=test)
