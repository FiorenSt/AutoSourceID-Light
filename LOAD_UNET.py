import sys
from U_Net import *


def LOAD_UNET(): ### description format/whatever
    ###########LOAD MODELS
    return tf.keras.models.load_model(filepath='C:/Users/fiore/Desktop/UNI/Projects/Project7-PointSourcesIdentification/ASID-L/MODELS/TrainedModel.h5',custom_objects={'dice_coeff':dice_coeff},compile=False)


if __name__ == "__main__":
    file_name = sys.argv[1] ### 0 or 1?
    print(file_name)
    U_net = LOAD_UNET()
    #pred = U_net.predict(x=test)
