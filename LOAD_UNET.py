import sys
from U_Net import *


#####################################
#  LOAD U_NET MODULE FROM .h5 FILE  #
#####################################

def LOAD_UNET(filepath):
    return tf.keras.models.load_model(filepath=filepath,custom_objects={'dice_coeff':dice_coeff},compile=False)


if __name__ == "__main__":
    file_name = sys.argv[1]  #'./MODELS/TrainedModel.h5'
    U_net = LOAD_UNET(filepath=file_name)
