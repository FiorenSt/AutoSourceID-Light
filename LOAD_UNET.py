import sys
from U_Net import *


def LOAD_UNET(file_name):
    ###########LOAD MODELS
    return tf.keras.models.load_model(filepath='GITHUB_PROVISORY/MODELS/'+file_name, custom_objects={'dice_coeff': dice_coeff}, compile=False)



if __name__ == "__main__":
    file_name = sys.argv[1]
    U_net = LOAD_UNET(file_name)
    #pred = U_net.predict(x=test)
