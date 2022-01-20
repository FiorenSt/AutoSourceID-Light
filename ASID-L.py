import sys
from Training_Test_ValidationSet import *
from LOAD_UNET import *
from LaplacianOfGaussian import *
from LOAD_IMAGES import *



def ASID_L(DATA_PATH):

    #training, training_mask, test, test_mask, validation, validation_mask = load_train_images(2)   ###FOR TRAINING FROM SCRATCH AND/OR PREDICTING ON TEST SET
    test=load_pred_images(DATA_PATH)

    U_net = LOAD_UNET() #
    pred = U_net.predict(x=test)
    d=joblib_loop(pred)

    list = [item for sublist in d for item in sublist]
    list = np.array(list)  + 1  ##+1 because coordinates start from 1, not 0

    np.savetxt('C:/Users/fiore/Desktop/UNI/Projects/Project7-PointSourcesIdentification/GITHUB_PROVISORY/RESULTS/coordinates.txt', list, fmt='%1.2f')


if __name__ == "__main__":
    #DATA_PATH = sys.argv[1]
    DATA_PATH='C:/Users/fiore/Desktop/ML1_20210910_022724_red_cosmics_nobkgsub.fits'
    ASID_L(DATA_PATH)