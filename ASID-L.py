import sys
from Training_Test_ValidationSet import *
from LOAD_UNET import *
from LaplacianOfGaussian import *
from LOAD_IMAGES import *



def ASID_L(DATA_PATH, load_model=True):

    test=load_pred_images(DATA_PATH)
    if load_model:
        U_net = LOAD_UNET()  #
    else:
        training, training_mask, test, test_mask, validation, validation_mask = load_train_images(2)   ###FOR TRAINING FROM SCRATCH AND/OR PREDICTING ON TEST SET
        ...

    pred = U_net.predict(x=test)
    d=joblib_loop(pred)
    list = np.array([item for sublist in d for item in sublist]) + 1   ##+1 because coordinates start from 1, not 0

    np.savetxt('C:/Users/fiore/Desktop/UNI/Projects/Project7-PointSourcesIdentification/ASID-L/RESULTS/coordinates.txt', list,delimiter=',', fmt='%i')


if __name__ == "__main__":
    DATA_PATH = sys.argv[1]
    ASID_L(DATA_PATH)