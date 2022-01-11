import sys
from LOAD_UNET import *
from Training_Test_ValidationSet import *


U_net = LOAD_UNET('trained_snr0.h5')

training, training_mask, test, test_mask, validation, validation_mask = load_train_images(2)
pred = U_net.predict(x=test)