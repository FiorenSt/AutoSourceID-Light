
#
#                          ###           ######       ###      ### ##                ##
#                         #####         ##            ###      ###  ###              ##
#                        ### ###        ##            ###      ###   ###             ##
#                       ###   ###         #####       ###      ###     ###    ###    ##
#                      ### ### ###            ##      ###      ###   ###             ##
#                     ###       ###           ##      ###      ###  ###              ##
#                    ###         ###    #######       ###      ### ##                ###########
#



import argparse
from Training_Test_ValidationSet import *
from LOAD_UNET import *
from LaplacianOfGaussian import *
from LOAD_IMAGES import *
from RUN_UNET_fromScratch import *


parser = argparse.ArgumentParser()
parser.add_argument("-DATA_PATH",default='./TrainingSet/ML1_20200601_191800_red_cosmics_nobkgsub.fits',type=str)
parser.add_argument("-MODEL_PATH",default='./MODELS/TrainedModel.h5', type=str)
parser.add_argument("-train_model", action='store_true', default=False)
parser.add_argument("-demo_plot", action='store_true', default=False)
parser.add_argument("-epochs", type=int, default=10)
parser.add_argument("-snr_threshold", type=int, default=3)
parser.add_argument("-CPUs", type=int, default=1)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()


def ASID_L():

    ###load image to predict on
    image_in_patches, dim1, dim2 = load_pred_images(args.DATA_PATH)  ### dim.: (..., 256, 256, 1) ,  1,  1

    ###load model
    if args.train_model:
        training, training_mask, test, test_mask, validation, validation_mask = load_train_images(snr_threshold=args.snr_threshold)   ###FOR TRAINING FROM SCRATCH AND/OR PREDICTING ON TEST SET
        RUN_UNET(training,training_mask,validation,validation_mask,epochs=args.epochs)
        exit()
    else:
        U_net = LOAD_UNET(args.MODEL_PATH)

    ###U-Net prediction
    pred = U_net.predict(x=image_in_patches)  ### dim.: (..., 256, 256, 1)

    n_patches_x=int(dim1/256)
    n_patches_y=int(dim2/256)

    ###Laplacian of Gaussian
    d=joblib_loop(pred=pred,n_patches_x=n_patches_x,n_patches_y=n_patches_y,CPUs=args.CPUs)
    list = np.array([item for sublist in d for item in sublist]) + 1   ##+1 because coordinates start from 1, not 0
    np.savetxt('./RESULTS/coordinates.txt', list,delimiter=',', fmt='%i')


    ###DEMO PLOT TO CHECK THE RESULTS
    if args.demo_plot:
        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval as zscale

        blobs_log = blob_log(pred[100,:,:,0],min_sigma=1.43, max_sigma=1.43, num_sigma=1, threshold=.2,  exclude_border=False, overlap=0.8)

        color = 'red'

        fig,ax =plt.subplots(1,1, figsize=(8,8), sharex=True, sharey=True)
        vmin, vmax = zscale().get_limits(image_in_patches[100,:,:,0])
        plt.imshow(image_in_patches[100,:,:,0], vmin=vmin, vmax=vmax,origin='lower')

        for blob in blobs_log:
            y, x, r = blob
            c = plt.Circle((x, y), 2.5, color=color, linewidth=3, fill=False)
            ax.add_patch(c)
        plt.tight_layout()
        plt.draw()
        plt.pause(10)





if __name__ == "__main__":
   # DATA_PATH = sys.argv[1]
   # MODEL_PATH = sys.argv[2]   # './MODELS/TrainedModel.h5'
    ASID_L()