import numpy as np
from skimage.feature import blob_log
from joblib import Parallel, delayed



###############################
#  Loop for parallelization   #
###############################

def task(image,i,n_patches_x,n_patches_y):

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

def joblib_loop(pred,n_patches_x,n_patches_y,CPUs=6):
    return Parallel(n_jobs=CPUs)(delayed(task)(pred[i,:,:,0],i,n_patches_x,n_patches_y) for i in range(0,pred.shape[0]))




