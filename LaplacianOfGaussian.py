import numpy as np
from skimage.feature import blob_log
from joblib import Parallel, delayed


def task(image,i):
    blobs_log = blob_log(image, min_sigma=1.38, max_sigma=1.55, num_sigma=5, threshold=.1,
                         exclude_border=False, overlap=0.9)
    x_idx = (i % 41) * 256 + 32
    y_idx = int(i / 41) * 256 + 32
    x_coord = x_idx + blobs_log[:, 1]
    y_coord = y_idx + blobs_log[:, 0]
    return np.column_stack((x_coord,y_coord))


def joblib_loop(pred):
    return Parallel(n_jobs=6)(delayed(task)(pred[i,:,:,0],i) for i in range(0,pred.shape[0]))


#np.savetxt('Magellanic_coord.txt', list,delimiter=',',fmt='%i')




















