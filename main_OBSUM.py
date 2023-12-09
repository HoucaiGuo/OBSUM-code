import numpy as np
from functions import *
from sklearn.cluster import KMeans
from datetime import datetime
from OBSUM import OBSUM
from skimage.transform import downscale_local_mean
import os

###########################################################
#                  Parameters setting                     #
###########################################################

# Scale factor between coarse and fine image,
# i.e., for Sentinel-3 and Sentinel-2, it's 300 m / 10 m = 30
scale_factor = 30
# Number of land-cover classes in the base fine image
class_num = 5
# Size of the local unmixing window,
# it's recommended to use a large value, e.g., 15
win_size = 15
# Percentage of the selected fine pixels for object-level residual compensation,
# it's recommended to use a small value, e.g., 5
object_RC_percent = 5
# Size of the local window for similar pixels selection
similar_win_size = 31
# Number of similar pixels need to be selected
similar_num = 30
# Min and max value of the data,
# for land surface reflectance, use 0 and 1
min_val = 0.0
max_val = 1.0

dataset = "Sentinel-Butte"
tb = "20220713"
tp = "20220723"

# Path of the fine image at tb, i.e., the base fine image
F_tb_path = r"data/Sentinel-Butte/S2_20220213.tif"
# Path of the image objects segmented from the base fine image
F_tb_objects_path = r"data/Sentinel-Butte/S2_20220213_sam.tif"
# Path of the land-cover classification map of the base fine image
F_tb_class_path = r"data/Sentinel-Butte/S2_20220213_KMeans-5.tif"
# Path of the coarse image at tb, i.e., the base coarse image. The image need to be resized
# to the resolution of the fine image using a nearest neighbor interpolation.
C_tb_path = r"data/Sentinel-Butte/S3_20220213.tif"
# Path of the coarse image at tp, which will be downscaled by OBSUM. The image need to be resized
# # to the resolution of the fine image using a nearest neighbor interpolation.
C_tp_path = r"data/Sentinel-Butte/S3_20220723.tif"

# save path
F_tp_OBSUM_path = f"data/Sentinel-Butte/20220213/20220723/OBSUM/OBSUM_real.tif"

if __name__ == '__main__':
    # load the input images
    F_tb, F_tb_profile = read_raster(F_tb_path)
    C_tb = read_raster(C_tb_path)[0]
    C_tb_coarse = downscale_local_mean(C_tb, factors=(scale_factor, scale_factor, 1))
    C_tp = read_raster(C_tp_path)[0]
    C_tp_coarse = downscale_local_mean(C_tp, factors=(scale_factor, scale_factor, 1))
    F_tb_objects = read_raster(F_tb_objects_path)[0][:, :, 0]

    # if the land-cover classification map of the base fine image is not provided, classify the
    # fine image using the unsupervised K-Means algorithm
    if not os.path.exists(F_tb_class_path):
        kmeans = KMeans(n_clusters=class_num, max_iter=300)
        kmeans.fit(F_tb.reshape(-1, F_tb.shape[2]))
        F_tb_class = kmeans.labels_.reshape(F_tb.shape[0], F_tb.shape[1])
        write_raster(np.expand_dims(F_tb_class, axis=2), F_tb_profile, F_tb_class_path)
        print("Classified the fine image using K-Means algorithm!")
    else:
        F_tb_class = read_raster(F_tb_class_path)[0][:, :, 0]
        print("Loaded the land-cover classification map!")

    time0 = datetime.now()
    obsum = OBSUM(F_tb, C_tb_coarse, C_tp_coarse, F_tb_class, F_tb_objects,
                  class_num=class_num, scale_factor=scale_factor, win_size=win_size,
                  OL_RC_percent=object_RC_percent,
                  similar_win_size=similar_win_size, similar_num=similar_num,
                  min_val=min_val, max_val=max_val)
    F_tp_OBSUM = obsum.object_based_spatial_unmixing()
    time1 = datetime.now()
    time_span = time1 - time0
    print(f"Used {time_span.total_seconds():.2f} seconds!")

    # save the predicted image
    write_raster(F_tp_OBSUM, F_tb_profile, F_tp_OBSUM_path)





