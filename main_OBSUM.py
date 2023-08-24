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
scale_factor = 30
class_num = 5
win_size = 11
object_RC_percent = 15
similar_win_size = 31
similar_num = 30

dataset = "Butte"
tb = "20220213"
tp = "20220623"

# dataset = "Coleambally"
# tb = "20211223"
# tp = "20220803"

F_tb_path = f"data/{dataset}/S2_{tb}.tif"
F_tb_objects_path = f"data/{dataset}/S2_{tb}_sam.tif"
F_tb_class_path = f"data/{dataset}/S2_{tb}_KMeans-{class_num}.tif"
F_tp_path = f"data/{dataset}/S2_{tp}.tif"
C_tp_path = f"data/{dataset}/S3_{tp}.tif"

F_tp_OB_U_path = f"data/{dataset}/{tb}/{tp}/OBSUM/OB-U.tif"
F_tp_OL_RC_path = f"data/{dataset}/{tb}/{tp}/OBSUM/OL-RC.tif"
F_tp_OBSUM_path = f"data/{dataset}/{tb}/{tp}/OBSUM/OBSUM.tif"

if __name__ == '__main__':
    F_tb, F_tb_profile = read_raster(F_tb_path)
    print(F_tb_profile)
    F_tp = read_raster(F_tp_path)[0]
    C_tp = read_raster(C_tp_path)[0]
    C_tp_coarse = downscale_local_mean(C_tp, factors=(scale_factor, scale_factor, 1))
    F_tb_objects = read_raster(F_tb_objects_path)[0][:, :, 0]

    if not os.path.exists(F_tb_class_path):
        kmeans = KMeans(n_clusters=class_num, max_iter=300)
        kmeans.fit(F_tb.reshape(-1, F_tb.shape[2]))
        F_tb_class = kmeans.labels_.reshape(F_tb.shape[0], F_tb.shape[1])
        write_raster(np.expand_dims(F_tb_class, axis=2), F_tb_profile, F_tb_class_path)
        print("Classified the fine image using K-Means algorithm!")
    else:
        F_tb_class = read_raster(F_tb_class_path)[0][:, :, 0]
        print("Loaded classification map!")

    time0 = datetime.now()

    obsum = OBSUM(F_tb, C_tp_coarse, F_tb_class, F_tb_objects,
                  class_num=class_num, scale_factor=scale_factor, win_size=win_size,
                  OL_RC_percent=object_RC_percent,
                  similar_win_size=similar_win_size, similar_num=similar_num)
    F_tp_OB_U, F_tp_OL_RC, F_tp_OBSUM = obsum.object_based_spatial_unmixing()

    time1 = datetime.now()
    time_span = time1 - time0
    print(f"Used {time_span.total_seconds():.2f} seconds!")

    write_raster(F_tp_OB_U, F_tb_profile, F_tp_OB_U_path)
    write_raster(F_tp_OL_RC, F_tb_profile, F_tp_OL_RC_path)
    write_raster(F_tp_OBSUM, F_tb_profile, F_tp_OBSUM_path)





