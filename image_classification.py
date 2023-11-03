import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from functions import *

###########################################################
#                  Parameters setting                     #
###########################################################
class_num = 5
F_tb_path = r"data/Coleambally/S2_20211223.tif"
F_tb_class_path = rf"data/Coleambally/S2_20211223_KMeans-{class_num}.tif"


if __name__ == "__main__":
    image, profile = read_raster(F_tb_path)
    print(profile)

    image_pct2 = linear_pct_stretch(image, 2)
    image_pct2 = color_composite(image_pct2, [3, 2, 1])

    kmeans = KMeans(n_clusters=class_num, max_iter=300)
    kmeans.fit(image.reshape(-1, image.shape[2]))
    F_class = kmeans.labels_.reshape(image.shape[0], image.shape[1])

    # fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    # axes[0].imshow(image_pct2)
    # axes[1].imshow(F_class, cmap="Paired")
    # plt.show()

    write_raster(np.expand_dims(F_class, axis=2), profile, F_tb_class_path)
    print("Classified the fine image using K-Means algorithm!")



