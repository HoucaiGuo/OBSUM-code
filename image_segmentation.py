import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
from functions import *
from skimage.segmentation import mark_boundaries

###########################################################
#                  Parameters setting                     #
###########################################################
sam_checkpoint = r"F:\Code\Segment_Anything_Model\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
# device = "cpu"
F_tb_path = r"data/Coleambally/S2_20211223.tif"
F_tb_sam_path = r"data/Coleambally/S2_20211223_sam.tif"

if __name__ == "__main__":
    image, profile = read_raster(F_tb_path)
    print(profile)

    image_pct2 = linear_pct_stretch(image, 2)
    image_pct2 = image_pct2 * 255
    image_pct2 = image_pct2.astype(np.uint8)
    image_pct2 = color_composite(image_pct2, [3, 2, 1])
    image_pct2 = image_pct2[:, :, :3]

    image = (image - image.min()) / (image.max() - image.min())
    image = image * 255
    image = image.astype(np.uint8)
    image = color_composite(image, [3, 2, 1])
    image = image[:, :, :3]

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=30
    )
    masks = mask_generator_2.generate(image)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    objects = np.full(shape=(image.shape[0], image.shape[1]), fill_value=-1, dtype=np.int32)
    for object_idx in range(len(masks)):
        mask = masks[object_idx]["segmentation"]
        objects[mask] = object_idx

        area = masks[object_idx]["area"]
        print(f"object: {object_idx}, area: {area}")

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
    boundary = mark_boundaries(image_pct2, objects)
    axes[0].imshow(image_pct2)
    axes[1].imshow(boundary)
    axes[2].imshow(objects, cmap="gray")
    axes[3].imshow(objects == -1, cmap="gray")
    axes[0].set_title("image")
    axes[1].set_title("object boundaries")
    axes[2].set_title("objects")
    axes[3].set_title("background")
    plt.show()

    objects[objects == -1] = len(masks)

    objects = np.expand_dims(objects, axis=2)
    write_raster(objects, profile, F_tb_sam_path)










