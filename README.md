# OBSUM-code
<p align = "justify" style="line-height:150%">Python code of our paper "OBSUM: An object-based spatial unmixing model for spatiotemporal fusion of remote sensing images" (https://www.sciencedirect.com/science/article/pii/S0034425724000579#ab0010)<br></p>
<p align = "justify" style="line-height:150%">OBSUM can fuse images with different spatial and temporal resolutions, e.g., Sentinel-2 and Sentinel-3, to generate images with both high spatial and temporal resolutions, i.e., daily Sentinel-2-like images.<br></p>
<p align = "justify" style="line-height:150%">If you found our paper useful, please kindly cite: Guo, H., Ye, D., Xu, H., & Bruzzone, L. (2024). OBSUM: An object-based spatial unmixing model for spatiotemporal fusion of remote sensing images. Remote Sensing of Environment, 304, 114046. https://doi.org/https://doi.org/10.1016/j.rse.2024.114046<br></p>
<p align = "justify" style="line-height:150%">Please feel free to contact with me if you have any trouble in running the OBSUM code: (houcai.guo@unitn.it)<br></p>

## Image segmentation
<p align = "justify" style="line-height:150%">Please visit https://github.com/facebookresearch/segment-anything for detailed guidance on installing the Segment Anything Model (SAM) and downloading the model checkpoint.<br></p>
<p align = "justify" style="line-height:150%">If you prefer the Multiresolution Segmentation in the eCognition software, please use the raw image instead of the surface reflectance for segmentation.<br></p>

## Experimental data
**Dataset** (Fig. 4 and 5 in the paper, 2.17 GB): https://pan.baidu.com/s/1GZ9t628ncmFGz3D4aVfvWg?pwd=0416

**All fused images** (Fig. 11 and 12 in the paper, 4.61 GB): https://pan.baidu.com/s/1_CNq-xOxwdmTW3CNssHffw?pwd=0416

**Note**: Please visit https://www.lfd.uci.edu/~gohlke/pythonlibs/ to download the .whl file for GDAL and Rasterio (if needed).

