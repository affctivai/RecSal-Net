# RecSal-Net
This is the PyTorch implementation of the EEG-Deformer in our paper:

ChaeEun Woo, SuMin Lee, Soo Min Park, Byung Hyung Kim, "RecSal-Net: Recursive Saliency Network for Video Saliency Prediction", 2025.

It is a recursive transformer network that combines a transformer-based encoder with a recursive feature integration strategy for Video Saliency Prediction (VSP).

# Network structure of RecSal-Net

<div align="center">
<img src="images/Fig1.png" alt="RecSal-Net Architecture" width="600">

Fig.1 RecSal-Net structure
</div>

The overall architecture of RecSal-Net. (a) The RecSal-Net model, including the backbone, recursive blocks, and decoder. (b) The recursive block, which applies recursive processing.
