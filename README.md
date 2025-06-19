# RecSal-Net

This repository provides the PyTorch implementation of RecSal-Net, introduced in our paper:

ChaeEun Woo, SuMin Lee, Soo Min Park, and Byung Hyung Kim,
“RecSal-Net: Recursive Saliency Network for Video Saliency Prediction,” Neurocomputing, 2025.

RecSal-Net is a recursive transformer architecture that integrates a transformer-based encoder with a recursive feature integration mechanism, specifically designed for the task of Video Saliency Prediction (VSP).

# Network structure of RecSal-Net

<div align="center">
<img src="./images/Fig1.png" alt="RecSal-Net structure" width=600>

Fig.1 RecSal-Net structure
</div>

The overall architecture of RecSal-Net. (a) The RecSal-Net model, including a transformer-based encoder, recursive blocks, and a decoder. (b) The recursive block, which iteratively refines multi-scale spatiotemporal features.

# Prepare the Python virtual environment

Please create an Anaconda virtual environment by:

> $ conda create -n RS python=3.8 -y

Activate the virtual environment by:

> $ conda activate RS

Install the requirements by:

> $ pip3 install -r requirements.txt

# Run the code

Please download the pre-trained VST [here](https://github.com/Zhongdao/Video-Swin-Transformer) and the DHF1K dataset [here](https://github.com/wenguanwang/DHF1K).

<pre>
Project/
│
├── saved_models/
│   └── RecSalNet.pth
│
├── data/
│   └── DHF1K/
│       ├── train/
│       └── val/
│
├── dataloader.py
├── loss.py
├── model.py
├── swin_transformer.py
├── test.py
├── train.py
├── utils.py
├── requirements.txt
└── swin_small_patch244_window877_kinetics400_1k.pth
</pre>

You can run the code by:
> $ python3 train.py

The results will be saved in a folder named saved_models.

After you finish all the training processes, you can use test.py to generate the predicted saliency maps and compute all evaluation metrics by:
> $ python3 test.py

# Cite

Please cite our paper if you use our code in your own work:
```
```
