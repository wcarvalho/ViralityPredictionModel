#!/bin/bash

set -x # Below this line, each command will be printed to terminal before being executed.

# setup new environment for image
bash setup.sh

# first filter the link data into image and video
python image_filter.py

# download first segment of the entire image
python image_downloader.py 0

# convert image into same format
python image_preprocess.py 0

# run ResNet to extract image
CUDA_VISIBLE_DEVICES=0 python image_resnet.py
