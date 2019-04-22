#!/bin/bash

conda create --name downloader -y
source activate downloader
conda config --remove channels anaconda
conda config --add channels conda-forge
conda install -c conda-forge python=3.6.4 numba ffmpeg -y
conda install -c anaconda cudatoolkit=8.0 cudnn=7.0 -y
conda install pytorch torchvision -c pytorch -y
pip install requests gpustat tensorboardX visdom ipdb pudb tqdm h5py Pillow
pip install --upgrade youtube-dl


