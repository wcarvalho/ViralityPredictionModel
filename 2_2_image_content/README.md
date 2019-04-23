# Visual Information Extractor

## Environment Setup for Image/Video Extractor
```
conda create --name downloader -y
source activate downloader
conda config --remove channels anaconda
conda config --add channels conda-forge
conda install -c conda-forge python=3.6.4 numba ffmpeg -y
conda install -c anaconda cudatoolkit=8.0 cudnn=7.0 -y
conda install pytorch torchvision -c pytorch -y
pip install requests gpustat tensorboardX visdom ipdb pudb tqdm h5py Pillow
pip install --upgrade youtube-dl
```
Same command set can be found in setup.sh


## Dataset Folder setting
./dataset (current folder)

./resources (current folder)



## Specific details

For downloading video, I use 'youtube-dl' which downloads file to the current folder only.

So, for downloading video, I need to create a subfolder to execute it.

I run the video/image downloader with 16 cores



## Steps

image_filter -> image_downloader -> image_preprocess -> image_resnet
( -> video_downloader -> ... after image_filter)

One-liner script is at run.sh
