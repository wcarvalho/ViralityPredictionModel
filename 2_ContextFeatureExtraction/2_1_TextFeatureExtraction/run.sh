#!/bin/bash


# install environment
bash setup.sh

# run preprocessing for the segment 0 (you can change it to 0~3, or any other number of splits)
python bert_preprocess.py 0

# run extraction code for the segment 0
CUDA_VISIBLE_DEVICES=0 python bery_with_feature.py 0

# convert the result to hdf5 file
python bert_postprocess.py
