#!/bin/bash


# install environment
bash setup.sh

# create an output directory
mkdir ./results

# generate a friend network
python friend_graph.py ./data/ ./results/

# process raw data and divide into train, val, and test sets
python preprocess.py ./results/ ./data/ ./results/
