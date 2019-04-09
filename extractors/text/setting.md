# BERT-based tokenizer

## Environment Setup for BERT Extractor
```
conda create --name bert -y
source activate bert
conda config --remove channels anaconda
conda config --add channels conda-forge
conda install -c conda-forge python=3.6.4 numba -y
conda install -c anaconda cudatoolkit=8.0 cudnn=7.0 -y
conda install pytorch torchvision -c pytorch -y
pip install requests gpustat tensorboardX visdom ipdb pudb tqdm boto3 h5py regex

git clone https://github.com/huggingface/pytorch-pretrained-BERT.git
cd pytorch-pretrained-BERT
source activate bert
pip install .
cd ../
source activate bert
```

## Dataset Folder setting
../../dataset (root folder)
./checkpoints (current folder)
