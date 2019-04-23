# BERT-based Feature Embedding Extractor

## Environment Setup for BERT Extractor: same as ./setup.sh file
```
conda create --name bert -y
source activate bert
conda config --remove channels anaconda
conda config --add channels conda-forge
conda install -c conda-forge python=3.6.4 numba -y
conda install -c anaconda cudatoolkit=8.0 cudnn=7.0 -y
conda install pytorch torchvision -c pytorch -y
pip install requests gpustat tensorboardX visdom ipdb pudb tqdm boto3 h5py unidecode regex

git clone https://github.com/huggingface/pytorch-pretrained-BERT.git
cd pytorch-pretrained-BERT
source activate bert
pip install .
cd ../
source activate bert
```


## Dataset Folder setting

./dataset (current folder)

./checkpoints (current folder) : my code will automatically download checkpoints under this directory


## Specific details
batch_size = 256
max_char = 280
Take the very last hidden layer of '[CLS]'


## Execute order
bert_preprocess -> bert_with_feature -> bert_postprocess
(one liner script: run.sh)
