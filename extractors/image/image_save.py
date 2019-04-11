from __future__ import print_function
from __future__ import division

import torch
import sys
import os
import glob
import numpy as np
import pickle
from tqdm import trange, tqdm
from PIL import Image
BUCKET_PATH = '/home/yunseokj/mining_proj/github/dataset/image_bucket'

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import h5py

BATCH_SIZE = 512
#TARGET_PATH ='/mnt/brain4/datasets/Twitter/final/image'
TARGET_PATH = '/home/yunseokj/mining_proj/github/dataset/'

def main(argv):
    resnet152 = models.resnet152(pretrained=True)
    modules = list(resnet152.children())[:-1]
    resnet152 = nn.Sequential(*modules)
    resnet152 = resnet152.to('cuda')

    for p in resnet152.parameters():
        p.requires_grad = False

    bucket_path_list = glob.glob(BUCKET_PATH +'/*.pickle')
    bucket_path_len = len(bucket_path_list)
    bucket_path_list.clear()

    TO_BUCKET = bucket_path_len
    if len(argv) >= 2:
        FROM_BUCKET = int(argv[1])
        if len(argv) >= 3:
            TO_BUCKET = min(TO_BUCKET, int(argv[2]))
    else:
        FROM_BUCKET = 0

    for idx in range(FROM_BUCKET, TO_BUCKET):
        # save output by bucket
        with open(os.path.join(BUCKET_PATH, '{:06d}.pickle'.format(idx)), 'rb') as f:
            bucket_list = pickle.load(f)

        hf = h5py.File(os.path.join(TARGET_PATH, '{}_{}.h5'.format(bucket_list[0][0], bucket_list[-1][0])), 'w')
        for batch_idx in trange((len(bucket_list) + BATCH_SIZE - 1) // BATCH_SIZE, ncols=70):
            batch_list = bucket_list[batch_idx * BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
            tensor_list = []
            key_list = []
            for key, single_path in batch_list:
                npy_file = np.load(single_path)
                result_img = Variable(torch.from_numpy(npy_file).unsqueeze(0))
                key_list.append(key)
                tensor_list.append(result_img)
            tensor_list = torch.cat(tensor_list, 0)
            tensor_list = tensor_list.to('cuda')

            # remove gradient once again for a sanity check
            resnet152.zero_grad()
            # Predict the fc layer value
            with torch.no_grad():
                output = resnet152.forward(tensor_list)
            output_list = output.data.cpu().numpy()

            for pid, value in zip(key_list, output_list):
                grp = hf.create_group(str(pid))
                grp.create_dataset('img', data=value)
            hf.flush()
            key_list.clear()
        bucket_list.clear()

if __name__ == "__main__":
    main(sys.argv)


