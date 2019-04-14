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

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import h5py

BATCH_SIZE = 512
BUCKET_SIZE = 512 * 100
#TARGET_PATH ='/mnt/brain4/datasets/Twitter/final/image'
TARGET_PATH = '/data/yunseokj/mining/image_feature_output'
SPLIT_SIZE = 677500
SORTED_PATH = '/data/yunseokj/mining/image_sorted'

def main(argv):
    if len(argv) != 2:
        print('wrong python image_index_reordering new_bucket_no')
    bucket_no = int(argv[1])
    bucket_folder = os.path.join(SORTED_PATH, '{}_{}'.format(SPLIT_SIZE * bucket_no, SPLIT_SIZE * (bucket_no + 1)))

    with open('/data/yunseokj/mining/orig_image{}.pickle'.format(bucket_no), 'rb') as f:
        file_list = pickle.load(f)

    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)

    resnet152 = models.resnet152(pretrained=True)
    modules = list(resnet152.children())[:-1]
    resnet152 = nn.Sequential(*modules)
    resnet152 = resnet152.to('cuda')

    for p in resnet152.parameters():
        p.requires_grad = False


    def run_inference(values):
        tensor_list = torch.cat(values, 0)
        tensor_list = tensor_list.to('cuda')

        # remove gradient once again for a sanity check
        resnet152.zero_grad()
        # Predict the fc layer value
        with torch.no_grad():
            output = resnet152.forward(tensor_list)
            # remove other dimensions
            output = torch.squeeze(torch.squeeze(output, 2), 2)
        output_list = output.data.cpu().numpy()
        return output_list

    def save_result(keys, results):
        hf = h5py.File(os.path.join(TARGET_PATH, '{}_{}.h5'.format(keys[0], keys[-1])), 'w')
        for pid, value in zip(keys, results):
            grp = hf.create_group(str(pid))
            grp.create_dataset('img', data=value)
        hf.flush()

    key_list = []
    tensor_list = []
    value_list = []
    for single_file_info in tqdm(file_list, ncols=70):
        target_path = os.path.join(bucket_folder, '{}.npy'.format(single_file_info[0]))
        if not os.path.exists(target_path):
            continue
        npy_file = np.load(target_path)
        result_img = Variable(torch.from_numpy(npy_file).unsqueeze(0))

        key_list.append(single_file_info[0])
        tensor_list.append(result_img)

        if len(tensor_list) == BATCH_SIZE:
            temp_output_list = run_inference(tensor_list)
            tensor_list.clear()
            value_list.extend(temp_output_list)

        if len(key_list) == BUCKET_SIZE:
            save_result(key_list, value_list)
            key_list.clear()
            value_list.clear()

    if len(tensor_list) != 0:
        temp_output_list = run_inference(tensor_list)
        tensor_list.clear()
        value_list.extend(temp_output_list)

    if len(key_list) != 0:
        save_result(key_list, value_list)
        key_list.clear()
        value_list.clear()

if __name__ == "__main__":
    main(sys.argv)


