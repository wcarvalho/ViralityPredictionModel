from __future__ import print_function
from __future__ import division

import torch
import sys
import os
import numpy as np
from tqdm import trange, tqdm
import glob
import operator
import pickle
from PIL import Image
SPLIT_SIZE = 677500
BUCKET_BASE = 50000
BASE_PATH = '/home/yunseokj/mining_proj/github/dataset/image_cache'
BUCKET_PATH = '/home/yunseokj/mining_proj/github/dataset/image_bucket'

def main(argv):
    BATCH_SIZE = 64
    key_path_dict = {}
    for split_no in range(16):
        split_folder = '{}_{}'.format(SPLIT_SIZE * split_no, SPLIT_SIZE * (split_no + 1))
        file_list = glob.glob(os.path.join(BASE_PATH, split_folder, '*.*'))
        key_value_dict = {}
        for single_path in tqdm(file_list, ncols=70):
            try:
                im = Image.open(single_path)
                if im is None:
                    continue
            except IOError:
                continue
            except Exception:
                continue
            _, filename = os.path.split(single_path)
            basename, _ = os.path.splitext(filename)
            basename = int(basename)
            key_path_dict[basename] = single_path

            # FOR DEBUGGING
            if len(key_path_dict) >= BUCKET_BASE:
                break

        file_list.clear()

        # FOR DEBUGGING
        if len(key_path_dict) >= BUCKET_BASE:
            break

    if not os.path.exists(BUCKET_PATH):
        os.makedirs(BUCKET_PATH)

    sorted_list = sorted(key_path_dict.items(), key=operator.itemgetter(0))
    # save output by bucket
    for idx in trange((len(sorted_list) + BUCKET_BASE - 1) // BUCKET_BASE, ncols=70):
        bucket_list = sorted_list[idx * BUCKET_BASE:(idx+1) * BUCKET_BASE]
        if len(bucket_list) == 0:
            continue
        with open(os.path.join(BUCKET_PATH, '{:06d}.pickle'.format(idx)), 'wb') as f:
            pickle.dump(bucket_list, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main(sys.argv)


