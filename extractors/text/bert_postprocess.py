from __future__ import print_function
from __future__ import division

import sys
import re
import os
from tqdm import trange, tqdm
import h5py
import glob
import numpy as np
import operator

BUCKET_BASE = 50000

def main(argv):
    word_feature_dict = {}
    write_dir = '/mnt/brain4/datasets/Twitter/final/text'
    for split_no in range(4):
        base_dir = '../../dataset/bert_text_output{}'.format(split_no)
        list_of_items = glob.glob(base_dir+'/*.npz')
        num_feature_inputs = len(list_of_items)
        list_of_items.clear()

        for file_id in trange(num_feature_inputs, ncols=60):
            npz_data = np.load(os.path.join(base_dir, '{:06d}.npz'.format(file_id)))
            for pid, value in zip(npz_data['pid'], npz_data['value']):
                word_feature_dict[pid] = value
    # sort first
    sorted_list = sorted(word_feature_dict.items(), key=operator.itemgetter(0))

    # save output by bucket
    for idx in trange((len(sorted_list) + BUCKET_BASE - 1) // BUCKET_BASE, ncols=60):
        batch_list = sorted_list[idx * BUCKET_BASE:(idx+1) * BUCKET_BASE]
        if len(batch_list) == 0:
            continue
        hf = h5py.File(os.path.join(write_dir, '{}_{}.h5'.format(batch_list[0][0], batch_list[-1][0])), 'w')
        for pid, vlaue in batch_list:
            grp = hf.create_group(str(pid))
            grp.create_dataset('text', data=value)
        hf.flush()


if __name__ == "__main__":
    main(sys.argv)


