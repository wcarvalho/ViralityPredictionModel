from __future__ import print_function
from __future__ import division

import sys
import re
import os
from tqdm import trange, tqdm
import h5py
import glob
import numpy as np
import pickle

BUCKET_BASE = 2000

def main(argv):
    word_feature_dict = {}
    write_dir = '/mnt/brain4/datasets/Twitter/final/'
    for split_no in range(4):
        base_dir = '../../dataset/bert_text_output{}'.format(split_no)
        list_of_items = glob.glob(base_dir+'/*.npz')
        num_feature_inputs = len(list_of_items)
        list_of_items.clear()

        for file_id in trange(num_feature_inputs, ncols=60):
            npz_data = np.load(os.path.join(base_dir, '{:06d}.npz'.format(file_id)))
            for pid, value in zip(npz_data['pid'], npz_data['value']):
                word_feature_dict[pid] = value

    # first save the large chunk
    with open(os.path.join(write_dir, 'user_text.pickle'), 'wb') as f:
        pickle.dump(word_feature_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    #keys = np.array(word_feature_dict.keys())
    #min_id = np.amin(keys)
    #max_id = np.amax(keys)
    #
    #bucket_min = min_id // BUCKET_BASE
    #bucket_max = max_id // BUCKET_BASE
    #
    #for bucket_id in range(bucket_min, bucket_max+1):
    #    hf = h5py.File('/mnt/brain4/datasets/Twitter/user_text_{}.h5'.format(bucket_id * BUCKET_BASE), 'w')
    #    for offset in range(BUCKET_BASE):
    #        pid = bucket_id * BUCKET_BASE + offset
    #        if pid in word_feature_dict:
    #            grp = hf.create_group(str(pid))
    #            grp.create_dataset('text', data=word_feature_dict[pid])
    #    hf.flush()


if __name__ == "__main__":
    main(sys.argv)


