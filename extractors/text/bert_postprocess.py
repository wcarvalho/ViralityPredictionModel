from __future__ import print_function
from __future__ import division

import sys
import re
import os
from tqdm import trange, tqdm
import h5py
import glob
import numpy as np

def main(argv):
    hf = h5py.File('/mnt/brain4/datasets/Twitter/user_text.h5', 'w')
    for split_no in range(4):
        base_dir = '../../dataset/bert_text_output{}'.format(split_no)
        list_of_items = glob.glob(base_dir+'/*.npz')
        num_feature_inputs = len(list_of_items)
        list_of_items.clear()

        for file_id in trange(num_feature_inputs, ncols=60):
            npz_data = np.load(os.path.join(base_dir, '{:06d}.npz'.format(file_id)))
            for pid, value in zip(npz_data['pid'], npz_data['value']):
                grp = hf.create_group(str(pid))
                grp.create_dataset('text', data=value)
            hf.flush()

if __name__ == "__main__":
    main(sys.argv)


