from __future__ import print_function
from __future__ import division

import sys
import os
import glob
import numpy as np
from tqdm import trange, tqdm
import h5py

PREV_PATH ='/data/yunseokj/mining/temp_image_feature_output'
TARGET_PATH = '/data/yunseokj/mining/image_feature_output_reorg'

def main(argv):
    file_list = glob.glob(os.path.join(PREV_PATH, '*.h5'))
    for single_file in tqdm(file_list, ncols=60):

        # 1. read
        keys = []
        values = []
        f = h5py.File(single_file, 'r')
        for single_key in list(f.keys()):
            keys.append(single_key)
            target = np.array(f[single_key]['img'])
            if target.ndim == 3:
                values.append(np.squeeze(target, axis=(1,2)))
            else:
                values.append(target)

        # 2. write
        filename = os.path.basename(single_file)
        target_file = os.path.join(TARGET_PATH, filename)
        hf = h5py.File(target_file, 'w')
        for pid, value in zip(keys, values):
            grp = hf.create_group(str(pid))
            grp.create_dataset('img', data=value)
        hf.flush()

if __name__ == "__main__":
    main(sys.argv)


