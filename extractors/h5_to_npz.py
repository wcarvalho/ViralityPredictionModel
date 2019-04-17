import h5py
import numpy as np
import os
import glob
from tqdm import tqdm

BASE_PATH = '/data/yunseokj/mining/image_feature_output'
TARGET_FOLDER = '/data/yunseokj/mining/image_feature_npz/{}.npz'
KEY_NAME = 'img'

#BASE_PATH = '/data/wcarvalh/twitter/features_from_yunseok/text'
#TARGET_FOLDER = '/data/wcarvalh/twitter/features_from_yunseok/text_npz/{}.npz'
#KEY_NAME = 'text'

files = glob.glob(os.path.join(BASE_PATH, '*.h5'))

for single_file in tqdm(files, ncols=60):
    base_name = os.path.basename(single_file)
    file_name, _ = os.path.splitext(base_name)
    target_file_path = TARGET_FOLDER.format(file_name)

    new_dict = {}
    f = h5py.File(single_file, 'r', swmr=True)
    for single_key in list(f.keys()):
        new_dict[int(single_key)] = np.array(f[single_key][KEY_NAME])
    np.savez(target_file_path, new_dict)
    new_dict.clear()
