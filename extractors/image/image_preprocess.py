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
BASE_PATH = '/home/yunseokj/mining_proj/github/dataset/image_cache'
NPY_PATH = '/home/yunseokj/mining_proj/github/dataset/image_rescale'
import torchvision.transforms as transforms

def main(argv):
    start = int(argv[1]) * 4
    end = start + 4

    resize = transforms.Resize(256)
    center_crop = transforms.CenterCrop(224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    for split_no in range(start, end):
        split_folder = '{}_{}'.format(SPLIT_SIZE * split_no, SPLIT_SIZE * (split_no + 1))
        prev_set = set()
        save_folder = os.path.join(NPY_PATH, split_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # load prev files first
        file_list = glob.glob(os.path.join(save_folder, '*.npy'))
        for single_path in tqdm(file_list, ncols=70):
            _, filename = os.path.split(single_path)
            basename, _ = os.path.splitext(filename)
            basename = int(basename)
            prev_set.add(basename)
        file_list.clear()

        # update img that is not processed before
        file_list = glob.glob(os.path.join(BASE_PATH, split_folder, '*.*'))
        for single_path in tqdm(file_list, ncols=70):
            _, filename = os.path.split(single_path)
            basename, _ = os.path.splitext(filename)
            basename = int(basename)
            if basename in prev_set:
                continue

            try:
                im = Image.open(single_path)
                if im is None:
                    continue
            except IOError:
                continue
            except Exception:
                continue
            im = im.convert('RGB')
            result_img = normalize(to_tensor(center_crop(resize(im)))).data.cpu().numpy()
            np.save(os.path.join(save_folder, '{:06d}.npy'.format(basename)), result_img)
        file_list.clear()


if __name__ == "__main__":
    main(sys.argv)

