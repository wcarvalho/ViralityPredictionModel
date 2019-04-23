from __future__ import print_function
from __future__ import division

import torch
import sys
import os
import glob
import numpy as np
from tqdm import trange, tqdm
import operator
import pickle
import shutil
from PIL import Image
SPLIT_SIZE = 677500
SORTED_PATH = './dataset/image_sorted'
CACHE_DIR = './dataset/image_cache'
import torchvision.transforms as transforms

def main(argv):
    if len(argv) != 2:
        print('wrong python image_preprocess.py new_bucket_no')
    bucket_no = int(argv[1])

    FROM_LINE = SPLIT_SIZE * bucket_no
    TO_LINE = SPLIT_SIZE * (bucket_no + 1)
    file_list = glob.glob(os.path.join(CACHE_DIR, '{}_{}/*.*'.format(FROM_LINE, TO_LINE)))

    resize = transforms.Resize(256)
    center_crop = transforms.CenterCrop(224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    bucket_folder = os.path.join(SORTED_PATH, '{}_{}'.format(FROM_LINE, TO_LINE))
    if not os.path.exists(bucket_folder):
        os.makedirs(bucket_folder)

    for single_file_info in tqdm(file_list, ncols=70):
        key_with_ext = os.path.basename(single_file_info)
        key, _ = os.path.splitext(key_with_ext)
        target_path = os.path.join(bucket_folder, '{}.npy'.format(key))
        try:
            im = Image.open(single_file_info)
            if im is None:
                continue
        except IOError:
            continue
        except Exception:
            continue
        if im.mode == 'P':
            im = im.convert('RGBA')

        if im.mode in ('RGBA', 'LA'):
            background = Image.new(im.mode[:-1], im.size, (255, 255, 255))
            background.paste(im, im.split()[-1])
            im = background
            im = im.convert('RGB')
        else:
            im = im.convert('RGB')
        result_img = normalize(to_tensor(center_crop(resize(im)))).data.cpu().numpy()
        np.save(target_path, result_img)


if __name__ == "__main__":
    main(sys.argv)

