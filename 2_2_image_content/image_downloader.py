from __future__ import print_function
from __future__ import division

import sys
import re
import csv
import os
import numpy as np
from tqdm import tqdm
import operator
import subprocess
import shutil

#TOTAL_VID = 2935479
#TOTAL_IMG = 10837498
IMAGE_EXT = ['png', 'jpg']
CACHE_DIR = './dataset/image_cache/'
IMAGE_TEMPLATE = 'wget --quiet -O {} {}'
SPLIT_SIZE = 677500


def main(argv):
    if len(argv) != 2:
        print('wrong command. It should be python iamge_downloader.py split_no')
        return

    bucket_no = int(argv[1])
    FROM_LINE = SPLIT_SIZE * bucket_no
    TO_LINE = SPLIT_SIZE * (bucket_no + 1)

    key_link_dict = {}
    with open('./dataset/orig_image_link.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, target in enumerate(reader):
            if (idx >= TO_LINE) or (idx < FROM_LINE):
                continue
            key = int(target[0])
            img_url = target[2]
            key_link_dict[key] = img_url
    sorted_list = sorted(key_link_dict.items(), key=operator.itemgetter(0))

    item_dir = os.path.join(CACHE_DIR, '{}_{}'.format(FROM_LINE, TO_LINE))
    os.makedirs(item_dir)
    for key, single_url in tqdm(sorted_list, ncols=60):
        _, ext = os.path.splitext(single_url)

        output_path = os.path.join(item_dir, '{}{}'.format(key, ext))
        command = IMAGE_TEMPLATE.format(output_path, single_url)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()

if __name__ == "__main__":
    main(sys.argv)


