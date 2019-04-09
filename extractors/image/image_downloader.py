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
CACHE_DIR = '/data/yunseokj/mining/image_cache/'
IMAGE_TEMPLATE = 'wget --quiet -O {} {}'

def main(argv):
    if len(argv) != 3:
        print('wrong command. It should be python bery.py FROM_LINE NUM_LINE')
        return

    FROM_LINE = int(argv[1])
    NUM_LINE = int(argv[2])

    key_link_dict = {}
    with open('../../dataset/orig_image_link.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, target in enumerate(reader):
            if idx >= FROM_LINE + NUM_LINE or idx < FROM_LINE:
                continue
            key = int(target[0])
            img_url = target[2]
            key_link_dict[key] = img_url
    sorted_list = sorted(key_link_dict.items(), key=operator.itemgetter(0))

    item_dir = os.path.join(CACHE_DIR, '{}_{}'.format(FROM_LINE, NUM_LINE))
    os.makedirs(item_dir)
    for key, single_url in tqdm(sorted_list):
        _, ext = os.path.splitext(single_url)

        output_path = os.path.join(item_dir, '{}{}'.format(key, ext))
        command = IMAGE_TEMPLATE.format(output_path, single_url)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()

if __name__ == "__main__":
    main(sys.argv)


