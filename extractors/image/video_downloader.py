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
#IMAGE_EXT = ['png', 'jpg']
VIDEO_EXT = ['mp4', 'mpd', 'gif', 'm3u8']
SUPPORT_VID_FORM = ['gif', 'flv', 'm4a', 'mp4', 'ogg', 'webm']
CACHE_DIR = './'
VIDEO_TEMPLATE = 'youtube-dl {}'


def main(argv):
    if len(argv) != 3:
        print('wrong command. It should be python bery.py FROM_LINE NUM_LINE')
        return

    FROM_LINE = int(argv[1])
    TO_LINE = int(argv[2])

    key_link_dict = {}
    with open('../../orig_video_link.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, target in enumerate(reader):
            if (idx >= TO_LINE) or (idx < FROM_LINE):
                continue
            key = int(target[0])
            vid_url = target[2]
            key_link_dict[key] = vid_url
    sorted_list = sorted(key_link_dict.items(), key=operator.itemgetter(0))

    for key, single_url in tqdm(sorted_list, ncols=60):
        command = VIDEO_TEMPLATE.format(single_url)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        #print(key, single_url, process.returncode)

        moving_target = []
        for ext in SUPPORT_VID_FORM:
            filename = subprocess.Popen("ls -1 *.{}".format(ext), shell=True, stdout=subprocess.PIPE)
            result_file = filename.stdout.read().strip()
            if len(result_file) > 3:
                moving_target.append(result_file.decode('ascii'))

        if len(moving_target) > 0:
            item_dir = os.path.join(CACHE_DIR, '{}_{}'.format(FROM_LINE, TO_LINE), str(key))
            os.makedirs(item_dir)
            for single_item in moving_target:
                shutil.move(single_item, os.path.join(item_dir, single_item))


if __name__ == "__main__":
    main(sys.argv)


