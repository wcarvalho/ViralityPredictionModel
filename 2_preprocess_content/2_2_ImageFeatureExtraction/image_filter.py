from __future__ import print_function
from __future__ import division

import sys
import re
import csv
import os
import numpy as np
from tqdm import trange


IMAGE_EXT = ['png', 'jpg']
VIDEO_EXT = ['mp4', 'mpd', 'gif', 'm3u8']

def main(argv):
    new_csv_img = []
    new_csv_vid = []
    with open('./dataset/orig_link.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for target in reader:
            key = int(target[0])

            img_list = []
            vid_list = []
            # num_items = len(target)-1
            for single_item in target[1:]:
                _, ext = os.path.splitext(single_item)
                ext = ext[1:].lower()
                if ext in IMAGE_EXT:
                    img_list.append(single_item)
                else: # ext not in VIDEO_EXT):
                    vid_list.append(single_item)

            if len(vid_list) != 0:
                # take the first video as main
                new_csv_vid.append([key, 'video', vid_list[0]])
            else:
                # take the first image as main
                new_csv_img.append([key, 'image', img_list[0]])

    with open('./dataset/orig_video_link.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for line in new_csv_vid:
            writer.writerow(line)

    with open('./dataset/orig_image_link.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for line in new_csv_img:
            writer.writerow(line)


if __name__ == "__main__":
    main(sys.argv)


