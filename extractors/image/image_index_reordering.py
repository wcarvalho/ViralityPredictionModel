from __future__ import print_function
from __future__ import division

import sys
import os
from tqdm import trange, tqdm
import operator
import csv
from PIL import Image
import pickle

SPLIT_SIZE = 677500
BUCKET_BASE = 50000
ORIG_PATH = '/data/yunseokj/mining/image_cache'
RESCALED_PATH = '/data/yunseokj/mining/image_rescale'

def main(argv):
    #if len(argv) != 2:
    #    print('wrong python image_index_reordering new_bucket_no')

    #target_validator_id = int(argv[1])

    key_bucketno_dict = {}
    with open('/data/yunseokj/mining/orig_image_link.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, target in enumerate(tqdm(reader, total=677500*16)):
            key = int(target[0])
            _, ext = os.path.splitext(target[2])
            key_bucketno_dict[key] = [idx // 677500, ext]

    sorted_list = sorted(key_bucketno_dict.items(), key=operator.itemgetter(0))

    new_bucket_id = -1
    sorted_key_imgpath = []
    for new_key_id, (post_id, cache_bucket_item) in enumerate(tqdm(sorted_list)):
        if new_key_id % SPLIT_SIZE == 0:
            new_bucket_id = new_key_id // SPLIT_SIZE
            if len(sorted_key_imgpath) != 0:
                with open('/data/yunseokj/mining/orig_image{}.pickle'.format(new_bucket_id - 1), 'wb') as f:
                    pickle.dump(sorted_key_imgpath, f, protocol=pickle.HIGHEST_PROTOCOL)

            sorted_key_imgpath.clear()

#        if new_bucket_id != target_validator_id:
#            continue
        cache_bucket_num, cache_bucket_ext = cache_bucket_item
        img_path = os.path.join(ORIG_PATH,
                                '{}_{}'.format(SPLIT_SIZE * cache_bucket_num,
                                               SPLIT_SIZE * (cache_bucket_num + 1)),
                                '{}{}'.format(post_id, cache_bucket_ext))
        if not os.path.exists(img_path):
            continue

        rescale_path = os.path.join(RESCALED_PATH,
                                    '{}_{}'.format(SPLIT_SIZE * cache_bucket_num,
                                                   SPLIT_SIZE * (cache_bucket_num + 1)),
                                    '{}.npy'.format(post_id))

        if not os.path.exists(rescale_path):
            #try:
            #    im = Image.open(img_path[0])
            #    if im is None:
            #        continue
            #except IOError:
            #    continue
            #except Exception:
            #    continue
            target_path = img_path
        else:
            #target_path = os.path.join(new_bucket_path, '{}.npy'.format(post_id))
            #shutil.move(rescale_path, target_path)
            target_path = rescale_path

        # now I know that image exists and readable
        sorted_key_imgpath.append([post_id, new_bucket_id, target_path])

    if len(sorted_key_imgpath) != 0:
        with open('/data/yunseokj/mining/orig_image{}.pickle'.format(new_bucket_id), 'wb') as f:
            pickle.dump(sorted_key_imgpath, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main(sys.argv)

