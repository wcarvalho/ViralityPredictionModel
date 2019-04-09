from __future__ import print_function
from __future__ import division

import torch
from pytorch_pretrained_bert import BertTokenizer

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

import sys
import re
import csv
import pickle
from tqdm import trange
from unidecode import unidecode


match = re.compile(r'(?:\@|https?\://)\S+|\[CLS\]|\[SEP\]|\#')



def main(argv):
    if len(argv) not in [2,3]:
        print('wrong command. It should be python bery.py SPLIT_NO [BATCH_SIZE]')
        return

    SPLIT_NO = int(argv[1])
    if len(argv) == 3:
        BATCH_SIZE = int(argv[2])
    else:
        BATCH_SIZE = 256


    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                              do_lower_case=False,
                                              max_len=1024,
                                              do_basic_tokenize=True,
                                              cache_dir='./checkpoints/')
    id_word_list = []
    with open('../../dataset/orig_text_segment{}'.format(SPLIT_NO)) as f:
        reader = csv.reader(f, delimiter=',')
        for target in reader:
            key = int(target[0])
            value = ','.join(target[1:])

            # remove url, mention, hashtag, special token used in BERT
            value = match.sub("", value)
            # remove non-ascii stuffs
            value = unidecode(value)
            # put special char for BERT
            value = '[CLS] ' + value[:150] + ' [SEP]'

            # tokenize
            tokenized_text = tokenizer.tokenize(value)

           # convert to index
            indexes = tokenizer.convert_tokens_to_ids(tokenized_text)

            # save it to array
            id_word_list.append([key, len(indexes), indexes])
            #if len(id_word_list) > 5000:
            #    break

    with open('../../dataset/feature_text_segment{}_unidecode.pickle'.format(SPLIT_NO), 'wb') as f:
        pickle.dump(id_word_list, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(sys.argv)


