from __future__ import print_function
from __future__ import division

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

import sys
import re
import csv
import numpy as np
import h5py
from tqdm import trange

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
            # put special char for BERT
            value = '[CLS] ' + value[:280] + ' [SEP]'

            # tokenize
            tokenized_text = tokenizer.tokenize(value)

           # convert to index
            indexes = tokenizer.convert_tokens_to_ids(tokenized_text)

            # save it to array
            id_word_list.append([key, indexes])
            #if len(id_word_list) > 5000:
            #    break

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-multilingual-cased',
                                      cache_dir='./checkpoints/')
    # If you have a GPU, put everything on cuda
    model.to('cuda')

    # disable every gradient
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


    hf = h5py.File('../../dataset/text_features{}.h5'.format(SPLIT_NO), 'w')
    answer_dict = {}
    for idx in trange(((len(id_word_list) + BATCH_SIZE - 1) // BATCH_SIZE), ncols=60):
        batch_list = id_word_list[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        if len(batch_list) == 0:
            continue

        # group each of them
        id_list = [_[0] for _ in batch_list]
        indexed_tokens = [_[1] for _ in batch_list]

        tokens_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(_) for _ in indexed_tokens], batch_first=True)
        ## cut down the tensor
        #tokens_tensor = tokens_tensor[:, :256]
        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = torch.zeros_like(tokens_tensor)

        # remove gradient once again for a sanity check
        model.zero_grad()
        # Predict hidden states features for each layer
        with torch.no_grad():
            _, pooled_output = model.forward(input_ids=tokens_tensor,
                                     token_type_ids=segments_tensors,
                                     output_all_encoded_layers=False)

            # pooled_output: [batch_size, hidden_size (768)]
        value_list = pooled_output.data.cpu().numpy()
        for each_key, each_value in zip(id_list, value_list):
            grp = hf.create_group(str(each_key))
            grp.create_dataset('text', data=each_value)
    hf.close()


if __name__ == "__main__":
    main(sys.argv)


