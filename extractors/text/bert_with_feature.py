from __future__ import print_function
from __future__ import division

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

import sys
import re
import glob
import os
import pickle
import numpy as np
import h5py
from tqdm import tqdm, trange

match = re.compile(r'(?:\@|https?\://)\S+|\[CLS\]|\[SEP\]|\#')



def main(argv):
    if len(argv) not in [2,3]:
        print('wrong command. It should be python bert_with_feature.py SPLIT_NO [BATCH_SIZE]')
        return

    SPLIT_NO = int(argv[1])
    if len(argv) == 3:
        BATCH_SIZE = int(argv[2])
    else:
        BATCH_SIZE = 256


    base_folder = '../../dataset/feature_text_segment{}'.format(SPLIT_NO)
    output_folder = '../../dataset/bert_text_output{}'.format(SPLIT_NO)
    os.makedirs(output_folder)
    list_of_items = glob.glob(base_folder+'/*.pickle')
    num_feature_inputs = len(list_of_items)
    list_of_items.clear()

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-multilingual-cased',
                                      cache_dir='./checkpoints/')
    # If you have a GPU, put everything on cuda
    model.to('cuda')

    # disable every gradient
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # create one large h5 file for summary
    #hf = h5py.File('../../dataset/text_features{}.h5'.format(SPLIT_NO), 'w')

    for file_id in trange(num_feature_inputs, ncols=60):
        with open(os.path.join(base_folder, '{:06d}.pickle'.format(file_id)), 'rb') as f:
            batch_list = pickle.load(f)
        if len(batch_list) == 0:
            continue

        # group each of them
        id_list = [_[0] for _ in batch_list]
        indexed_tokens = [_[2] for _ in batch_list]
        batch_list.clear()

        tokens_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(_) for _ in indexed_tokens], batch_first=True)
        indexed_tokens.clear()
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
        np.savez(os.path.join(output_folder, '{:06d}.npz'.format(file_id)), pid=np.array(id_list), value=value_list)
        id_list.clear()
        #for each_key, each_value in zip(id_list, value_list):
        #    grp = hf.create_group(str(each_key))
        #    grp.create_dataset('text', data=each_value)
    #hf.close()


if __name__ == "__main__":
    main(sys.argv)


