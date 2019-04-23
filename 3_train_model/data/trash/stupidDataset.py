import linecache
from pprint import pprint
import os
import yaml
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import random
from tqdm import tqdm
import argparse
import h5py

from data.twitter_chunk import TwitterDatasetChunk, close_h5py_filelist, open_data_files
from src.utils import get_filenames


class StupidTwitterDataset(object):
  """docstring for StupidTwitterDataset"""
  def __init__(self, main_file, main_file_length, key, user_size, text_size, image_size, colnames, label_files, label_map, text_files, image_files, dummy_user_vector=False, shuffle=False, batch_size=1024, num_workers=4):
    super(StupidTwitterDataset, self).__init__()

    self.main_file = main_file
    self.main_file_length = main_file_length

    self.colnames = colnames
    self.label_map = label_map
    self.key = key

    self.user_size = user_size
    self.text_size = text_size
    self.image_size = image_size


    self.label_files = label_files
    self.text_files = text_files
    self.image_files = image_files

    self.dummy_user_vector = dummy_user_vector

  def __getitem(self, idx):

    line = linecache.getline(self.main_file, idx)
    csv_line = csv.reader([line])
    import ipdb; ipdb.set_trace()

  def __len__(self):
    return self.main_file_length
    # self.shuffle = shuffle
    # self.batch_size = batch_size
    # self.num_workers = num_workers

    # self.chunk_indx = 0
    # self.overall_indx = 0
    # self.num_chunks = len(self.chunks)
    # self.num_batches_in_chunk = None

    # self.twitter_dataset = None

    # self.iterator = self.load_iterator(self.chunks[self.chunk_indx])
    # self.max_iterator_num_batches = self.num_batches_in_chunk



def main():
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  args_to_print = {name:args[name] for name in args if not "filenames" in name}
  pprint(args_to_print)
  pprint(unknown)

  if args['split_map'] and args['master_filenames']:
    with open(args['split_map'], 'r') as f:
      split_map = yaml.load(f, Loader=yaml.FullLoader)
      split_map = {name: set(split_map[name]) for name in split_map}

      train_files = [f for f in args['master_filenames'] if os.path.basename(f) in split_map['train']]
      valid_files = [f for f in args['master_filenames'] if os.path.basename(f) in split_map['valid']]
      test_files = [f for f in args['master_filenames'] if os.path.basename(f) in split_map['test']]
  elif args['master_filenames'] and not args['split_map']:
    raise RuntimeError("need map to split master-filenames into train/valid/test")
  elif args['train_length'] and args['train_filenames'] > 0:
    raise RuntimeError("Only support file length when have one big csv. If use many smaller, than read them all into memory")
  else:
    train_files = args['train_filenames']
    valid_files = args['valid_filenames']
    test_files = args['test_filenames']

  if not header_file: raise RuntimeError("need header file for master csv data")
  with open(header_file) as f:
    colnames = f.readlines()[0].strip().split(",")

  text_files, image_files, label_files = open_data_files(args['text_files'], args['image_files'], args['label_files'])

  import ipdb; ipdb.set_trace()
  twitter_dataset = StupidTwitterDataset(
    main_file=train_files[0],
    main_file=args['train_length'],
    key=args['key'],
    colnames=colnames,
    user_size=args['user_size'],
    text_size=args['text_size'],
    image_size=args['image_size'],
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=args['dummy_user_vector']
    )

  train_dataloader = DataLoader(twitter_dataset, batch_size=args['batch_size'],
      shuffle=args['shuffle'], num_workers=args['num_workers'])

  for batch in tqdm(train_dataloader):
    pass
  #   (main_file=train_files[0], key, user_size, text_size, image_size, colnames, label_files, label_map, text_files, image_files, dummy_user_vector=False, shuffle=False, batch_size=1024, num_workers=4):

  # text_files, image_files, label_files = open_data_files(args['text_filenames'], args['image_filenames'], args['label_filenames'])

  # colnames, label_map, train_dataloader, _, _ = load_train_valid_test_loaders(
  #     train_files=train_files,
  #     valid_files=valid_files,
  #     test_files=test_files,
  #     header_file=args['header'],
  #     label_map_file=args['label_map'],
  #     label_files = label_files,
  #     text_files = text_files,
  #     image_files = image_files,
  #     key=args['key'],
  #     user_size=args['user_size'],
  #     text_size=args['text_size'],
  #     image_size=args['image_size'],
  #     dummy_user_vector=args['dummy_user_vector'],
  #     shuffle=args['shuffle'],
  #     batch_size=args['batch_size'],
  #     num_workers=args['num_workers']
  #     )

  # for batch in tqdm(train_dataloader): pass
  # print("Finished going through all files...")

  # close_h5py_filelist([f for f in train_files])
  # close_h5py_filelist([f for f in valid_files])
  # close_h5py_filelist([f for f in test_files])

  # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
  main()