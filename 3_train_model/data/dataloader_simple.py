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

from data.twitter_chunk import TwitterDatasetChunk
from data.utils import get_overlapping_data_files

def main():
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  args_to_print = {name:args[name] for name in args if not "files" in name}
  pprint(args_to_print)
  pprint(unknown)

  if args['data_header']:
    with open(args['data_header']) as f:
      data_header = f.readlines()[0].strip().split(",")
  if args['label_header']:
    with open(args['label_header']) as f:
      label_header = f.readlines()[0].strip().split(",")

  train_data_files=args['train_data_files']
  train_image_files=args['train_image_files']
  train_text_files=args['train_text_files']
  train_label_files=args['train_label_files']
  key = args['key']

  user_size = args['user_size']
  text_size = args['text_size']
  image_size = args['image_size']
  dummy_user_vector = args['dummy_user_vector']
  shuffle = args['shuffle']
  batch_size = args['batch_size']
  num_workers = args['num_workers']

  train_data_files[6:], train_image_files[6:], train_text_files[6:], train_label_files[6:] = get_overlapping_data_files(train_data_files, train_image_files, train_text_files, train_label_files)

  for train_data_file, train_image_file, train_text_file, train_label_file in tqdm(zip(train_data_files, train_image_files, train_text_files, train_label_files), desc="files"):
      dataset = TwitterDatasetChunk(
        data_file=train_data_file,
        image_file=train_image_file,
        text_file=train_text_file,
        label_file=train_label_file,
        key=key,
        data_header=data_header,
        label_header=label_header,
        user_size=user_size,
        text_size=text_size,
        image_size=image_size,
        dummy_user_vector=dummy_user_vector
        )
      dataloader = DataLoader(dataset, batch_size=args['batch_size'],
        shuffle=args['shuffle'], num_workers=args['num_workers'])

      for batch in tqdm(dataloader, desc='dataloader'): pass




if __name__ == '__main__':
  main()

