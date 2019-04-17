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


class TwitterDataloader(object):
  """docstring for TwitterDataloader"""
  def __init__(self,
      data_files,
      image_files,
      text_files,
      label_files,
      key,
      data_header,
      label_header,
      user_size,
      text_size,
      image_size,
      dummy_user_vector=False,
      shuffle=False,
      batch_size=1024,
      num_workers=4):
    super(TwitterDataloader, self).__init__()

    self.data_files=data_files
    self.image_files=image_files
    self.text_files=text_files
    self.label_files=label_files
    self.zipped_files = [i for i in zip(data_files, image_files, text_files, label_files)]

    self.key=key
    self.data_header=data_header
    self.label_header=label_header

    self.user_size=user_size
    self.text_size=text_size
    self.image_size=image_size

    self.dummy_user_vector=dummy_user_vector
    self.shuffle=shuffle
    self.batch_size=batch_size
    self.num_workers=num_workers

    self.file_indx = 0
    self.overall_indx = 0
    self.num_chunks = len(self.data_files)
    self.num_batches_in_chunk = None


    self.iterator = self.load_iterator(self.file_indx)
    self.max_iterator_num_batches = self.num_batches_in_chunk

  def load_iterator(self, indx):
    data_file, image_file, text_file, label_file = self.zipped_files[indx]
    twitter_dataset = TwitterDatasetChunk(
      data_file=data_file,
      image_file=image_file,
      text_file=text_file,
      label_file=label_file,
      key=self.key,
      data_header=self.data_header,
      label_header=self.label_header,
      user_size=self.user_size,
      text_size=self.text_size,
      image_size=self.image_size,
      dummy_user_vector=self.dummy_user_vector
    )

    self.chunk_batch_indx = 0
    self.num_batches_in_chunk = len(twitter_dataset)//self.batch_size + (len(twitter_dataset) % self.batch_size > 0)


    dataloader = DataLoader(twitter_dataset, batch_size=self.batch_size,
                            shuffle=self.shuffle, num_workers=self.num_workers)

    return iter(dataloader)

  def __len__(self):
    return self.num_chunks*self.max_iterator_num_batches

  def __iter__(self):
    self.file_indx = 0
    self.overall_indx = 0
    return self

  def __next__(self):
    self.overall_indx += 1
    try:
      self.chunk_batch_indx += 1

      to_return = next(self.iterator)
      return to_return
    # else, new file:
    except StopIteration as si:
      self.file_indx += 1

      if self.file_indx >= self.num_chunks:
        raise StopIteration
      else:
        file=self.data_files[self.file_indx]
        tqdm.write("%s finished" % file)
        self.iterator = self.load_iterator(file)
        return next(self.iterator)

    except Exception as e:
      raise e


def load_train_valid_test_loaders(train_files, valid_files, test_files, header_file, label_map_file, label_files, text_files, image_files, key, user_size, text_size, image_size,
 dummy_user_vector, shuffle, batch_size, num_workers):

  if not label_map_file: raise RuntimeError("need label map to find label files for corresponding master csv data")
  with open(label_map_file, 'r') as f:
    label_map = yaml.load(f, Loader=yaml.FullLoader)

  if not header_file: raise RuntimeError("need header file for master csv data")
  with open(header_file) as f:
    colnames = f.readlines()[0].strip().split(",")

  train_dataloader = TwitterDataloader(
    chunks=train_files,
    key=key,
    user_size=user_size,
    text_size=text_size,
    image_size=image_size,
    colnames=colnames,
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=dummy_user_vector,
    shuffle=shuffle,
    batch_size=batch_size,
    num_workers=num_workers)

  valid_dataloader = TwitterDataloader(
    chunks=valid_files,
    key=key,
    user_size=user_size,
    text_size=text_size,
    image_size=image_size,
    colnames=colnames,
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=dummy_user_vector,
    shuffle=False, 
    batch_size=batch_size,
    num_workers=num_workers)

  test_dataloader = TwitterDataloader(
    chunks=test_files,
    key=key,
    user_size=user_size,
    text_size=text_size,
    image_size=image_size,
    colnames=colnames,
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=dummy_user_vector,
    shuffle=False, 
    batch_size=batch_size,
    num_workers=num_workers)

  return [colnames,
          label_map,
          train_dataloader,
          valid_dataloader,
          test_dataloader]

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

  train_data_files, train_image_files, train_text_files, train_label_files = get_overlapping_data_files(train_data_files, train_image_files, train_text_files, train_label_files)

  data_loader = TwitterDataloader(
    data_files=train_data_files,
    image_files=train_image_files,
    text_files=train_text_files,
    label_files=train_label_files,
    key=key,
    data_header=data_header,
    label_header=label_header,
    user_size=user_size,
    text_size=text_size,
    image_size=image_size,
    dummy_user_vector=dummy_user_vector,
    shuffle=shuffle,
    batch_size=batch_size,
    num_workers=num_workers,
  )

  for batch in tqdm(data_loader): pass
  print("Finished going through all files...")

if __name__ == '__main__':
  main()

