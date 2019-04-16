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


class TwitterDataloader(object):
  """docstring for TwitterDataloader"""
  def __init__(self, chunks, key, user_size, text_size, image_size, colnames, label_files, label_map, text_files, image_files, dummy_user_vector=False, shuffle=False, batch_size=1024, num_workers=4):
    super(TwitterDataloader, self).__init__()

    self.chunks = chunks
    self.colnames = colnames

    self.user_size = user_size
    self.text_size = text_size
    self.image_size = image_size

    self.label_files = label_files
    self.label_map = label_map
    self.text_files = text_files
    self.image_files = image_files

    self.dummy_user_vector = dummy_user_vector
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.key = key
    self.num_workers = num_workers

    self.chunk_indx = 0
    self.overall_indx = 0
    self.num_chunks = len(self.chunks)
    self.num_batches_in_chunk = None

    self.twitter_dataset = None

    self.iterator = self.load_iterator(self.chunks[self.chunk_indx])
    self.max_iterator_num_batches = self.num_batches_in_chunk

  def load_iterator(self, chunk):
    try:
      self.data_df = pd.read_csv(chunk, sep=",", names=self.colnames, header=None)
    except Exception as e:
      print(e)
      import ipdb; ipdb.set_trace()
      raise e
    self.twitter_dataset = TwitterDatasetChunk(
      df=self.data_df,
      key=self.key,
      colnames=self.colnames,
      user_size=self.user_size,
      text_size=self.text_size,
      image_size=self.image_size,
      label_files=self.label_files,
      label_map=self.label_map,
      text_files=self.text_files,
      image_files=self.image_files,
      dummy_user_vector=self.dummy_user_vector
      )

    self.chunk_batch_indx = 0
    self.num_batches_in_chunk = len(self.twitter_dataset)//self.batch_size + (len(self.twitter_dataset) % self.batch_size > 0)


    self.dataloader = DataLoader(self.twitter_dataset, batch_size=self.batch_size,
                            shuffle=self.shuffle, num_workers=self.num_workers)

    return iter(self.dataloader)

  def __len__(self):
    return self.num_chunks*self.max_iterator_num_batches

  def __iter__(self):
    self.chunk_indx = 0
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
      self.chunk_indx += 1

      if self.chunk_indx >= self.num_chunks:
        raise StopIteration
      else:
        file=self.chunks[self.chunk_indx]
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
  else:
    train_files = args['train_filenames']
    valid_files = args['valid_filenames']
    test_files = args['test_filenames']

  text_files = []
  image_files = []
  label_files = []
  text_files, image_files, label_files = open_data_files(args['text_filenames'], args['image_filenames'], args['label_filenames'])

  colnames, label_map, train_dataloader, _, _ = load_train_valid_test_loaders(
      train_files=train_files,
      valid_files=valid_files,
      test_files=test_files,
      header_file=args['header'],
      label_map_file=args['label_map'],
      label_files = label_files,
      text_files = text_files,
      image_files = image_files,
      key=args['key'],
      user_size=args['user_size'],
      text_size=args['text_size'],
      image_size=args['image_size'],
      dummy_user_vector=args['dummy_user_vector'],
      shuffle=args['shuffle'],
      batch_size=args['batch_size'],
      num_workers=args['num_workers']
      )

  for batch in tqdm(train_dataloader): pass
  print("Finished going through all files...")

if __name__ == '__main__':
  main()
  # close_h5py_filelist([f for f in train_files])
  # close_h5py_filelist([f for f in valid_files])
  # close_h5py_filelist([f for f in test_files])

