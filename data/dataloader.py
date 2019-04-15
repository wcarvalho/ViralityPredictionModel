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
from src.utils import get_filenames

def load_train_valid_test_loaders(all_master_files, header_file, label_map_file, split_map_file, key, user_size, dummy_user_vector, shuffle, batch_size, num_workers):

  if not header_file: raise RuntimeError("need header file for master csv data")
  with open(header_file) as f:
    colnames = f.readlines()[0].strip().split(",")

  if not label_map_file: raise RuntimeError("need label map to find label files for corresponding master csv data")
  with open(label_map_file, 'r') as f:
    label_map = yaml.load(f)

  if not split_map_file: raise RuntimeError("need map to split files into train/valid/test")
  with open(split_map_file, 'r') as f:
    split_map = yaml.load(f)

  split_map = {name: set(split_map[name]) for name in split_map}
  train_files = [f for f in all_master_files if f in split_map['train']]
  valid_files = [f for f in all_master_files if f in split_map['valid']]
  test_files = [f for f in all_master_files if f in split_map['test']]

  train_dataloader = TwitterDataloader(
    chunks=train_files,
    key=key,
    user_size=user_size,
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
          split_map,
          train_files,
          valid_files,
          test_files,
          train_dataloader,
          valid_dataloader,
          test_dataloader]

class TwitterDataloader(object):
  """docstring for TwitterDataloader"""
  def __init__(self, chunks, key, user_size, colnames, label_files, label_map, text_files, image_files, dummy_user_vector=False, shuffle=False, batch_size=1024, num_workers=4):
    super(TwitterDataloader, self).__init__()

    self.chunks = chunks
    self.colnames = colnames
    self.user_size = user_size
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
    self.chunk_batch_indx = 0
    self.num_chunks = len(self.chunks)
    self.num_batches_in_chunk = None

    self.iterator = self.load_iterator(self.chunks[self.chunk_indx])

  def load_iterator(self, chunk):
    self.twitter_dataset = TwitterDatasetChunk(
      filename=chunk,
      key=self.key,
      colnames=self.colnames,
      user_size=self.user_size,
      label_files=self.label_files,
      label_map=self.label_map,
      text_files=self.text_files,
      image_files=self.image_files,
      dummy_user_vector=self.dummy_user_vector)
    self.num_batches_in_chunk = len(self.twitter_dataset)//self.batch_size + (len(self.twitter_dataset) % self.batch_size > 0)
    self.dataloader = DataLoader(self.twitter_dataset, batch_size=self.batch_size,
                            shuffle=self.shuffle, num_workers=self.num_workers)
    return iter(self.dataloader)

  def __len__(self):
    return self.num_chunks

  def __iter__(self):
    if self.shuffle: 
      shuffle(self.chunks)
    self.chunk_indx = 0
    return self

  def __next__(self):
    if self.chunk_batch_indx < self.num_batches_in_chunk:
      self.chunk_batch_indx += 1
      tqdm.write("\t%d/%d" % (self.chunk_batch_indx, len(self.twitter_dataset)))
      return next(self.iterator)
    else:
      self.chunk_indx += 1
      tqdm.write("file %d" % self.chunk_indx)
      if self.chunk_indx >= self.num_chunks:
        raise StopIteration
      else:
        self.iterator = self.load_iterator(self.chunks[self.chunk_indx])
        self.chunk_batch_indx = 0
        return next(self.iterator)

    # except Exception as e:
    #   raise e

if __name__ == '__main__':
  
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  if args['header']:
    with open(args['header']) as f:
      colnames = f.readlines()[0].strip().split(",")

  label_files = args['label_filenames']
  text_files = args['text_filenames']
  image_files = args['image_filenames']

  if not args['label_map']: raise RuntimeError("need map to find label files")
  with open(args['label_map'], 'r') as f:
    label_map = yaml.load(f)

  colnames, label_map, split_map, train_files, _, _, train_dataloader, _, _ = load_train_valid_test_loaders(
      all_master_files=args['master_filenames'],
      header_file=args['header'],
      label_map_file=args['label_map'],
      split_map_file=args['split_map'],
      key=args['key'],
      user_size=args['user_size'],
      dummy_user_vector=args['dummy_user_vector'],
      shuffle=args['shuffle'],
      batch_size=args['batch_size'],
      num_workers=args['num_workers'])

  for batch in tqdm(train_dataloader): pass
  print("Finished going through all files...")
  import ipdb; ipdb.set_trace()