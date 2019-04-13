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

class TwitterDataloader(object):
  """docstring for TwitterDataloader"""
  def __init__(self, chunks, key, colnames, label_files, label_map, text_files, image_files, dummy_user_vector=False, shuffle=False, batch_size=1024, num_workers=4):
    super(TwitterDataloader, self).__init__()

    self.chunks = chunks
    self.colnames = colnames
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
    self.num_chunks = len(self.chunks)

    self.iterator = self.load_iterator(self.chunks[self.chunk_indx])

  def load_iterator(self, chunk):
    self.twitter_dataset = TwitterDatasetChunk(chunk, self.key, self.colnames, self.label_files, self.label_map, self.text_files, self.image_files, self.dummy_user_vector)
    self.dataloader = DataLoader(self.twitter_dataset, batch_size=self.batch_size,
                            shuffle=self.shuffle, num_workers=self.num_workers)
    return iter(self.dataloader)

  def __len__(self):
    return self.num_chunks

  def __iter__(self):
    if self.shuffle: shuffle(self.chunks)
    self.chunk_indx = 0
    return self

  def __next__(self):
    try:
      return next(self.iterator)
    except StopIteration as si:
      self.chunk_indx += 1
      if self.chunk_indx >= self.num_chunks:
        raise StopIteration
      else:
        self.iterator = self.load_iterator(self.chunks[self.chunk_indx])
        return next(self.iterator)

    except Exception as e:
      raise e



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

  dataloader = TwitterDataloader(chunks=args['master_filenames'],
    colnames=colnames,
    key=args['key'],
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files, 
    dummy_user_vector=args['dummy_user_vector'],
    shuffle=False, batch_size=args['batch_size'], num_workers=4)

  for batch in dataloader: pass
