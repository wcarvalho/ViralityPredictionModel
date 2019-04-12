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

def conditional_load_h5(h5_file):
  if h5_file: return h5py.File(h5_file, 'r')
  else: return None

class TwitterDataloader(object):
  """docstring for TwitterDataloader"""
  def __init__(self, chunks, colnames, label_files, text_files, image_files, shuffle=False, batch_size=1024, num_workers=4):
    super(TwitterDataloader, self).__init__()

    self.chunks = chunks
    self.colnames = colnames
    self.label_files = label_files
    self.text_files = text_files
    self.image_files = image_files
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.num_workers = num_workers

    self.chunk_indx = 0
    self.num_chunks = len(self.chunks)

    self.iterator = self.load_iterator(self.chunks[self.chunk_indx])

  def load_iterator(self, chunk):
    self.twitter_dataset = TwitterDatasetChunk(chunk, self.colnames, self.label_files, self.text_files, self.image_files)
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

  colnames = args['colnames']
  if args['header']:
    with open(args['header']) as f:
      colnames = f.readlines()[0].strip().split(",")

  dataloader = TwitterDataloader(chunks=args['master_filenames'],
    colnames=colnames,
    label_files=args['label_filenames'],
    text_files=args['text_filenames'],
    image_files=args['image_filenames'], 
    shuffle=False, batch_size=args['batch_size'], num_workers=4)

  for generator in tqdm(dataloader): pass
  for generator in tqdm(dataloader): break

