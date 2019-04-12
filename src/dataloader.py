# NOTE: best option seems to be to shuffle the csv outside of python every epoch and then read in chunks at a time

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import random
import argparse
import h5py

def conditional_load_h5(h5_file):
  if h5_file: return h5py.File(h5_file, 'r')
  else: return None

class TwitterDataloader(object):
  """docstring for TwitterDataloader"""
  def __init__(self, master_csv_file, label_h5_file, img_h5_file, text_h5_file, colnames, num_rows, batch_size):
    super(TwitterDataloader, self).__init__()
    if not master_csv_file:
      raise RuntimeError

    self.df = pd.read_csv(master_csv_file, sep=",", names=colnames, header=None, chunksize=batch_size)

    self.labels = conditional_load_h5(label_h5_file)
    self.images = conditional_load_h5(img_h5_file)
    self.text = conditional_load_h5(text_h5_file)

    self.batch_size = batch_size
    self.master_csv_file = master_csv_file
    self.colnames = colnames
    if not num_rows:
      self.num_rows = sum(1 for line in open(self.main_csv_file))
    else:
      self.num_rows = num_rows

    self.num_batches = self.num_rows//self.batch_size
    if self.num_rows%self.batch_size:
      self.num_batches += 1

  def unique_ids(self):
    df = pd.read_csv(self.master_csv_file, sep=",", names=self.colnames, header=None)
    r_pid_unique = set(df['r_pid'].unique())
    c_pid_unique = set(df['c_pid'].unique())
    return r_pid_unique.union(c_pid_unique)

  def __len__(self):
    return self.num_batches

  def __iter__(self):
    self.indx = 0
    return self

  def __next__(self):
    if self.indx < self.num_rows:
      self.indx += self.batch_size
      rows = next(self.df)
      outputs = [rows[col] for col in self.colnames]
      if self.label_h5_file: import ipdb; ipdb.set_trace()
      else outputs += [None, None, None, None]


      if self.img_h5_file: import ipdb; ipdb.set_trace()
      else outputs += [None]

      if self.text_h5_file: import ipdb; ipdb.set_trace()
      else outputs += [None]

    else:
      raise StopIteration

if __name__ == '__main__':
  
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)


  dataloader = TwitterDataloader(args['master_filename'], args['label_filename'], args['image_filename'], args['text_filename'], args['colnames'], args['file_length'], args['batch_size'])
