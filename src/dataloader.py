# NOTE: best option seems to be to shuffle the csv outside of python every epoch and then read in chunks at a time

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import random
import argparse

class TwitterDataloader(object):
  """docstring for TwitterDataloader"""
  def __init__(self, filename, colnames, file_length, batch_size):
    super(TwitterDataloader, self).__init__()
    if not filename:
      raise RuntimeError

    self.df = pd.read_csv(filename, sep=",", names=colnames, header=None, chunksize=batch_size)

    self.batch_size = batch_size
    self.filename = filename
    self.colnames = colnames
    if not file_length:
      self.file_length = sum(1 for line in open(self.main_csv_file))
    else:
      self.file_length = file_length

  def unique_ids(self):
    df = pd.read_csv(self.filename, sep=",", names=self.colnames, header=None)
    r_pid_unique = set(df['r_pid'].unique())
    c_pid_unique = set(df['c_pid'].unique())
    return r_pid_unique.union(c_pid_unique)

  def __iter__(self):
    self.indx = 0
    return self

  def __next__(self):
    if self.indx < self.file_length:
      self.indx += self.batch_size
      return next(self.df)
    else:
      raise StopIteration

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--filename', type=str, required=True)
  parser.add_argument('-bs', '--batch-size', type=int, default=128)
  parser.add_argument('-fl', '--file-length', type=int, default=0)
  # parser.add_argument('-s', '--shuffle', type=int, default=0, choices=[0,1])
  parser.add_argument('-cn', '--colnames', type=str, default=["r_pid", "r_uid", "r_t", "p_pid", "p_uid", "p_t", "c_pid", "c_uid", "c_t", "text", "data"], nargs='+')
  args, unknown = parser.parse_known_args()

  batch_size = args.batch_size

  dataloader = TwitterDataloader(args.filename, args.colnames, args.file_length, args.batch_size)
