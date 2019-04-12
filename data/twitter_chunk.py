import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import random
import argparse
import h5py

class TwitterDatasetChunk(Dataset):
  """docstring for TwitterDatasetChunk"""
  def __init__(self, filename, colnames, label_files, text_files, image_files):
    super(TwitterDatasetChunk, self).__init__()

    self.filename = filename
    self.colnames = colnames
    self.label_files = label_files
    self.text_files = text_files
    self.image_files = image_files

    self.df = pd.read_csv(filename, sep=",", names=colnames, header=None)
    self.colnames = colnames

  def __getitem__(self, idx):

    # import ipdb; ipdb.set_trace()
    return idx
    # import ipdb; ipdb.set_trace()
    # data = self.df.iloc[idx].as_matrix()
    # labels = None
    # images = None
    # text = None


    # NOTE: best option seems to be to shuffle the csv outside of python every epoch and then read in chunks at a time
    # indexing into the csv seems very difficult
    # presumably, the CSV will be very large and we don't want to read it all into memory.

    # if self.default.data_loader == "regular": return self.regular_get(i)
    # skip = sorted(random.sample(range(self.file_length),self.file_length-128))
    # best option 
    # with open(self.main_csv_file) as f:
    #   r = csv.DictReader(f)
    #   import ipdb; ipdb.set_trace()

  def __len__(self): return len(self.df)
