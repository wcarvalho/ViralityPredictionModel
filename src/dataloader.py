import torch
from torch.utils.data import Dataset, DataLoader
# import pandas
import csv
import random
import argparse

class TwitterDataset(Dataset):
  """docstring for TwitterDataset"""
  def __init__(self, main_csv_file, colnames, file_length=0):
    super(TwitterDataset, self).__init__()
    self.main_csv_file = main_csv_file
    self.colnames = colnames
    if not file_length:
      self.file_length = sum(1 for line in open(self.main_csv_file))
    else:
      self.file_length = file_length

  def __getitem__(self, i):
    # NOTE: best option seems to be to shuffle the csv outside of python every epoch and then read in chunks at a time
    # indexing into the csv seems very difficult
    # presumably, the CSV will be very large and we don't want to read it all into memory.

    # if self.default.data_loader == "regular": return self.regular_get(i)
    # skip = sorted(random.sample(range(self.file_length),self.file_length-128))
    # best option 
    # with open(self.main_csv_file) as f:
    #   r = csv.DictReader(f)
    #   import ipdb; ipdb.set_trace()
    # df = pandas.read_csv(self.main_csv_file, names=colnames, header=None, skiprows=skip)

  def __len__(self): return self.file_length
    
if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--filename', type=str)
  parser.add_argument('-bs', '--batch-size', type=int, default=128)
  parser.add_argument('-fl', '--file-length', type=int, default=0)
  parser.add_argument('-s', '--shuffle', type=int, default=1, choices=[0,1])
  args, unknown = parser.parse_known_args()

  batch_size = args.batch_size

  # skip = sorted(random.sample(range(file_length),file_length-batch_size))

  colnames = ["r_pid", "r_uid", "r_t", "p_pid", "p_uid", "p_t", "c_pid", "c_uid", "c_t"]
  
  dataset = TwitterDataset(args.filename, colnames)
  dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=args.shuffle,)
  for x in dataloader:
    continue
