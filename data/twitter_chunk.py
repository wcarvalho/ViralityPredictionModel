import time
import yaml
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
from tqdm import tqdm
import random
import argparse
import h5py
import numpy as np

from src.utils import get_filenames

def close_h5py_filelist(file_list):
  for file in file_list:
    file.close()

def open_files(file_list, key_fn, opener, name):
  return [(key_fn(file), opener(file)) for file in tqdm(file_list, desc="opening %s files" % name)]

def open_data_files(text_files, image_files, label_files):
  h5_filename = lambda x:  os.path.splitext(os.path.basename(x))[0]
  text_files = open_files(text_files, h5_filename, lambda x: h5py.File(x, 'r',  libver='latest', swmr=True), "h5py text")
  image_files = open_files(image_files, h5_filename, lambda x: h5py.File(x, 'r',  libver='latest', swmr=True), "h5py image")
  label_files = open_files(label_files, lambda x: os.path.basename(x), lambda x: pd.read_csv(x, sep=",", names=['root_postID','tree_size','max_depth','avg_depth'], header=None), "label")
  return text_files, image_files, label_files

def binary_search(files, split_func, pid):
  if not len(files): return None, None

  mid = len(files)//2
  start, end = split_func(files[mid])

  #base case here
  if len(files) == 1:
    if pid >= start and pid <= end:
      return files[0]
    else: return None, None

  if pid > end:
    return binary_search(files[mid+1:], split_func, pid)
  elif pid < start:
    return binary_search(files[:mid], split_func, pid)
  else:
    return files[mid]

def split_h5file(filename):
  # remove suffix
  # split by "_"
  filename = os.path.splitext(filename)[0]
  return [int(x) for x in filename.split("_")]

def load_h5py_data(files, pid, data_type="text", default_size=1024):
  filename, h5py_file = binary_search(files, lambda x: split_h5file(x[0]), pid)
  if not filename:
    error = "Corresponding %s file for root_postID %d wasn't found..." % (data_type, pid)
    # raise RuntimeError(error)
    return torch.zeros(default_size)
  try:
    group = h5py_file[str(pid)]
    if data_type in group.keys():
      data = group.get(data_type)[()]
      return torch.from_numpy(data).float()
    else:
      return torch.zeros(default_size)
  except Exception as ke:
    return torch.zeros(default_size)
  # except TypeError as te:
  #   print(te)
  #   import ipdb; ipdb.set_trace()
  #   raise te
  # except Exception as e:
    # raise e

class TwitterDatasetChunk(Dataset):
  """docstring for TwitterDatasetChunk"""
  def __init__(self, df, key, colnames, user_size, text_size, image_size, label_files, label_map, text_files, image_files, dummy_user_vector=False):
    super(TwitterDatasetChunk, self).__init__()

    self.df = df
    self.colnames = colnames
    
    self.user_size = user_size
    self.text_size = text_size
    self.image_size = image_size

    self.label_files = label_files
    self.label_map = label_map
    self.text_files = text_files
    self.image_files = image_files
    self.key = key
    self.dummy_user_vector = dummy_user_vector

    self.colnames = colnames

  def __getitem__(self, idx):


    master_data = self.df.iloc[idx]
    p_id = master_data[self.key]

    tree_size = []
    max_depth = []
    avg_depth = []

    # load labels
    if self.label_files:
      label_file, label_df = binary_search(self.label_files, lambda x: self.label_map[x[0]], p_id)
      if not label_file:
        raise RuntimeError("no label file found for pid %d " % p_id)

      tree_size = label_df[label_df['root_postID'] == p_id]['tree_size'].unique()
      max_depth = label_df[label_df['root_postID'] == p_id]['max_depth'].unique()
      avg_depth = label_df[label_df['root_postID'] == p_id]['avg_depth'].unique()
      tree_size = torch.from_numpy(tree_size)
      max_depth = torch.from_numpy(max_depth)
      avg_depth = torch.from_numpy(avg_depth)

      if len(tree_size) > 1: 
        import ipdb; ipdb.set_trace()
        raise RuntimeError("why do you get multiple values for a single root_postID?")

    text_data = []
    if self.text_files:
      text_data = load_h5py_data(self.text_files, int(p_id), "text", self.text_size)

    image_data = []
    if self.image_files:
      image_data = load_h5py_data(self.image_files, int(p_id), "img", self.image_size)

    if self.dummy_user_vector:
      root_vector = torch.randn([self.user_size])
      previous_vector = torch.randn([self.user_size])
      current_vector = torch.randn([self.user_size])
    else:
      root_vector = torch.from_numpy(master_data[self.colnames[10:20]].values).float()
      previous_vector = torch.from_numpy(master_data[self.colnames[20:30]].values).float()
      current_vector = torch.from_numpy(master_data[self.colnames[30:]].values).float()
  

    return [  root_vector,
              previous_vector,
              current_vector,
              master_data['path_length_to_root'],
              text_data, 
              image_data,
              tree_size,
              max_depth,
              avg_depth,
            ]

  def __len__(self): return len(self.df)

def main():
  
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  text_files, image_files, label_files = open_data_files(args['text_filenames'], args['image_filenames'], args['label_filenames'])


  if args['header']:
    with open(args['header']) as f:
      colnames = f.readlines()[0].strip().split(",")

  if not args['label_map']: raise RuntimeError("need map to find label files")
  with open(args['label_map'], 'r') as f:
    label_map = yaml.load(f)

  files = args['master_filenames'] if args['master_filenames'] else args['train_filenames']

  dataset = TwitterDatasetChunk(
    filename=pd.read_csv(files[0], sep=",", names=self.colnames, header=None),
    key=args['key'],
    colnames=colnames,
    user_size=args['user_size'],
    text_size=args['text_size'],
    image_size=args['image_size'],
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=args['dummy_user_vector']
    )

  for i in range(1, 10):
    dataset.__getitem__(i)

if __name__ == '__main__':
  main()