import yaml
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import random
import argparse
import h5py


from src.utils import get_filenames

def binary_search(files, split_func, pid):
  if not len(files): return None

  mid = len(files)//2
  start, end = split_func(files[mid])

  #base case here
  if len(files) == 1:
    if pid >= start and pid <= end:
      return files[0]
    else: return None

  if pid > end:
    return binary_search(files[mid+1:], split_func, pid)
  elif pid < start:
    return binary_search(files[:mid], split_func, pid)
  else:
    return files[mid]

def split_h5file(full_path):
  # get filename
  # remove suffix
  # split by "_"
  filename = os.path.splitext(os.path.basename(full_path))[0]
  return [int(x) for x in filename.split("_")]

def load_h5py_data(files, pid, data_type="text"):
  file = binary_search(files, split_h5file, pid)
  # import ipdb; ipdb.set_trace()
  print(file)
  print(pid)
  if not file:
    error = "Corresponding %s file for root_postID %d wasn't found..." % (data_type, p_id)
    import ipdb; ipdb.set_trace()
    raise RuntimeError(error)
  with h5py.File(file, 'r') as f:
    return f[str(pid)].get(data_type)[()]


class TwitterDatasetChunk(Dataset):
  """docstring for TwitterDatasetChunk"""
  def __init__(self, filename, key, colnames, label_files, label_map, text_files, image_files, dummy_user_vector=False):
    super(TwitterDatasetChunk, self).__init__()

    self.filename = filename
    self.colnames = colnames
    self.label_files = label_files
    self.label_map = label_map
    self.text_files = text_files
    self.image_files = image_files
    self.key = key
    self.dummy_user_vector = dummy_user_vector

    self.df = pd.read_csv(filename, sep=",", names=colnames, header=None)
    self.length = len(self.df)
    self.colnames = colnames

    self.label_df = None
    self.current_label_file = None


  def __getitem__(self, idx):

    master_data = self.df.iloc[idx]
    p_id = master_data[self.key]

    tree_size = []
    max_depth = []
    avg_depth = []
    # load labels
    if self.label_files:
      label_file = binary_search(self.label_files, lambda x: self.label_map[x], p_id)
      if not label_file:
        raise RuntimeError("not label file found for pid %d " % p_id)
      if label_file != self.current_label_file: # load new one
        self.label_df = pd.read_csv(label_file, sep=",", names=['root_postID','tree_size','max_depth','avg_depth'], header=None)
        self.current_label_file = label_file

      tree_size = self.label_df[self.label_df['root_postID'] == p_id]['tree_size'].values
      max_depth = self.label_df[self.label_df['root_postID'] == p_id]['max_depth'].values
      avg_depth = self.label_df[self.label_df['root_postID'] == p_id]['avg_depth'].values
      tree_size = torch.from_numpy(tree_size)
      max_depth = torch.from_numpy(max_depth)
      avg_depth = torch.from_numpy(avg_depth)

      if len(tree_size) > 1: raise RuntimeError("why do you get multiple values for a single root_postID?")

    text_data = []
    if self.text_files:
      text_data = load_h5py_data(self.text_files, p_id, "text")
      text_data = torch.from_numpy(text_data).float()

    image_data = []
    if self.image_files:
      image_data = load_h5py_data(self.image_files, p_id, "image")
      image_data = torch.from_numpy(image_data).float()

    if self.dummy_user_vector:
      root_vector = torch.randn([20])
      previous_vector = torch.randn([20])
      current_vector = torch.randn([20])
    else:
      master_data['root_userID']
      master_data['previous_userID']
      master_data['current_userID']
      raise NotImplementedError
    
    # import ipdb; ipdb.set_trace()
    # root_to_current_path_length = torch.from_numpy(master_data['root_to_current_path_length'])

    return [  root_vector,
              previous_vector,
              current_vector,
              master_data['root_to_current_path_length'],
              text_data, 
              image_data,
              tree_size,
              max_depth,
              avg_depth,
            ]

  def __len__(self): return self.length

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

  dataset = TwitterDatasetChunk(filename=args['master_filenames'][0],
    colnames=colnames,
    key=args['key'],
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    dummy_user_vector=args['dummy_user_vector'],
    image_files=image_files)

  dataset.__getitem__(3)
