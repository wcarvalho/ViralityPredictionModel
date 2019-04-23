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

def load_h5py_file(files, pid, data_type="text"):
  file = binary_search(files, split_h5file, pid)
  # import ipdb; ipdb.set_trace()
  if not file:
    error = "Corresponding %s file for root_postID %d wasn't found..." % (data_type, p_id)
    import ipdb; ipdb.set_trace()
    raise RuntimeError(error)
  return file
  # print("opening h5py %s" % os.path.basename(file))
  # return h5py.File(file, 'r',  libver='latest', swmr=True)
  #   data = f[str(pid)].get(data_type)[()]
  # # print("closing h5py %s" % os.path.basename(file))
  # return data

def close_random_h5py_file(open_h5pys):
  file_to_remove = random.choice(list(open_h5pys.keys()))
  open_h5pys[file_to_remove].close()
  # tqdm.write("closed hfpy %s" % file_to_remove)
  open_h5pys.pop(file_to_remove)

class TwitterDatasetChunk(Dataset):
  """docstring for TwitterDatasetChunk"""
  def __init__(self, filename, key, colnames, user_size, label_files, label_map, text_files, image_files, dummy_user_vector=False, open_label_dfs={}, open_text_h5pys={}, open_image_h5pys={}, max_label_files_open=10, max_hfpy_files_open=3):
    super(TwitterDatasetChunk, self).__init__()

    self.filename = filename
    self.colnames = colnames
    self.user_size = user_size
    self.label_files = label_files
    self.label_map = label_map
    self.text_files = text_files
    self.image_files = image_files
    self.key = key
    self.dummy_user_vector = dummy_user_vector
    self.max_label_files_open = max_label_files_open
    self.max_hfpy_files_open = max_hfpy_files_open

    self.df = pd.read_csv(filename, sep=",", names=colnames, header=None)
    self.colnames = colnames

    self.label_df = None

    self.open_label_dfs = open_label_dfs
    self.open_text_h5pys = open_text_h5pys
    self.open_image_h5pys = open_image_h5pys

    self.current_label_file = None

  def set_main_data(self, filename):
    self.df = pd.read_csv(filename, sep=",", names=self.colnames, header=None)

  def load_h5py_data(self, open_files_dict, file, pid, datatype):
    if not file in open_files_dict:
      if len(open_files_dict) > self.max_hfpy_files_open:
        close_random_h5py_file(open_files_dict)
      f = h5py.File(file, 'r',  libver='latest', swmr=True)
      open_files_dict[file] = f
      # tqdm.write("opened hfpy %s, %d" % (file, len(open_files_dict)))

    h5py_file = open_files_dict[file]
    data = h5py_file[str(pid)].get(datatype)[()]
    return torch.from_numpy(data).float()

  def __getitem__(self, idx):

    # sleep_time = float(np.random.randint(1,idx+1))/1000

    # print("%s. sleep time %f" % (str(self.open_label_dfs.keys()), sleep_time))
    

    master_data = self.df.iloc[idx]
    p_id = master_data[self.key]

    tree_size = []
    max_depth = []
    avg_depth = []

    # load labels
    if self.label_files:
      label_file = binary_search(self.label_files, lambda x: self.label_map[os.path.basename(x)], p_id)
      if not label_file:
        raise RuntimeError("not label file found for pid %d " % p_id)

      if not label_file in self.open_label_dfs:
        # time.sleep(sleep_time)
        # immediately after entering this if statement, set the key to None.
        # now many other parallel workers will see this and 
        if not label_file in self.open_label_dfs:
          self.open_label_dfs[label_file] = pd.read_csv(label_file, sep=",", names=['root_postID','tree_size','max_depth','avg_depth'], header=None)
          # time.sleep(sleep_time)
          # tqdm.write("opened csv %s, %d" % (label_file, len(self.open_label_dfs)))
      
      # while not self.open_label_dfs[label_file]:
      label_df = self.open_label_dfs[label_file]

      # randomly close csv when have more than 10 open
      while len(self.open_label_dfs) > self.max_label_files_open:
        to_close = random.choice(self.open_label_dfs.keys())
        self.open_label_dfs.pop(to_close)
        # tqdm.write("closed csv %s" % to_close)

      tree_size = label_df[label_df['root_postID'] == p_id]['tree_size'].values
      max_depth = label_df[label_df['root_postID'] == p_id]['max_depth'].values
      avg_depth = label_df[label_df['root_postID'] == p_id]['avg_depth'].values
      tree_size = torch.from_numpy(tree_size)
      max_depth = torch.from_numpy(max_depth)
      avg_depth = torch.from_numpy(avg_depth)

      if len(tree_size) > 1: raise RuntimeError("why do you get multiple values for a single root_postID?")

    text_data = []
    if self.text_files:
      text_file = load_h5py_file(self.text_files, p_id, "text")
      text_data = self.load_h5py_data(self.open_text_h5pys, text_file, p_id, "text")


    image_data = []
    if self.image_files:
      image_file = load_h5py_file(self.image_files, p_id, "image")
      image_data = self.load_h5py_data(self.open_image_h5pys, image_file, p_id, "image")

    if self.dummy_user_vector:
      root_vector = torch.randn([self.user_size])
      previous_vector = torch.randn([self.user_size])
      current_vector = torch.randn([self.user_size])
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

  def __len__(self): return len(self.df)

def main():
  
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

  dataset = TwitterDatasetChunk(
    filename=args['master_filenames'][0],
    key=args['key'],
    colnames=colnames,
    user_size=args['user_size'],
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=args['dummy_user_vector']
    )

  for i in range(1, 10):
    dataset.__getitem__(3)
    dataset.set_main_data(args['master_filenames'][i])
    dataset.__getitem__(3)

if __name__ == '__main__':
  main()