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

from data.utils import get_filenames

def load_h5py_data(h5py_file, pid, data_type="text", default_size=1024):
  try:
    group = h5py_file[str(pid)]
    if data_type in group.keys():
      data = group.get(data_type)[()]
      return torch.from_numpy(data).float()
    else:
      return torch.zeros(default_size)
  except Exception as ke:
    return torch.zeros(default_size)

class TwitterDatasetChunk(Dataset):
  """docstring for TwitterDatasetChunk"""
  def __init__(self, 
    data_file,
    image_file,
    text_file,
    label_file,
    key,
    data_header,
    label_header,
    user_size,
    text_size,
    image_size,
    dummy_user_vector=False):
    super(TwitterDatasetChunk, self).__init__()

    self.data_file = data_file
    self.image_file = image_file
    self.text_file = text_file
    self.label_file = label_file

    self.key = key

    self.data_header = data_header
    self.label_header = label_header

    self.user_size = user_size
    self.text_size = text_size
    self.image_size = image_size

    self.dummy_user_vector = dummy_user_vector

    # load files
    self.data = pd.read_csv(data_file, sep=",", names=self.data_header, header=None)
    if label_file: self.labels = pd.read_csv(label_file, sep=",", names=self.label_header, header=None)

    if image_file: self.image_data = h5py.File(image_file, 'r')
    else: self.image_data = None
    if text_file: self.text_data = h5py.File(text_file, 'r')
    else: self.text_data = None
  
  def __len__(self): return len(self.data)

  def __getitem__(self, idx):


    data = self.data.iloc[idx]
    pid = data[self.key]


    # load labels
    tree_size = []
    max_depth = []
    avg_depth = []
    if self.label_file:

      tree_size = self.labels[self.labels['root_postID'] == pid]['tree_size'].unique()[0]
      max_depth = self.labels[self.labels['root_postID'] == pid]['max_depth'].unique()[0]
      avg_depth = self.labels[self.labels['root_postID'] == pid]['avg_depth'].unique()[0]
      tree_size = torch.tensor(tree_size)
      max_depth = torch.tensor(max_depth)
      avg_depth = torch.tensor(avg_depth)

      # if len(tree_size) > 1: 
      #   tree_size = tree_size[:1]
      #   max_depth = max_depth[:1]
      #   avg_depth = avg_depth[:1]

        # import ipdb; ipdb.set_trace()
        # raise RuntimeError("why do you get multiple values for a single root_postID?")

    text_data = []
    if self.text_file:
      text_data = load_h5py_data(self.text_data, int(pid), "text", self.text_size)

    image_data = []
    if self.image_file:
      image_data = load_h5py_data(self.image_data, int(pid), "img", self.image_size)

    if self.dummy_user_vector:
      root_vector = torch.randn([self.user_size])
      previous_vector = torch.randn([self.user_size])
      current_vector = torch.randn([self.user_size])

    else:
      root_vector = torch.from_numpy(data[self.data_header[10:20]].values).float()
      previous_vector = torch.from_numpy(data[self.data_header[20:30]].values).float()
      current_vector = torch.from_numpy(data[self.data_header[30:]].values).float()

    return [  root_vector,
              previous_vector,
              current_vector,
              data['path_length_to_root'],
              text_data, 
              image_data,
              tree_size,
              max_depth,
              avg_depth,
            ]


  def __del__(self):
    # print("closing %s and friends" % self.data_file)
    self.close()

  def close(self):
    if self.image_data: 
      self.image_data.close()
      self.image_data = None
    if self.text_data: 
      self.text_data.close()
      self.text_data = None

def main():

  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)


  if args['data_header']:
    with open(args['data_header']) as f:
      data_header = f.readlines()[0].strip().split(",")
  if args['label_header']:
    with open(args['label_header']) as f:
      label_header = f.readlines()[0].strip().split(",")

  data_files = args['valid_data_files']
  image_files = args['valid_image_files']
  text_files = args['valid_text_files']
  label_files = args['valid_label_files']
  key = args['key']
  user_size = args['user_size']
  text_size = args['text_size']
  image_size = args['image_size']
  dummy_user_vector = args['dummy_user_vector']

  for data_file, image_file, text_file, label_file in tqdm(zip(data_files, image_files, text_files, label_files)):
    dataset = TwitterDatasetChunk(
      data_file=data_file,
      image_file=image_file,
      text_file=text_file,
      label_file=label_file,
      key=key,
      data_header=data_header,
      label_header=label_header,
      user_size=user_size,
      text_size=text_size,
      image_size=image_size,
      dummy_user_vector=dummy_user_vector
      )
    if not len(dataset):
      import ipdb; ipdb.set_trace()

  # dataloader = DataLoader(dataset, batch_size=args['batch_size'],
  #                           shuffle=args['shuffle'], num_workers=args['num_workers'])

  # for batch in tqdm(dataloader): pass

if __name__ == '__main__':
  main()