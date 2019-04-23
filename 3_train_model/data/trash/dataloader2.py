from pprint import pprint
import os
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

def close_h5py_filelist(file_list):
  for file in file_list:
    file.close()

def open_files(file_list, key_fn, opener, name):
  return {key_fn(file): opener(file) for file in tqdm(file_list, desc="opening %s files" % name)}

def load_train_valid_test_loaders(all_master_files, header_file, label_map_file, split_map_file, label_files, text_files, image_files, key, user_size, dummy_user_vector, shuffle, batch_size, num_workers, max_label_files_open, max_hfpy_files_open):

  if not header_file: raise RuntimeError("need header file for master csv data")
  with open(header_file) as f:
    colnames = f.readlines()[0].strip().split(",")

  if not label_map_file: raise RuntimeError("need label map to find label files for corresponding master csv data")
  with open(label_map_file, 'r') as f:
    label_map = yaml.load(f, Loader=yaml.FullLoader)

  if not split_map_file: raise RuntimeError("need map to split files into train/valid/test")
  with open(split_map_file, 'r') as f:
    split_map = yaml.load(f, Loader=yaml.FullLoader)

  split_map = {name: set(split_map[name]) for name in split_map}

  train_files = [f for f in all_master_files if os.path.basename(f) in split_map['train']]
  valid_files = [f for f in all_master_files if os.path.basename(f) in split_map['valid']]
  test_files = [f for f in all_master_files if os.path.basename(f) in split_map['test']]

  h5_filename = lambda x:  os.path.splitext(os.path.basename(x))[0]


  text_files = open_files(text_files, h5_filename, lambda x: h5py.File(x, 'r',  libver='latest', swmr=True), "h5py text")
  image_files = open_files(image_files, h5_filename, lambda x: h5py.File(x, 'r',  libver='latest', swmr=True), "h5py image")
  label_files = open_files(label_files, lambda x: os.path.basename(x), lambda x: pd.read_csv(x, sep=",", names=['root_postID','tree_size','max_depth','avg_depth'], header=None), "label")

  train_dataloader = TwitterDataloader(
    chunks=train_files,
    key=key,
    user_size=user_size,
    colnames=colnames,
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=dummy_user_vector,
    shuffle=shuffle,
    batch_size=batch_size,
    num_workers=num_workers,
    max_label_files_open=max_label_files_open,
    max_hfpy_files_open=max_hfpy_files_open)

  valid_dataloader = TwitterDataloader(
    chunks=valid_files,
    key=key,
    user_size=user_size,
    colnames=colnames,
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=dummy_user_vector,
    shuffle=False, 
    batch_size=batch_size,
    num_workers=num_workers,
    max_label_files_open=max_label_files_open,
    max_hfpy_files_open=max_hfpy_files_open)

  test_dataloader = TwitterDataloader(
    chunks=test_files,
    key=key,
    user_size=user_size,
    colnames=colnames,
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=dummy_user_vector,
    shuffle=False, 
    batch_size=batch_size,
    num_workers=num_workers,
    max_label_files_open=max_label_files_open,
    max_hfpy_files_open=max_hfpy_files_open)

  return [colnames,
          label_map,
          split_map,
          train_files,
          valid_files,
          test_files,
          train_dataloader,
          valid_dataloader,
          test_dataloader]

class TwitterDataloader(object):
  """docstring for TwitterDataloader"""
  def __init__(self, chunks, key, user_size, colnames, label_files, label_map, text_files, image_files, dummy_user_vector=False, shuffle=False, batch_size=1024, num_workers=4):
    super(TwitterDataloader, self).__init__()

    self.chunks = chunks
    self.colnames = colnames
    self.user_size = user_size
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
    self.overall_indx = 0
    self.num_chunks = len(self.chunks)
    self.num_batches_in_chunk = None

    self.twitter_dataset = None
    self.open_label_dfs={}
    self.open_text_h5pys={}
    self.open_image_h5pys={}
    self.iterator = self.load_iterator(self.chunks[self.chunk_indx])
    self.max_iterator_num_batches = self.num_batches_in_chunk

  def load_iterator(self, chunk):
    if not self.twitter_dataset:
      self.twitter_dataset = TwitterDatasetChunk(
        filename=chunk,
        key=self.key,
        colnames=self.colnames,
        user_size=self.user_size,
        label_files=self.label_files,
        label_map=self.label_map,
        text_files=self.text_files,
        image_files=self.image_files,
        dummy_user_vector=self.dummy_user_vector,
        open_label_dfs=self.open_label_dfs,
        open_text_h5pys=self.open_text_h5pys,
        open_image_h5pys=self.open_image_h5pys,
        max_label_files_open=self.max_label_files_open,
        max_hfpy_files_open=self.max_hfpy_files_open
        )
    else:
      self.twitter_dataset.set_main_data(chunk)

    self.chunk_batch_indx = 0
    self.num_batches_in_chunk = len(self.twitter_dataset)//self.batch_size + (len(self.twitter_dataset) % self.batch_size > 0)


    self.dataloader = DataLoader(self.twitter_dataset, batch_size=self.batch_size,
                            shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=self.num_workers > 1)

    return iter(self.dataloader)

  def __len__(self):
    return self.num_chunks*self.max_iterator_num_batches

  def __iter__(self):
    if self.shuffle: 
      shuffle(self.chunks)
    self.chunk_indx = 0
    self.overall_indx = 0
    return self

  def __next__(self):
    self.overall_indx += 1
    # if self.chunk_batch_indx < self.num_batches_in_chunk:
    try:
      self.chunk_batch_indx += 1
      # tqdm.write("\t%d/%d" % (self.chunk_batch_indx, self.num_batches_in_chunk))
      to_return = next(self.iterator)
      return to_return
    # else, new file:
    except StopIteration as si:
      self.chunk_indx += 1

      if self.chunk_indx >= self.num_chunks:
        raise StopIteration
      else:
        tqdm.write("file %d finished" % self.chunk_indx)
        self.iterator = self.load_iterator(self.chunks[self.chunk_indx])
        return next(self.iterator)

    except Exception as e:
      raise e

if __name__ == '__main__':
  
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  args_to_print = {name:args[name] for name in args if not "filenames" in name}
  pprint(args_to_print)
  pprint(unknown)

  colnames, label_map, split_map, train_files, valid_files, _, train_dataloader, _, _ = load_train_valid_test_loaders(
      all_master_files=args['master_filenames'],
      header_file=args['header'],
      label_map_file=args['label_map'],
      split_map_file=args['split_map'],
      label_files = args['label_filenames'],
      text_files = args['text_filenames'],
      image_files = args['image_filenames'],
      key=args['key'],
      user_size=args['user_size'],
      dummy_user_vector=args['dummy_user_vector'],
      shuffle=args['shuffle'],
      batch_size=args['batch_size'],
      num_workers=args['num_workers'],
      max_label_files_open=args['max_label_files_open'],
      max_hfpy_files_open=args['max_hfpy_files_open'])

  for batch in tqdm(train_dataloader): pass
  print("Finished going through all files...")

  close_h5py_filelist(train_files)
  close_h5py_filelist(valid_files)

  import ipdb; ipdb.set_trace()