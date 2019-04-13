import argparse
import time
from pprint import pprint
import os
import pandas as pd
from tqdm import trange, tqdm
from src.utils import path_exists
import h5py
import operator
import dateutil.parser


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--files', type=str, nargs="+", required=True)
  parser.add_argument('-o', '--outfile', type=str, default="data/max_month.txt")
  args, unknown = parser.parse_known_args()
  args = vars(args)

  pprint(args)

  filepath_exists(args['outfile'])

  nfiles = len(args['files'])
  train_files = args['files'][:int(nfiles*.7)]

  last_30_first = args['files'][int(nfiles*.7):int(nfiles*.8)]
  last_30_second = args['files'][int(nfiles*.8):int(nfiles*.9)]
  last_30_third = args['files'][int(nfiles*.9):]


  validation_files = last_30_second
  test_files = last_30_first + last_30_third

  split = {
    'train':train_files,
    'valid':validation_files,
    'test':test_files
  }

  with open(args['outfile'], 'w') as f:
    yaml.dump(ends, f, default_flow_style=False)
