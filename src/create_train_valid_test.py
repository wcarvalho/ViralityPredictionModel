import time
from pprint import pprint
import os
import pandas as pd
from tqdm import trange, tqdm
from src.utils import path_exists
import h5py
import operator
import dateutil.parser

def write_set(path, name, ll):
  with open(os.path.join(path, name), 'w') as f:
    for item in ll:
      f.write("%d\n" % item)

if __name__ == '__main__':
  from src.config import load_parser
  parser = load_parser()
  parser.add_argument('-k', '--key', type=str, default="root_postID")
  parser.add_argument('-bb', '--max-train-month', type=int, default=6)
  parser.add_argument('-o', '--outdir', type=str, default="data/splits/")
  args, unknown = parser.parse_known_args()
  args = vars(args)

  pprint(args)

  path_exists(args['outdir'])

  train_set = set()
  valid_set = set()
  test_set = set()

  colnames = args['colnames']
  key = args['key']
  if args['header']:
    with open(args['header']) as f:
      colnames = f.readlines()[0].strip().split(",")

  print("reading csv")
  df = pd.read_csv(args['master_filename'], sep=",", names=colnames, header=None, chunksize=args['batch_size'])

  for batch_df in tqdm(df):
    batch_df['month'] = batch_df['root_timestamp'].transform(lambda x: dateutil.parser.parse(time.ctime(x)).month)
    train_df = batch_df[batch_df['month'] <= args["max_train_month"]]
    train_set.update(batch_df[key].unique())

    df_valid_test = batch_df[batch_df['month'] > args["max_train_month"]]
    df_valid = df_valid_test[df_valid_test[key] % 2 == 0]
    df_test = df_valid_test[df_valid_test[key] % 2 == 1]

    valid_set.update(df_valid[key].unique())
    test_set.update(df_test[key].unique())

  write_set(args['outdir'], "train.txt", train_set)
  write_set(args['outdir'], "valid.txt", valid_set)
  write_set(args['outdir'], "test.txt", test_set)


