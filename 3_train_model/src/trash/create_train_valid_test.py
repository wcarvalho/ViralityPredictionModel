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
  parser.add_argument('-mtm', '--max-month', type=int, default=6)
  parser.add_argument('-o', '--outdir', type=str, default="data/splits/")
  args, unknown = parser.parse_known_args()
  args = vars(args)

  pprint(args)

  path_exists(args['outdir'])

  train_set = set()
  valid_set = set()
  test_set = set()

  colnames = args['colnames']
  if args['header']:
    with open(args['header']) as f:
      colnames = f.readlines()[0].strip().split(",")

  max_train_month = args["max_month"] - 2

  for file in tqdm(args['files']):
    df = pd.read_csv(file, sep=",", names=colnames, header=None)
    df['month'] = df['root_timestamp'].transform(lambda x: dateutil.parser.parse(time.ctime(x)).month)

    train_df = df[df['month'] <= max_train_month]
    train_set.update(df[key].unique())

    df_valid_test = df[df['month'] > max_train_month]
    df_valid = df_valid_test[df_valid_test[key] % 2 == 0]
    df_test = df_valid_test[df_valid_test[key] % 2 == 1]
    import ipdb; ipdb.set_trace()
    valid_set.update(df_valid[key].unique())
    test_set.update(df_test[key].unique())

  write_set(args['outdir'], "train.txt", sorted(train_set))
  write_set(args['outdir'], "valid.txt", sorted(valid_set))
  write_set(args['outdir'], "test.txt", sorted(test_set))


    del df

  with open(args['outfile'], 'w') as f:
      f.write("%d\n" % max_month)

  for batch_df in tqdm(df):
