import yaml
import time
from pprint import pprint
import os
import pandas as pd
from tqdm import trange, tqdm
from src.utils import filepath_exists
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
  parser.add_argument('-f', '--files', type=str, nargs="+", required=True)
  parser.add_argument('-o', '--outfile', type=str, default="data/max_month.txt")
  args, unknown = parser.parse_known_args()
  args = vars(args)
  pprint(args)


  filepath_exists(args['outfile'])

  colnames = args['colnames']
  if args['header']:
    with open(args['header']) as f:
      colnames = f.readlines()[0].strip().split(",")

  ends = {}
  for file in tqdm(args['files']):
    df = pd.read_csv(file, sep=",", names=colnames, header=None)
    ends[file] = [int(df[args['key']].min()), int(df[args['key']].max())]
    del df

  pprint(ends)
  with open(args['outfile'], 'w') as f:
    yaml.dump(ends, f, default_flow_style=False)

