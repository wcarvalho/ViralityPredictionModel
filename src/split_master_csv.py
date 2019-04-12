from pprint import pprint
import os
import pandas as pd
from tqdm import trange, tqdm
from src.utils import path_exists
import h5py
import operator

if __name__ == '__main__':
  from src.config import load_parser
  parser = load_parser()
  parser.add_argument('-k', '--key', type=str, default="root_postID")
  parser.add_argument('-bb', '--bucket-base', type=int, default=50000)
  parser.add_argument('-o', '--outdir', type=str, default="data/data/")
  args, unknown = parser.parse_known_args()
  args = vars(args)

  pprint(args)

  colnames = args['colnames']
  key = args['key']
  if args['header']:
    with open(args['header']) as f:
      colnames = f.readlines()[0].strip().split(",")

  bucket_base = args['bucket_base']
  path_exists(args['outdir'])

  print("reading csv")
  df = pd.read_csv(args['master_filename'], sep=",", names=colnames, header=None)

  dfs = dict(tuple(df.groupby(args['key'])))
  print("sorting csv")
  sorted_list = sorted(dfs.items(), key=operator.itemgetter(0))

  print("writing csvs to individual hd5 files")
  for idx in trange((len(sorted_list) + bucket_base - 1) // bucket_base, ncols=60):
    batch_list = sorted_list[idx * bucket_base:(idx+1) * bucket_base]
    if len(batch_list) == 0:
        continue
    fname = os.path.join(args['outdir'], '{}_{}.h5'.format(batch_list[0][0], batch_list[-1][0]))
    hf = h5py.File(fname, 'w')
    for pid, df in batch_list:
        grp = hf.create_group(args['key'])
        for colname in colnames:
          if colname == args['key']: continue
          grp.create_dataset(colname, data=df[colname].values)
    hf.flush()

