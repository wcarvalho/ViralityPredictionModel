
if __name__ == '__main__':
  from tqdm import tqdm
  import argparse
  import h5py
  from src.utils import filepath_exists

  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--files', type=str, nargs='+', required=True)
  parser.add_argument('-o', '--output', type=str, required=True)
  parser.add_argument('-ik', '--inner-key', type=str, required=True)
  args, unknown = parser.parse_known_args()
  args = vars(args)

  filepath_exists(args['output'])

  print("merging %d files to %s" % (len(args['files']), args['output']))
  with h5py.File(args['output'], 'w') as out_file:

    for file in tqdm(args['files']):
      f = h5py.File(file, 'r')
      for group_key in list(f.keys()):
        value = f[group_key][args['inner_key']].value
        grp = out_file.create_group(group_key)
        grp.create_dataset(args['inner_key'], data=value)
      f.close()

    out_file.flush()





  print("finished writing to %s" % args['output'])