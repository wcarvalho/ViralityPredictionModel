
if __name__ == '__main__':
  import argparse
  import h5py

  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--files', type=str, nargs='+', required=True)
  parser.add_argument('-o', '--output', type=str, required=True)
  parser.add_argument('-ik', '--inner-key', type=str, required=True)
  args, unknown = parser.parse_known_args()
  args = vars(args)

  out_file = h5py.File(args['output'], 'w')

  for file in files:
    f = h5py.File(tem_file, 'r')

    for group_key in list(f.keys()):
      value = f[group_key][args['inner_key']].value
      grp = out_file.create_group(group_key)
      grp.create_dataset(args['inner_key'], data=value)

  out_file.flush()
  out_file.close()
