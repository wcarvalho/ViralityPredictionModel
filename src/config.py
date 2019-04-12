import argparse
def load_parser():

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type=str, default='basic', choices=['basic', 'feature'])
  parser.add_argument('-mf', '--master-filename', type=str, required=True)
  parser.add_argument('-lf', '--label-filename', type=str, default=None)
  parser.add_argument('-if', '--image-filename', type=str, default=None)
  parser.add_argument('-tf', '--text-filename', type=str, default=None)
  parser.add_argument('-bs', '--batch-size', type=int, default=128)
  parser.add_argument('-fl', '--file-length', type=int, default=0)
  parser.add_argument('-e', '--epochs', type=int, default=1000)
  # parser.add_argument('-s', '--shuffle', type=int, default=1, choices=[0,1])
  parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
  parser.add_argument('-cn', '--colnames', type=str, default=["r_pid", "r_uid", "r_t", "p_pid", "p_uid", "p_t", "c_pid", "c_uid", "c_t", "text", "data"], nargs='+')
  return parser
