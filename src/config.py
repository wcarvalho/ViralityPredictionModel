import argparse
def load_parser():

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type=str, default='basic', choices=['basic', 'feature'])
  parser.add_argument('-mf', '--master-filenames', type=str, nargs='+', default=[])
  parser.add_argument('-lf', '--label-filenames', type=str, nargs='+', default=[])
  parser.add_argument('-lm', '--label-map', type=str, default=None, help='dict with start,end pids of label chunks')
  parser.add_argument('-if', '--image-filenames', type=str, nargs='+', default=[])
  parser.add_argument('-tf', '--text-filenames', type=str, nargs='+', default=[])
  parser.add_argument('-bs', '--batch-size', type=int, default=1024)
  parser.add_argument('-k', '--key', type=str, default="root_postID")
  parser.add_argument('-e', '--epochs', type=int, default=1000)
  parser.add_argument('-s', '--vocab-size', type=int, default=13649798)
  # parser.add_argument('-fl', '--file-length', type=int, default=0)
  # parser.add_argument('-s', '--shuffle', type=int, default=1, choices=[0,1])
  parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
  parser.add_argument('-he', '--header', type=str, default=None)
  parser.add_argument('-cn', '--colnames', type=str, default=["r_pid", "r_uid", "r_t", "p_pid", "p_uid", "p_t", "c_pid", "c_uid", "c_t", "text", "data"], nargs='+')
  return parser
