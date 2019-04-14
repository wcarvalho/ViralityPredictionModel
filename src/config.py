import argparse
def load_parser():


  parser = argparse.ArgumentParser()


  logging = parser.add_argument_group("logging settings")
  logging.add_argument('-ld', '--log-dir', type=str, default='logs/feature_model/tb')
  logging.add_argument('-ckpt', '--checkpoint', type=str, default='logs/feature_model/ckpt.th')


  file = parser.add_argument_group("file settings")
  file.add_argument('-mf', '--master-filenames', type=str, nargs='+', default=[])
  file.add_argument('-lf', '--label-filenames', type=str, nargs='+', default=[])
  file.add_argument('-lm', '--label-map', type=str, default=None, help='dict with start,end pids of label chunks')
  file.add_argument('-s,', '--split-map', type=str, default=None, help='dictionary with train/valid/test splits for --master-filenames')
  file.add_argument('-if', '--image-filenames', type=str, nargs='+', default=[])
  file.add_argument('-tf', '--text-filenames', type=str, nargs='+', default=[])
  file.add_argument('-k', '--key', type=str, default="root_postID")
  file.add_argument('-he', '--header', type=str, default=None)



  training = parser.add_argument_group("training settings")
  training.add_argument('-bs', '--batch-size', type=int, default=1024)
  training.add_argument('-s', '--seed', type=int, default=1)
  training.add_argument('-e', '--epochs', type=int, default=1000)
  training.add_argument('--macro-lambda', type=int, default=1)
  training.add_argument('--micro-lambda', type=int, default=1)
  training.add_argument('--num-workers', type=int, default=6)
  training.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
  training.add_argument('--no-cuda', action='store_true', default=False)
  training.add_argument('--all-gpu', type=int, default=0, choices=[0,1])
  training.add_argument('--shuffle', type=int, default=1, choices=[0,1])
  training.add_argument('--target', type=str, default='tree_size', choices=['tree_size', 'max_depth', 'avg_depth'])


  model = parser.add_argument_group("model settings")
  model.add_argument('-vs', '--vocab-size', type=int, default=13649798)
  model.add_argument('-us', '--user-size', type=int, default=20, help='size of initial user vector')
  model.add_argument('--image-size', type=int, default=2048, help='size of initial image vector')
  model.add_argument('--text-size', type=int, default=768, help='size of initial text vector')
  model.add_argument('--hidden-size', type=int, default=256, help='size of initial text vector')
  model.add_argument('--joint-embedding-size', type=int, default=256, help='size of initial text vector')
  model.add_argument('-dv', '--dummy-user-vector', action='store_true', default=False)

  return parser



