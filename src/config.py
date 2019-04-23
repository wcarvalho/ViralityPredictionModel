import argparse
def load_parser():


  parser = argparse.ArgumentParser()


  logging = parser.add_argument_group("logging settings")
  logging.add_argument('-ld', '--log-dir', type=str, default=None)
  logging.add_argument('-ckpt', '--checkpoint', type=str, default=None)
  logging.add_argument('-v', '--verbosity', type=int, default=0, help='1=important prints. 2=detailed prints.')
  logging.add_argument('-sf', '--save-frequency', type=int, default=1000, help='save every k batches. useful for big data')


  file = parser.add_argument_group("file settings")
  # file.add_argument('-mf', '--master-filenames', type=str, nargs='+', default=[])

  file.add_argument('--train-data-files', type=str, nargs='+', default=[])
  file.add_argument('--train-image-files', type=str, nargs='+', default=[])
  file.add_argument('--train-text-files', type=str, nargs='+', default=[])
  file.add_argument('--train-label-files', type=str, nargs='+', default=[])

  file.add_argument('--valid-data-files', type=str, nargs='+', default=[])
  file.add_argument('--valid-image-files', type=str, nargs='+', default=[])
  file.add_argument('--valid-text-files', type=str, nargs='+', default=[])
  file.add_argument('--valid-label-files', type=str, nargs='+', default=[])

  file.add_argument('--test-data-files', type=str, nargs='+', default=[])
  file.add_argument('--test-image-files', type=str, nargs='+', default=[])
  file.add_argument('--test-text-files', type=str, nargs='+', default=[])
  file.add_argument('--test-label-files', type=str, nargs='+', default=[])

  # file.add_argument('-lf', '--label-filenames', type=str, nargs='+', default=[])
  # file.add_argument('-lm', '--label-map', type=str, default=None, help='dict with start,end pids of label chunks')
  # file.add_argument('-s,', '--split-map', type=str, default=None, help='dictionary with train/valid/test splits for --master-filenames. or you can use train/valid/test-filenames')
  # file.add_argument('-if', '--image-filenames', type=str, nargs='+', default=[])
  # file.add_argument('-tf', '--text-filenames', type=str, nargs='+', default=[])
  file.add_argument('-k', '--key', type=str, default="root_postID")
  file.add_argument('--data-header', type=str, default=None)
  file.add_argument('--label-header', type=str, default=None)

  # file.add_argument('--max-label-files-open', type=int, default=2)
  # file.add_argument('--max-hfpy-files-open', type=int, default=2)



  training = parser.add_argument_group("training settings")
  training.add_argument('-bs', '--batch-size', type=int, default=128)
  training.add_argument('-s', '--seed', type=int, default=1)
  training.add_argument('-e', '--epochs', type=int, default=10000)
  training.add_argument('--macro-lambda', type=int, default=1)
  training.add_argument('--micro-lambda', type=int, default=.001)
  training.add_argument('--num-workers', type=int, default=6)
  training.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
  training.add_argument('--no-cuda', action='store_true', default=False)
  training.add_argument('--all-gpu', type=int, default=0, choices=[0,1])
  training.add_argument('--user-only', type=int, default=0, choices=[0,1])
  training.add_argument('--content-only', type=int, default=0, choices=[0,1])
  training.add_argument('--shuffle', type=int, default=1, choices=[0,1])
  training.add_argument('--target', type=str, default='tree_size', choices=['tree_size', 'max_depth', 'avg_depth'])


  model = parser.add_argument_group("model settings")
  model.add_argument('-vs', '--vocab-size', type=int, default=13649798)
  model.add_argument('-us', '--user-size', type=int, default=10, help='size of initial user vector')
  model.add_argument('--image-size', type=int, default=2048, help='size of initial image vector')
  model.add_argument('--text-size', type=int, default=768, help='size of initial text vector')
  model.add_argument('--hidden-size', type=int, default=256, help='size of initial text vector')
  model.add_argument('--joint-embedding-size', type=int, default=256, help='size of initial text vector')
  model.add_argument('-dv', '--dummy-user-vector', action='store_true', default=False)

  return parser



