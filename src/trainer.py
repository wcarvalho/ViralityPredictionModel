import yaml
from data.dataloader import TwitterDataloader
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from src.models.feature_model import FeatureModel
from src.utils import path_exists

def negative_sampling_loss(positive, negative):

  positive_y = torch.ones(positive.shape[0])
  negative_y = torch.zeros(negative.shape[0])

  positive_loss = F.binary_cross_entropy_with_logits(positive, positive_y)
  negative_loss = F.binary_cross_entropy_with_logits(negative, negative_y)

  return positive_loss + negative_loss


class Trainer(object):
  """docstring for Trainer"""
  def __init__(self, model, optimizer, seed=1, log_dir=None, checkpoint=None):
    super(Trainer, self).__init__()
    self.model = model
    self.optimizer = optimizer
    self.log_dir = log_dir
    self.checkpoint = checkpoint
    self.seed=seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    if checkpoint:
      filepath_exists(checkpoint)

      if os.path.exists(checkpoint):
        self.load_from_ckpt(checkpoint)

    if log_dir:
      path_exists(log_dir)
      self.writer = SummaryWriter(log_dir)
      print("logging to %s" % log_dir)
    else:
      self.writer = None

  def train(self, dataloader):
    
    for epoch in self.epochs:
      self.train_epoch(dataloader)

  def train_epoch(self):
    data_iter = iter(dataloader)
    batch_0 = next(data_iter)
    batch_1 = batch_0
    for batch_ind in range(len(dataloader)-1):
      r_vector, p_vector, c_vector, rc_length, text_data, image_data, tree_size, max_depth, avg_depth = batch_1

      batch_2 = next(data_iter)
      r_vector_other, p_vector_other, c_vector_other, _, _, _, _, _, _ = batch_2

      p_followed_true, p_followed_false, p_value, c_value, r_value, target = model(r_vector, p_vector, c_vector, r_vector_other, p_vector_other, c_vector_other, image_data, text_data)

      p_follow_loss = negative_sampling_loss(p_followed_true, p_followed_false)
      batch_1 = batch_2

  def load_from_ckpt(self, checkpoint_file):
    print("loading %s" % checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    self.epoch  = checkpoint['epoch']+1
    self.iter = checkpoint['iter']
    # self.min_valid = checkpoint['min_valid']

    self.model.load_state_dict(checkpoint['model'])
    self.optimizer.load_state_dict(checkpoint['optimizer'])
    torch.manual_seed(checkpoint['seed'])
    np.random.seed(checkpoint['seed'])

  def save_to_ckpt(self, checkpoint_file):
    torch.save({
        'epoch': self.epoch,
        # 'min_valid': self.min_valid,
        'model': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'seed': self.seed,
        'iter': self.iter,
      }, checkpoint_file)
    print("Saving %s" % checkpoint_file)

if __name__ == '__main__':

  from pprint import pprint
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  if args['header']:
    with open(args['header']) as f:
      colnames = f.readlines()[0].strip().split(",")

  label_files = args['label_filenames']
  text_files = args['text_filenames']
  image_files = args['image_filenames']

  if not args['label_map']: raise RuntimeError("need map to find label files")
  with open(args['label_map'], 'r') as f:
    label_map = yaml.load(f)


  model = FeatureModel(user_size=args['user_size'],
    image_embed_size=1024,
    text_embed_size=768,
    hidden_size=256,
    joint_embedding_size=256)
  optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

  trainer = Trainer(model=model,
    optimizer=optimizer,
    seed=args['seed'],
    log_dir=args['log_dir'],
    checkpoint=args['checkpoint']
  )

  dataloader = TwitterDataloader(chunks=args['master_filenames'],
    colnames=colnames,
    key=args['key'],
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=args['dummy_user_vector'],
    shuffle=False, batch_size=args['batch_size'], num_workers=4)

  trainer.train(dataloader)

