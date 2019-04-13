import os

import numpy as np
import yaml
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from data.dataloader import TwitterDataloader
from src.models.feature_model import FeatureModel
from src.utils import path_exists, filepath_exists, AverageMeter

EPS = 1e-12

def negative_sampling_loss(positive, negative):

  # convert negative to batch_size x num_negative
  negative = torch.stack(negative).permute(1,0,2).squeeze(2) 

  # concatenate both negative and positive alone second dimension
  pred = torch.cat([positive, negative], dim=-1)
  
  # now create labels of positive and negative along respective columns
  true = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)], dim=-1)

  # don't normalize yet
  loss = F.binary_cross_entropy_with_logits(pred, true, reduction='none')

  # sum over columns and do mean over batch
  loss = loss.sum(dim=-1).mean()

  return loss

def depth_loss(r_value, p_value, c_value, rc_length):
  """Expected depth consistency loss"""
  if (rc_length<1.0).any():
    error = "Why is any length between root and child below zero?"
    import ipdb; ipdb.set_trace()
    raise (error)

  r_log = (r_value+EPS).log()
  p_log = (p_value+EPS).log()
  c_log = (c_value+EPS).log()

  rc_loss = F.mse_loss(r_log, (rc_length + c_value).log())  # log(r_value) = log(l + c_value)
  rp_loss = F.mse_loss(r_log, (rc_length - 1 + p_value + EPS).log())  # r_value = l - 1 + c_value
  pc_loss = F.mse_loss(p_log, c_log)                    # p_value = 1 + c_value

  return rc_loss, rp_loss, pc_loss

class Trainer(object):
  """docstring for Trainer"""
  def __init__(self, model, optimizer, device, target_name="tree_size", seed=1, micro_lambda=1, macro_lambda=1, max_epoch=10, log_dir=None, checkpoint=None):
    super(Trainer, self).__init__()
    self.model = model
    self.optimizer = optimizer
    self.device = device
    self.target_name = target_name
    self.seed = seed
    self.micro_lambda = micro_lambda
    self.macro_lambda = macro_lambda
    self.max_epoch = max_epoch
    self.log_dir = log_dir
    self.checkpoint = checkpoint

    self.start_epoch = 0

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

    self.meters = {meter: AverageMeter() for meter in ['negative_sampling_loss', 'rc_loss', 'rp_loss', 'pc_loss', 'target_loss']}

  def train(self, dataloader):
    
    pbar = tqdm(range(self.start_epoch, self.max_epoch))
    pbar.set_description("Epoch %d" % self.start_epoch)
    for epoch in pbar:
      self.model.train()
      self.train_epoch(dataloader)
      self.meters.reset()
      pbar.set_description("Epoch %d" % epoch)

  def train_epoch(self, dataloader):
    
    data_iter = iter(dataloader)
    batch_0 = next(data_iter)
    batch_1 = batch_0

    pbar = tqdm(range(len(dataloader)-1))
    pbar.set_description("target: 0, follow: 0, recur: 0")
    for batch_ind in pbar:
      # ---- Get data for computing outputs ---
      r_vector, p_vector, c_vector, rc_length, text_data, image_data, tree_size, max_depth, avg_depth = batch_1

      batch_2 = next(data_iter)
      r_vector_other, p_vector_other, c_vector_other, _, _, _, _, _, _ = batch_2

      # ---- Compute outputs ---
      p_followed_true, p_followed_false, p_value, c_value, r_value, pred_target = model(r_vector, p_vector, c_vector, r_vector_other, p_vector_other, c_vector_other, image_data, text_data)

      # ---- Compute Losses ---
      p_follow_loss = negative_sampling_loss(p_followed_true, p_followed_false)
      rc_loss, rp_loss, pc_loss = depth_loss(r_value, p_value, c_value, rc_length.float().to(self.device))
      recursive_loss = rc_loss + rp_loss + pc_loss

      if self.target_name == 'tree_size': target = tree_size
      elif self.target_name == 'max_depth':target = max_depth
      elif self.target_name == 'avg_depth':target = avg_depth
      else:
        raise RuntimeError("target %s not supported" % self.target_name)
      target_loss = F.mse_loss(pred_target, target.float().to(self.device))

      total_loss = target_loss + self.micro_lambda*p_follow_loss + self.macro_lambda*recursive_loss
      import ipdb; ipdb.set_trace()
      # ---- Backprop ---
      self.optimizer.zero_grad()
      total_loss.backward()
      self.optimizer.step()

      # ---- Record Losses ---
      batch_len = r_vector.shape[0]
      self.meters['negative_sampling_loss'].update(p_follow_loss.item(), batch_len)
      self.meters['rc_loss'].update(rc_loss.item(), batch_len)
      self.meters['rp_loss'].update(rp_loss.item(), batch_len)
      self.meters['pc_loss'].update(pc_loss.item(), batch_len)
      self.meters['target_loss'].update(target_loss.item(), batch_len)

      # ----- Set pbar/tqdm description --- 
      pbar.set_description("target: %.2f, follow: %.2f, recur: %.2f" % (
        self.meters['target_loss'].average,
        self.meters['negative_sampling_loss'].average,
        self.meters['rc_loss'].average + self.meters['rp_loss'].average + self.meters['pc_loss'].average
        ))
      batch_1 = batch_2


  def load_from_ckpt(self, checkpoint_file):
    print("loading %s" % checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    self.start_epoch = checkpoint['epoch']+1
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

  device = torch.device("cpu" if (args['no_cuda'] or not torch.cuda.is_available()) else "cuda")

  model = FeatureModel(user_size=args['user_size'],
    image_embed_size=1024,
    text_embed_size=768,
    hidden_size=256,
    joint_embedding_size=256)
  if torch.cuda.device_count() > 1 and args['all_gpu']:
    print("Using %d GPUS!" % torch.cuda.device_count())
    model = nn.DataParallel(model)
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

  trainer = Trainer(model=model,
    optimizer=optimizer,
    device=device,
    target_name=args['target'],
    seed=args['seed'],
    micro_lambda=args['micro_lambda'],
    macro_lambda=args['macro_lambda'],
    max_epoch=args['epochs'],
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

