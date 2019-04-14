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
from src.utils import path_exists, filepath_exists, AverageMeter, tensor_is_set

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

  if (r_value == 0.0).all():
    import ipdb; ipdb.set_trace()
  r_log = (r_value+EPS).log()
  p_log = (p_value+EPS).log()
  c_log = (c_value+EPS).log()

  rc_loss = F.mse_loss(r_log, (rc_length + c_value).log())  # log(r_value) = log(l + c_value)
  rp_loss = F.mse_loss(r_log, (rc_length - 1 + p_value + EPS).log())  # r_value = l - 1 + c_value
  pc_loss = F.mse_loss(p_log, c_log)                    # p_value = 1 + c_value

  return rc_loss, rp_loss, pc_loss

class Trainer(object):
  """docstring for Trainer"""
  def __init__(self, model, optimizer, device, target_name="tree_size", seed=1, micro_lambda=1, macro_lambda=1, max_epoch=10, log_dir=None, checkpoint_file=None):
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

    self.start_epoch = self.min_epoch = 0
    self.min_valid = 1e10
    self.patience = 20
    self.patience_used = 0
    self.train_losses = []
    self.valid_losses = []

    torch.manual_seed(seed)
    np.random.seed(seed)

    self.checkpoint_file = checkpoint_file
    if checkpoint_file:
      filepath_exists(checkpoint_file)

      if os.path.exists(checkpoint_file):
        self.load_from_ckpt(checkpoint_file)

    if log_dir:
      path_exists(log_dir)
      self.writer = SummaryWriter(log_dir)
      print("logging to %s" % log_dir)
    else:
      self.writer = None

    self.meters = {meter: AverageMeter() for meter in ['negative_sampling_loss', 'rc_loss', 'rp_loss', 'pc_loss', 'target_loss', 'total_loss']}

  def write_meters_to_tensorboard(self, name, step):
    for meter in self.meters:
      self.writer.add_scalar("%s/%s" % (name, meter), self.meters[meter].average, step)

  def train(self, train_dataloader, valid_dataloader):

    train_loss = self.train_losses[-1] if self.train_losses else 0
    valid_loss = self.valid_losses[-1] if self.valid_losses else 0
    pbar = tqdm(range(self.start_epoch, self.max_epoch))

    pbar.set_description("Epoch %d. Train loss=%.2f, Valid loss = %.2f" % (self.start_epoch, train_loss, valid_loss))

    for epoch in pbar:
      note = ""
      self.model.train()
      train_loss = self.train_epoch(train_dataloader, backprop=True)
      if self.writer:
        self.write_meters_to_tensorboard("train", step)
      self.meters.reset()


      self.model.eval()
      valid_loss = self.train_epoch(valid_dataloader, backprop=False)
      if self.writer:
        self.write_meters_to_tensorboard("valid", step)
      self.meters.reset()

      self.train_losses.append(train_loss)
      self.valid_losses.append(valid_loss)

      self.valid_losses
      if valid_loss < self.min_valid:
        self.min_valid = valid_loss
        self.min_epoch = epoch
        self.patience_used = 0
        self.save_to_ckpt(self.checkpoint_file)
        note = "NEW MINIMUM"
      else:
        self.patience_used += 1
        note = "patience increased to %d" % self.patience_used


      if self.patience_used >= self.patience:
        print("patience exceeded. training finished")
        break

      pbar.set_description("Epoch %d. Train loss=%.2f, Valid loss = %.2f.%s" % (self.start_epoch, train_loss, valid_loss, note))

  def train_epoch(self, dataloader, backprop=False):
    
    data_iter = iter(dataloader)
    batch_0 = next(data_iter)
    batch_1 = batch_0
    epoch_loss = 0
    pbar = tqdm(range(len(dataloader)-1))
    pbar.set_description("target: 0, follow: 0, recur: 0")
    for batch_ind in pbar:
      batch_2 = next(data_iter)
      total_loss = self.train_instance(train_batch=batch_1, random_batch=batch_2, backprop=backprop)
      epoch_loss += total_loss

      # ----- Set pbar/tqdm description --- 
      pbar.set_description("target: %.2f, follow: %.2f, recur: %.2f" % (
        self.meters['target_loss'].average,
        self.meters['negative_sampling_loss'].average,
        self.meters['rc_loss'].average + self.meters['rp_loss'].average + self.meters['pc_loss'].average
        ))
      batch_1 = batch_2

    # for last instance, use batch 0 as the "random" batch
    epoch_loss += self.train_instance(train_batch=batch_2, random_batch=batch_0)
    return epoch_loss

  def train_instance(self, train_batch, random_batch, backprop=False):

    # ---- Get data for computing outputs ---
    r_vector, p_vector, c_vector, rc_length, text_data, image_data, tree_size, max_depth, avg_depth = train_batch
    r_vector_other, p_vector_other, c_vector_other, _, _, _, _, _, _ = random_batch

    # ---- Put data on device (CPU OR GPU) ---
    r_vector = r_vector.to(self.device)
    p_vector = p_vector.to(self.device)
    c_vector = c_vector.to(self.device)
    rc_length = rc_length.to(self.device)
    text_data = text_data.to(self.device) if tensor_is_set(text_data) else text_data
    image_data = image_data.to(self.device) if tensor_is_set(image_data) else image_data
    tree_size = tree_size.to(self.device)
    max_depth = max_depth.to(self.device)
    r_vector_other = r_vector_other.to(self.device)
    p_vector_other = p_vector_other.to(self.device)
    c_vector_other = c_vector_other.to(self.device)

    # ---- Compute outputs ---
    p_followed_true, p_followed_false, p_value, c_value, r_value, pred_target = self.model(r_vector, p_vector, c_vector, r_vector_other, p_vector_other, c_vector_other, image_data, text_data)

    # ---- Compute Losses ---
    p_follow_loss = negative_sampling_loss(p_followed_true, p_followed_false)
    rc_loss, rp_loss, pc_loss = depth_loss(r_value, p_value, c_value, rc_length.float())
    recursive_loss = rc_loss + rp_loss + pc_loss

    if self.target_name == 'tree_size': target = tree_size
    elif self.target_name == 'max_depth':target = max_depth
    elif self.target_name == 'avg_depth':target = avg_depth
    else:
      raise RuntimeError("target %s not supported" % self.target_name)


    # FIXME: average depth is a pretty big number. doing log may not be sufficient for avoiding exploding gradients
    target = (target.float() + EPS).log()
    target_loss = F.mse_loss(pred_target, target)

    total_loss = target_loss + self.micro_lambda*p_follow_loss + self.macro_lambda*recursive_loss

    # ---- Backprop ---
    if backprop:
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
    self.meters['total_loss'].update(total_loss.item(), batch_len)

    return total_loss

  def load_from_ckpt(self, checkpoint_file):
    print("loading %s" % checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    self.start_epoch = self.min_epoch = checkpoint['epoch'] + 1
    self.min_valid = checkpoint['min_valid']
    self.patience_used = checkpoint['patience_used']
    self.train_losses = checkpoint['train_losses']
    self.valid_losses = checkpoint['valid_losses']
    self.model.load_state_dict(checkpoint['model'])
    self.optimizer.load_state_dict(checkpoint['optimizer'])
    torch.manual_seed(checkpoint['seed'])
    np.random.seed(checkpoint['seed'])

  def save_to_ckpt(self, checkpoint_file):
    torch.save({
        'epoch': self.min_epoch,
        'min_valid': self.min_valid,
        'model': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'seed': self.seed,
        'patience_used' : self.patience_used,
        'train_losses': self.train_losses,
        'valid_losses': self.valid_losses,
      }, checkpoint_file)
    print("Saving %s" % checkpoint_file)


def main():
  from pprint import pprint
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  args_to_print = {name:args[name] for name in args if not "filenames" in name}
  pprint(args_to_print)

  if args['header']:
    with open(args['header']) as f:
      colnames = f.readlines()[0].strip().split(",")

  if not args['label_map']: raise RuntimeError("need label map to find label files for corresponding master csv data")
  with open(args['label_map'], 'r') as f:
    label_map = yaml.load(f)

  if not args['split_map']: raise RuntimeError("need map to split files into train/valid/test")
  with open(args['split_map'], 'r') as f:
    split_map = yaml.load(f)

  split_map = {name: set(split_map[name]) for name in split_map}
  train_files = [f for f in args['master_filenames'] if f in split_map['train']]
  valid_files = [f for f in args['master_filenames'] if f in split_map['valid']]

  device = torch.device("cpu" if (args['no_cuda'] or not torch.cuda.is_available()) else "cuda")

  model = FeatureModel(user_size=args['user_size'],
    image_embed_size=args['image_size'],
    text_embed_size=args['text_size'],
    hidden_size=args['hidden_size'],
    joint_embedding_size=args['joint_embedding_size'])


  if (torch.cuda.device_count() > 1) and args['all_gpu']:
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
    checkpoint_file=args['checkpoint']
  )

  label_files = args['label_filenames']
  text_files = args['text_filenames']
  image_files = args['image_filenames']

  train_dataloader = TwitterDataloader(
    chunks=train_files,
    colnames=colnames,
    key=args['key'],
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=args['dummy_user_vector'],
    shuffle=args['shuffle'],
    batch_size=args['batch_size'], 
    num_workers=args['num_workers'])

  valid_dataloader = TwitterDataloader(
    chunks=valid_files,
    colnames=colnames,
    key=args['key'],
    label_files=label_files,
    label_map=label_map,
    text_files=text_files,
    image_files=image_files,
    dummy_user_vector=args['dummy_user_vector'],
    shuffle=False, 
    batch_size=args['batch_size'], 
    num_workers=args['num_workers'])

  trainer.train(train_dataloader, valid_dataloader)


if __name__ == '__main__':
  main()