import os
import glob
import numpy as np
import yaml
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

from data.twitter_chunk import TwitterDatasetChunk
from data.utils import get_overlapping_data_files

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


  r_log = (r_value+EPS).log()
  p_log = (p_value+EPS).log()
  c_log = (c_value+EPS).log()

  rc_loss = F.mse_loss(r_log, (rc_length + c_value + EPS).log())
  rp_loss = F.mse_loss(r_log, (rc_length - 1 + p_value + EPS).log())
  pc_loss = F.mse_loss(p_log, c_log)

  if (rc_loss != rc_loss).any() or (rp_loss != rp_loss).any() or (pc_loss != pc_loss).any():
    import ipdb; ipdb.set_trace()
  return rc_loss, rp_loss, pc_loss

def match_sizes(batch_1, batch_2):
  """For whichever batch is smaller. keep repeating its elements until you"""

  num_times = 0
  if len(batch_2) < len(batch_1):
    while len(batch_2) < len(batch_1):
      difference = len(batch_1) - len(batch_2)
      batch_2 = torch.cat([batch_2, batch_2[:difference]], dim=0)
      num_times += 1
      if num_times > 10: import ipdb; ipdb.set_trace()

  elif len(batch_1) < len(batch_2):
    while len(batch_1) < len(batch_2):
      difference = len(batch_2) - len(batch_1)
      batch_1 = torch.cat([batch_1, batch_1[:difference]], dim=0)
      num_times += 1
      if num_times > 10: import ipdb; ipdb.set_trace()

  return batch_1, batch_2

class Trainer(object):
  """docstring for Trainer"""
  def __init__(self, 
    model,
    optimizer,
    device,
    key,
    data_header,
    label_header,
    user_size,
    text_size,
    image_size,
    dummy_user_vector=False,
    seed=1,
    micro_lambda=1,
    macro_lambda=1,
    max_epoch=10,
    batch_size=1024,
    num_workers=8,
    shuffle=True,
    log_dir=None,
    checkpoint_file=None,
    save_frequency=200,
    verbosity=0
    ):
    super(Trainer, self).__init__()
    self.model = model
    self.optimizer = optimizer
    self.device = device
    self.key = key
    self.data_header = data_header
    self.label_header = label_header
    self.user_size = user_size
    self.text_size = text_size
    self.image_size = image_size
    self.dummy_user_vector = dummy_user_vector
    self.seed = seed
    self.micro_lambda = micro_lambda
    self.macro_lambda = macro_lambda
    self.max_epoch = max_epoch
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.shuffle = shuffle
    self.log_dir = log_dir
    self.checkpoint_file = checkpoint_file
    self.verbosity = verbosity
    self.save_frequency = save_frequency

    self.start_epoch = self.min_epoch = 0
    self.start_batch_indx = 0
    self.min_valid = 1e10
    self.patience = 100
    self.patience_used = 0
    self.iteration = 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    self.meter_names = ['negative_sampling_loss', 'rc_loss', 'rp_loss', 'pc_loss', 'tree_size_loss', 'max_depth_loss', 'avg_depth_loss', 'target_loss', 'total_loss']
    self.name2indx = {name:indx for indx,name in enumerate(self.meter_names)}

    self.meters = {meter: AverageMeter() for meter in self.meter_names}
    self.batch_meters = {meter: AverageMeter() for meter in self.meter_names}

    self.train_losses = torch.zeros(len(self.meter_names), self.max_epoch)
    self.valid_losses = torch.zeros(len(self.meter_names), self.max_epoch)

    self.checkpoint_file = checkpoint_file
    if checkpoint_file:
      filepath_exists(checkpoint_file)

      self.load_from_ckpt(checkpoint_file)


    if log_dir:
      path_exists(log_dir)

      logs = sorted(glob.glob("%s.*" % log_dir))
      if logs:
        file, suffix = os.path.splitext(logs[-1])
        number = int(suffix[1:])
        log_dir += ".%.5d" % (number+1)
      else:
        log_dir += ".%.5d" % 1
      path_exists(log_dir)

      self.writer = SummaryWriter(log_dir)
      print("logging to %s" % log_dir)
    else:
      self.writer = None

  def write_meters_to_tensorboard(self, meters, name, step):
    for meter in meters:
      self.writer.add_scalar("%s/%s" % (name, meter), meters[meter].average, step)
    # tqdm.write("wrote to tensorboardX for %s" % name)

  def train(self, train_data_files, train_image_files, train_text_files, train_label_files, valid_data_files, valid_image_files, valid_text_files, valid_label_files):

    train_loss = self.train_losses[self.name2indx['total_loss']][self.start_epoch]
    valid_loss = self.valid_losses[self.name2indx['total_loss']][self.start_epoch]

    pbar = tqdm(range(self.start_epoch, self.max_epoch))

    pbar.set_description("Epoch %d. Train loss=%.2f, Valid loss = %.2f" % (self.start_epoch, train_loss, valid_loss))

    ntrain_files = len(train_data_files)
    nvalid_files = len(valid_data_files)

    for epoch in pbar:
      self.current_epoch = epoch
      note = ""
      self.model.train()
      self.model.zero_grad()
      torch.cuda.empty_cache()
      train_loss = self.train_epoch([train_data_files[epoch % ntrain_files]], [train_image_files[epoch % ntrain_files]], [train_text_files[epoch % ntrain_files]], [train_label_files[epoch % ntrain_files]], backprop=True)

      for indx, name in enumerate(self.meter_names):
        self.train_losses[indx][epoch] = self.meters[name].average

      if self.writer:
        self.write_meters_to_tensorboard(self.meters, "train", epoch)
      for meter in self.meters: self.meters[meter].reset()


      self.model.eval()
      self.model.zero_grad()
      torch.cuda.empty_cache()
      with torch.no_grad():
        valid_loss = self.train_epoch([valid_data_files[epoch % nvalid_files]], [valid_image_files[epoch % nvalid_files]], [valid_text_files[epoch % nvalid_files]], [valid_label_files[epoch % nvalid_files]], backprop=False)
      for indx, name in enumerate(self.meter_names):
        self.valid_losses[indx][epoch] = self.meters[name].average

      if self.writer:
        self.write_meters_to_tensorboard(self.meters, "valid", epoch)
      for meter in self.meters: self.meters[meter].reset()


      if valid_loss < self.min_valid:
        self.min_valid = valid_loss
        self.min_epoch = epoch
        self.patience_used = 0
        if self.checkpoint_file: self.save_to_ckpt(self.checkpoint_file, epoch=True)
        note = "NEW MINIMUM"
      else:
        self.patience_used += 1
        note = "patience increased to %d" % self.patience_used


      if epoch % 5 == 1:
        if self.checkpoint_file: self.save_to_ckpt(self.checkpoint_file, iteration=True)
      # if self.patience_used >= self.patience:
        # print("patience exceeded. training finished")
        # break

      pbar.set_description("Epoch %d. Train loss=%.2f, Valid loss = %.2f. %s" % (epoch, train_loss, valid_loss, note))

  def train_epoch(self, data_files, image_files, text_files, label_files, backprop=False):

    pbar = tqdm(zip(data_files, image_files, text_files, label_files), total=len(data_files))
    pbar.set_description("iter=0. target: 0, follow: 0, recur: 0")

    total_batch_indx = 0
    epoch_loss = 0
    for file_indx, (data_file, image_file, text_file, label_file) in enumerate(pbar):
      dataset = TwitterDatasetChunk(
        data_file=data_file,
        image_file=image_file,
        text_file=text_file,
        label_file=label_file,
        key=self.key,
        data_header=self.data_header,
        label_header=self.label_header,
        user_size=self.user_size,
        text_size=self.text_size,
        image_size=self.image_size,
        dummy_user_vector=self.dummy_user_vector
        )
      if not len(dataset.data):
        # empty dataset for some reason? continue
        continue
      dataloader = DataLoader(dataset, batch_size=self.batch_size,
        shuffle=self.shuffle, num_workers=self.num_workers)

      for batch_indx, batch in enumerate(dataloader):
        if batch_indx == 0:
          # initalize things at first batch
          # batch_0 = batch
          batch_1 = batch
        else:
          # if total_batch_indx < self.start_batch_indx:
          #   pass
          #   # total_batch_indx += 1
          #   # continue
          # else:
          #   self.start_batch_indx = total_batch_indx
          # otherwise act regularly
          batch_2 = batch
          if self.verbosity > 1: tqdm.write("\n------------\noutside: batch loaded")
          # get loss
          self.model.zero_grad()
          total_loss = self.train_instance(train_batch=batch_1, random_batch=batch_2, backprop=backprop)
          if self.verbosity > 1: tqdm.write("outside: loss computed")
          epoch_loss += total_loss

          # set next batch_1 to current batch_2 and work on that
          batch_1 = batch_2

          # if self.iteration % self.save_frequency == 1 and backprop:
            # if self.checkpoint_file: self.save_to_ckpt(self.checkpoint_file, self.iteration)
          if backprop:
            if self.writer: 
              self.write_meters_to_tensorboard(self.batch_meters, "batch-wise", self.iteration)
              # uncomment line below to store individual batch loss.
              # if commented, stores average over __long__ horizon
              for meter in self.batch_meters: self.batch_meters[meter].reset()


        # ----- Set pbar/tqdm description --- 
        pbar.set_description("iter=%d. target: %.2f, follow: %.2f, recur: %.2f" % (
          self.iteration,
          self.meters['target_loss'].average,
          self.meters['negative_sampling_loss'].average,
          self.meters['rc_loss'].average + self.meters['rp_loss'].average + self.meters['pc_loss'].average
          ))
        total_batch_indx += 1


      dataset.close()
    # for last instance, use batch 0 as the "random" batch
    # epoch_loss += self.train_instance(train_batch=batch_2, random_batch=batch_0)

    if not total_batch_indx:
      return 0
    return epoch_loss/total_batch_indx

  def train_instance(self, train_batch, random_batch, backprop=False):
    # ---- Get data for computing outputs ---
    r_vector, p_vector, c_vector, rc_length, text_data, image_data, tree_size, max_depth, avg_depth = train_batch
    r_vector_other, p_vector_other, c_vector_other, _, _, _, _, _, _ = random_batch

    # NOTE: the batches may be of different lengths so repeat the smaller one until this match. this is for positive/negative sampling so it's okay to repeat.
    # FIXME: more elegant solution?
    if self.verbosity > 1: tqdm.write("\tmatching sizes starting")
    if len(r_vector) != len(r_vector_other): return 0 # SKIP!!
    # r_vector, r_vector_other = match_sizes(r_vector, r_vector_other)
    # p_vector, p_vector_other = match_sizes(p_vector, p_vector_other)
    # c_vector, c_vector_other = match_sizes(c_vector, c_vector_other)
    if self.verbosity > 1: tqdm.write("\tmatching sizes done")

    # ---- Put data on device (CPU OR GPU) ---
    r_vector = r_vector.to(self.device)
    p_vector = p_vector.to(self.device)
    c_vector = c_vector.to(self.device)
    rc_length = rc_length.to(self.device)
    text_data = text_data.to(self.device) if tensor_is_set(text_data) else text_data
    image_data = image_data.to(self.device) if tensor_is_set(image_data) else image_data
    tree_size = tree_size.to(self.device)
    max_depth = max_depth.to(self.device)
    avg_depth = avg_depth.to(self.device)
    r_vector_other = r_vector_other.to(self.device)
    p_vector_other = p_vector_other.to(self.device)
    c_vector_other = c_vector_other.to(self.device)
    if self.verbosity > 1: tqdm.write("\tput data on device")

    # ---- Compute outputs ---
    p_followed_true, p_followed_false, p_value, c_value, r_value, pred_tree_size, pred_max_depth, pred_avg_depth = self.model(r_vector, p_vector, c_vector, r_vector_other, p_vector_other, c_vector_other, image_data, text_data)
    if self.verbosity > 1: tqdm.write("\trun model over data")

    # ---- Compute Losses ---
    p_follow_loss = negative_sampling_loss(p_followed_true, p_followed_false)
    rc_loss, rp_loss, pc_loss = depth_loss(r_value, p_value, c_value, rc_length.float())
    recursive_loss = rc_loss + rp_loss + pc_loss

    tree_size_loss = F.mse_loss(pred_tree_size, tree_size.float().log())
    max_depth_loss = F.mse_loss(pred_max_depth, max_depth.float().log())
    avg_depth_loss = F.mse_loss(pred_avg_depth, avg_depth.float().log())
    target_loss = tree_size_loss + max_depth_loss + avg_depth_loss

    if (target_loss!=target_loss).any():
      print("target_loss has nan")
      import ipdb; ipdb.set_trace()

    if (p_follow_loss!=p_follow_loss).any():
      print("p_follow_loss has nan")
      import ipdb; ipdb.set_trace()

    if (recursive_loss!=recursive_loss).any():
      print("recursive_loss has nan")
      import ipdb; ipdb.set_trace()
    total_loss = target_loss + self.micro_lambda*p_follow_loss + self.macro_lambda*recursive_loss
    if self.verbosity > 1: tqdm.write("\tcomputed losses")

    # ---- Backprop ---
    if backprop:
      self.optimizer.zero_grad()
      total_loss.backward()
      self.optimizer.step()
      self.iteration += 1
    if self.verbosity > 1: tqdm.write("\tbackprop")

    # ---- Record Losses ---
    batch_len = r_vector.shape[0]
    self.record_losses(self.meters, batch_len, p_follow_loss, rc_loss,rp_loss, pc_loss, tree_size_loss, max_depth_loss, avg_depth_loss, target_loss, total_loss)
    self.record_losses(self.batch_meters, batch_len, p_follow_loss, rc_loss,rp_loss, pc_loss, tree_size_loss, max_depth_loss, avg_depth_loss, target_loss, total_loss)

    if self.verbosity > 1: tqdm.write("\trecord losses ")
    return total_loss

  def record_losses(self, meters, batch_len, p_follow_loss, rc_loss,rp_loss, pc_loss, tree_size_loss, max_depth_loss, avg_depth_loss, target_loss, total_loss):
    meters['negative_sampling_loss'].update(p_follow_loss.item(), batch_len)
    meters['rc_loss'].update(rc_loss.item(), batch_len)
    meters['rp_loss'].update(rp_loss.item(), batch_len)
    meters['pc_loss'].update(pc_loss.item(), batch_len)
    meters['tree_size_loss'].update(tree_size_loss.item(), batch_len)
    meters['max_depth_loss'].update(max_depth_loss.item(), batch_len)
    meters['avg_depth_loss'].update(avg_depth_loss.item(), batch_len)
    meters['target_loss'].update(target_loss.item(), batch_len)
    meters['total_loss'].update(total_loss.item(), batch_len)

  def load_from_ckpt(self, checkpoint_file):
    file, suffix=os.path.splitext(checkpoint_file)

    iter_checkpoints = sorted(glob.glob("%s_iter*" % file))
    best_epoch_checkpoints = sorted(glob.glob("%s_epoch*" % file))

    if not len(best_epoch_checkpoints) and not len(iter_checkpoints): return
    elif not len(best_epoch_checkpoints):
      checkpoint_file = iter_checkpoints[-1]
    elif not len(iter_checkpoints):
      checkpoint_file = best_epoch_checkpoints[-1]
    else:
      # both avaialble
      iter_checkpoint_file = iter_checkpoints[-1]
      best_epoch_checkpoint_file = best_epoch_checkpoints[-1]

      # pick the latest of the two
      if os.path.getctime(best_epoch_checkpoint_file) >= os.path.getctime(iter_checkpoint_file):
        checkpoint_file = best_epoch_checkpoint_file
      else:
        checkpoint_file = iter_checkpoint_file


    checkpoint = torch.load(checkpoint_file)

    self.start_epoch = checkpoint['current_epoch'] + 1
    self.min_epoch = checkpoint['min_epoch']
    self.min_valid = checkpoint['min_valid']
    self.patience_used = checkpoint['patience_used']
    self.iteration = checkpoint['iteration']
    self.start_batch_indx = checkpoint['start_batch_indx']
    self.train_losses = checkpoint['train_losses']
    self.valid_losses = checkpoint['valid_losses']
    self.model.load_state_dict(checkpoint['model'])
    self.optimizer.load_state_dict(checkpoint['optimizer'])

    # for name in self.meters:
    #   self.meters[name].update(checkpoint[name], checkpoint["count"])

    torch.manual_seed(checkpoint['seed'])
    np.random.seed(checkpoint['seed'])

  def save_to_ckpt(self, checkpoint_file, iteration=None, epoch=None):

    file, suffix=os.path.splitext(checkpoint_file)
    if iteration:
      checkpoint_file = "%s_iter_%.9d%s" % (file, iteration, suffix)

    if epoch:
      checkpoint_file = "%s_epoch_%.5d_BEST%s" % (file, epoch, suffix)

    # to_save = {name: self.meters[name].average for name in self.meters}
    # to_save['count'] = next(iter(self.meters.items()))[1].count
    to_save = {}
    to_save.update({
            'min_epoch': self.min_epoch,
            'current_epoch': self.current_epoch,
            'min_valid': self.min_valid,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'seed': self.seed,
            'start_batch_indx': self.start_batch_indx,
            'iteration': self.iteration,
            'patience_used' : self.patience_used,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'loss_names': self.meter_names,
          })
    torch.save(to_save, checkpoint_file)
    print("Saving %s" % checkpoint_file)

def main():
  from pprint import pprint
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  args_to_print = {name:args[name] for name in args if not "files" in name}
  pprint(args_to_print)
  pprint(unknown)

  if args['data_header']:
    with open(args['data_header']) as f:
      data_header = f.readlines()[0].strip().split(",")
  if args['label_header']:
    with open(args['label_header']) as f:
      label_header = f.readlines()[0].strip().split(",")

  train_data_files=args['train_data_files']
  train_image_files=args['train_image_files']
  train_text_files=args['train_text_files']
  train_label_files=args['train_label_files']

  valid_data_files=args['valid_data_files']
  valid_image_files=args['valid_image_files']
  valid_text_files=args['valid_text_files']
  valid_label_files=args['valid_label_files']

  key = args['key']
  seed = args['seed']

  user_size = args['user_size']
  text_size = args['text_size']
  image_size = args['image_size']
  dummy_user_vector = args['dummy_user_vector']
  shuffle = args['shuffle']
  batch_size = args['batch_size']
  num_workers = args['num_workers']

  micro_lambda=args['micro_lambda']
  macro_lambda=args['macro_lambda']
  max_epoch=args['epochs']
  log_dir=args['log_dir']
  checkpoint_file=args['checkpoint']
  verbosity=args['verbosity']
  save_frequency=args['save_frequency']

  train_data_files, train_image_files, train_text_files, train_label_files = get_overlapping_data_files(train_data_files, train_image_files, train_text_files, train_label_files)

  valid_data_files, valid_image_files, valid_text_files, valid_label_files = get_overlapping_data_files(valid_data_files, valid_image_files, valid_text_files, valid_label_files)

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

  print(model)

  optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

  trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device=device,
    key=key,
    data_header=data_header,
    label_header=label_header,
    user_size=user_size,
    text_size=text_size,
    image_size=image_size,
    dummy_user_vector=dummy_user_vector,
    seed=seed,
    micro_lambda=micro_lambda,
    macro_lambda=macro_lambda,
    max_epoch=max_epoch,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=shuffle,
    log_dir=log_dir,
    checkpoint_file=checkpoint_file,
    verbosity=verbosity,
    save_frequency=save_frequency,
  )

  trainer.train(train_data_files, train_image_files, train_text_files, train_label_files, valid_data_files, valid_image_files, valid_text_files, valid_label_files)


if __name__ == '__main__':
  main()