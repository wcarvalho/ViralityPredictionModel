from src.dataloader import TwitterDataloader
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

def shuffle_batch(t):
  idx = torch.randperm(t.shape[0])
  return t[idx]

class UserEmbedder(nn.Module):
  """docstring for UserEmbedder"""
  def __init__(self, vocab_size, embedding_size=256, hidden_size=256):
    super(UserEmbedder, self).__init__()
    self.vocab_size = vocab_size
    self.embedder = nn.Embedding(self.vocab_size, embedding_size)  # 2 words in vocab, 5 dimensional embeddings
    self.nn = nn.Sequential(
      nn.Linear(embedding_size, hidden_size), nn.ReLU(),
    )

  def forward(self, user_id):
    return self.nn(self.embedder(user_id))

class BasicModel(nn.Module):
  """docstring for BasicModel"""
  def __init__(self, vocab_size, embedding_size=256, hidden_size=256):
    super(BasicModel, self).__init__()
    self.vocab_size = vocab_size
    self.embedder = UserEmbedder(vocab_size, embedding_size, hidden_size)
    self.p_followed_net = nn.Sequential(
      nn.Linear(2*embedding_size, 256), nn.ReLU(),
      nn.Linear(256, 1), nn.Sigmoid()
    )

    self.target_net = nn.Sequential(
      nn.Linear(embedding_size, 256), nn.ReLU(),
      nn.Linear(256, 1)
    )

    self.value_net = nn.Sequential(
      nn.Linear(embedding_size, 256), nn.ReLU(),
      nn.Linear(256, 1)
    )

  def negative_samples(self, p_embed, p_other_embed, r_embed, r_other_embed, c_embed, c_other_embed):
    p_followed_false = []
    # pick 3 negative samples
    f_embed = torch.cat([p_embed, p_other_embed], dim=-1)
    p_followed_false.append(self.p_followed(f_embed))

    f_embed = torch.cat([r_embed, r_other_embed], dim=-1)
    p_followed_false.append(self.p_followed(f_embed))

    f_embed = torch.cat([c_embed, c_other_embed], dim=-1)
    p_followed_false.append(self.p_followed(f_embed))

    return torch.cat(p_followed_false)

  def forward(self, rid, pid, cid, rid_other, pid_other, cid_other):
    # embedding of users
    p_embed = self.embedder(pid)
    c_embed = self.embedder(cid)
    r_embed = self.embedder(cid)

    r_other_embed = self.embedder(rid_other)
    p_other_embed = self.embedder(pid_other)
    c_other_embed = self.embedder(cid_other)


    # MACROSCOPIC INFO: for depth recursion
    p_value = self.value_net(p_embed)
    c_value = self.value_net(c_embed)
    r_value = self.value_net(r_embed)

    # for computing target value
    target = self.target_net(r_embed)

    # MICROSCOPIC INFO: probability of following. this is done via negative sampling
    # positive sample
    pc_embed = torch.cat([p_embed, c_embed], dim=-1)
    p_followed_true = self.p_followed(pc_embed)
    p_followed_false = self.negative_samples(p_embed, p_other_embed, r_embed, r_other_embed, c_embed, c_other_embed)

    return p_value, c_value, r_value, target, p_followed_true, p_followed_false


def negative_sampling_loss(positive, negative):

  positive_y = torch.ones(positive.shape[0])
  negative_y = torch.zeros(negative.shape[0])

  positive_loss = F.binary_cross_entropy_with_logits(positive, positive_y)
  negative_loss = F.binary_cross_entropy_with_logits(negative, negative_y)

  return positive_loss + negative_loss

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--filename', type=str)
  parser.add_argument('-bs', '--batch-size', type=int, default=128)
  parser.add_argument('-fl', '--file-length', type=int, default=0)
  parser.add_argument('-e', '--epochs', type=int, default=1000)
  # parser.add_argument('-s', '--shuffle', type=int, default=1, choices=[0,1])
  parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
  parser.add_argument('-cn', '--colnames', type=str, default=["r_pid", "r_uid", "r_t", "p_pid", "p_uid", "p_t", "c_pid", "c_uid", "c_t", "text", "data"], nargs='+')
  args, unknown = parser.parse_known_args()
  batch_size = args.batch_size
  args = vars(args)

  dataloader = TwitterDataloader(args['filename'], args['colnames'], args['file_length'], args['batch_size'])

  unique_ids = dataloader.unique_ids()
  id2indx = {uid:indx for indx, uid in enumerate(unique_ids)}

  model = BasicModel(vocab_size=len(unique_ids))
  optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

  for epoch in tqdm(args['epochs']):
    for data in dataloader:

      r_pid = torch.tensor([id2indx[d] for d in data['r_pid']], dtype=torch.long)
      p_pid = torch.tensor([id2indx[d] for d in data['p_pid']], dtype=torch.long)
      c_pid = torch.tensor([id2indx[d] for d in data['c_pid']], dtype=torch.long)

      p_value, c_value, r_value, target, p_followed_true, p_followed_false = model(r_pid, p_pid, c_pid, r_pid, p_pid, c_pid)

      neg_sampling_followed_loss = negative_sampling_loss(p_followed_true.squeeze(), p_followed_false.squeeze())
      break
