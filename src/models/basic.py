from src.dataloader import TwitterDataloader
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

def shuffle_batch(t):
  idx = torch.randperm(t.shape[0])
  return t[idx]

class UserEmbedder(nn.Module):
  """docstring for UserEmbedder"""
  def __init__(self, vocab_size, embedding_size=256):
    super(UserEmbedder, self).__init__()
    self.vocab_size = vocab_size
    self.emebdding = nn.Embedding(self.vocab_size, embedding_size)  # 2 words in vocab, 5 dimensional embeddings

  def forward(self, user_id):
    return self.emebdding(user_id)

class BasicModel(nn.Module):
  """docstring for BasicModel"""
  def __init__(self, vocab_size, embedding_size=256):
    super(BasicModel, self).__init__()
    self.vocab_size = vocab_size
    self.embedder = UserEmbedder(vocab_size, embedding_size)
    self.prob_followed = nn.Sequential(
      nn.Linear(2*embedding_size, 256), nn.ReLU(),
      nn.Linear(256, 256), nn.ReLU(),
      nn.Linear(256, 1), nn.Sigmoid()
    )

  def p_followed(self, p_embed, c_embed):
    pc_embed = torch.cat([p_embed, c_embed], dim=-1)
    return self.prob_followed(pc_embed)


  def forward(self, rid, pid, cid):
    p_embed = self.embedder(pid)
    c_embed = self.embedder(cid)
    r_embed = self.embedder(cid)


    p_followed_true = self.p_followed(p_embed, c_embed)
    
    p_followed_false = []
    # p_followed_false.append(self.p_followed(c_embed, p_embed)) # might this be true?
    p_followed_false.append(self.p_followed(p_embed, r_embed))
    p_followed_false.append(self.p_followed(c_embed, r_embed))
    p_followed_false.append(self.p_followed(p_embed, shuffle_batch(c_embed)))

    p_followed_false = torch.cat(p_followed_false)

    return p_followed_true, p_followed_false

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

  for data in dataloader:

    r_pid = torch.tensor([id2indx[d] for d in data['r_pid']], dtype=torch.long)
    p_pid = torch.tensor([id2indx[d] for d in data['p_pid']], dtype=torch.long)
    c_pid = torch.tensor([id2indx[d] for d in data['c_pid']], dtype=torch.long)

    p_followed_true, p_followed_false = model(r_pid, p_pid, c_pid)

    neg_sampling_followed_loss = negative_sampling_loss(p_followed_true.squeeze(), p_followed_false.squeeze())
    # c_pid = data['c_pid']
    # r_uid = data['r_uid']
    # r_t = data['r_t']
    # p_uid = data['p_uid']
    # p_t = data['p_t']
    # c_uid = data['c_uid']
    # c_t = data['c_t']
    break
