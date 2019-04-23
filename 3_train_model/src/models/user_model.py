import yaml
from data.dataloader import TwitterDataloader
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from src.utils import tensor_is_set

from src.models.modules import NeuralNetwork

class UserModel(nn.Module):
  """docstring for UserModel"""
  def __init__(self, user_size, hidden_size, joint_embedding_size):
    super(UserModel, self).__init__()

    self.StartUserEmbedder = NeuralNetwork(user_size, hidden_size, joint_embedding_size, n_hidden=4)

    self.FollowerUserEmbedder = NeuralNetwork(user_size, hidden_size, joint_embedding_size, n_hidden=4)

    self.FollowingPredictor = nn.Sequential(
      NeuralNetwork(joint_embedding_size, hidden_size, 1, n_hidden=1),
      nn.Sigmoid()
    )

    self.TreeSizePrediction = nn.Sequential(
      NeuralNetwork(joint_embedding_size, hidden_size, 1, n_hidden=1)
      )
    self.MaxDepthPrediction = nn.Sequential(
      NeuralNetwork(joint_embedding_size, hidden_size, 1, n_hidden=1)
      )
    self.AvgDepthPrediction = nn.Sequential(
      NeuralNetwork(joint_embedding_size, hidden_size, 1, n_hidden=1)
      )

    self.DepthPrediction = nn.Sequential(
      NeuralNetwork(joint_embedding_size, hidden_size, 1, n_hidden=1)
      )

  def follower_predictions(self, r_embed, p_embed, c_embed, c_vector, r_vector_other, p_vector_other, c_vector_other,):


    # MICROSCOPIC INFO: probability of following. this is done via negative sampling
    p_followed_true = self.FollowingPredictor(
      # element wise product
      torch.mul(p_embed, self.FollowerUserEmbedder(c_vector)
      )
    )

    p_followed_false = []
    p_followed_false.append(self.FollowingPredictor(
      torch.mul(p_embed, self.FollowerUserEmbedder(p_vector_other),
      )))
    p_followed_false.append(self.FollowingPredictor(
      torch.mul(c_embed, self.FollowerUserEmbedder(c_vector_other),
      )))
    p_followed_false.append(self.FollowingPredictor(
      torch.mul(r_embed, self.FollowerUserEmbedder(r_vector_other),
      )))
    
    returning = {k:v for k,v in zip(("p_followed_true", "p_followed_false"),(p_followed_true, torch.cat(p_followed_false)))}


    for k, v in returning.items():
      if (v != v).any():
        print(k)
        import ipdb; ipdb.set_trace()

    return p_followed_true, p_followed_false

  def forward(self, r_vector, p_vector, c_vector, r_vector_other, p_vector_other, c_vector_other):

    r_embed = self.StartUserEmbedder(r_vector)
    p_embed = self.StartUserEmbedder(p_vector)
    c_embed = self.StartUserEmbedder(c_vector)

    # MICROSCOPIC INFO: probability of following. this is done via negative sampling
    p_followed_true, p_followed_false = self.follower_predictions(r_embed, p_embed, c_embed, c_vector, r_vector_other, p_vector_other, c_vector_other)


    # MACROSCOPIC INFO: for depth recursion

    # add exponent to keep value >=0 for numerical stability
    p_value = self.DepthPrediction(p_embed).exp()
    c_value = self.DepthPrediction(c_embed).exp()
    r_value = self.DepthPrediction(r_embed).exp()

    # TARGETs
    tree_size = self.TreeSizePrediction(r_embed)
    max_depth = self.MaxDepthPrediction(r_embed)
    avg_depth = self.AvgDepthPrediction(r_embed)

    returning = {k:v for k,v in zip(("p_value", "c_value", "r_value", "tree_size", "max_depth", "avg_depth"),(p_value, c_value, r_value, tree_size, max_depth, avg_depth))}

    for k, v in returning.items():
      if (v != v).any():
        print(k)
        import ipdb; ipdb.set_trace()

    return p_followed_true, p_followed_false, p_value, c_value, r_value, tree_size, max_depth, avg_depth

def main():
  from pprint import pprint
  from src.config import load_parser

  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  model = UserModel(user_size=args['user_size'],
    hidden_size=args['hidden_size'],
    joint_embedding_size=args['joint_embedding_size'])

if __name__ == '__main__':
  main()
