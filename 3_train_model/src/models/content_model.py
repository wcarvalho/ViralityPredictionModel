import yaml
from data.dataloader import TwitterDataloader
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from src.utils import tensor_is_set

from src.models.modules import NeuralNetwork, ContentGate, JointContentEmbedder

class ContentModel(nn.Module):
  """docstring for ContentModel"""
  def __init__(self, image_embed_size, text_embed_size, hidden_size, joint_embedding_size):
    super(ContentModel, self).__init__()

    self.ContentEmbedder4Prediction = JointContentEmbedder(image_embed_size, text_embed_size, hidden_size, joint_embedding_size, n_hidden=4)


    self.TreeSizePrediction = nn.Sequential(
      NeuralNetwork(joint_embedding_size, hidden_size, 1, n_hidden=1)
      )

    self.MaxDepthPrediction = nn.Sequential(
      NeuralNetwork(joint_embedding_size, hidden_size, 1, n_hidden=1)
      )

    self.AvgDepthPrediction = nn.Sequential(
      NeuralNetwork(joint_embedding_size, hidden_size, 1, n_hidden=1)
      )


  def forward(self, image, text):
    # in case there are nans, set values to 0!!!
    text[(text != text)] = 0  
    image[(image != image)] = 0  

    # TARGETs
    content_embed = self.ContentEmbedder4Prediction(image, text)
    tree_size = self.TreeSizePrediction(content_embed)
    max_depth = self.MaxDepthPrediction(content_embed)
    avg_depth = self.AvgDepthPrediction(content_embed)

    returning = {k:v for k,v in zip(("tree_size", "max_depth", "avg_depth"),(tree_size, max_depth, avg_depth))}

    for k, v in returning.items():
      if (v != v).any():
        print(k)
        import ipdb; ipdb.set_trace()


    return tree_size, max_depth, avg_depth

def main():
  from pprint import pprint
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  model = ContentModel(
    image_embed_size=args['image_size'],
    text_embed_size=args['text_size'],
    hidden_size=args['hidden_size'],
    joint_embedding_size=args['joint_embedding_size'])

if __name__ == '__main__':
  main()
