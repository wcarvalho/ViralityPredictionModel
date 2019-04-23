import yaml
from data.dataloader import TwitterDataloader
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from src.utils import tensor_is_set


class NeuralNetwork(nn.Module):
  """docstring for NeuralNetwork"""
  def __init__(self, input_size, hidden_size, output_size, n_hidden=1, keep=.8):
    super(NeuralNetwork, self).__init__()
    hidden_layers = [nn.Dropout(p=1.0-keep), nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()]*n_hidden
    self.nn = nn.Sequential(
      nn.Linear(input_size, hidden_size), nn.ReLU(),
      *hidden_layers,
      nn.Linear(hidden_size, output_size)
    )
  
  def forward(self, x): return self.nn(x)

class ContentGate(nn.Module):
  """docstring for ContentGate"""
  def __init__(self, input_size):
    super(ContentGate, self).__init__()
    self.nn = nn.Sequential(
      nn.Linear(input_size, input_size), nn.Sigmoid(),
    )
  def forward(self, x): return self.nn(x)

class JointContentEmbedder(nn.Module):
  """docstring for JointContentEmbedder"""
  def __init__(self, image_embed_size, text_embed_size, hidden_size, output_size, n_hidden):
    super(JointContentEmbedder, self).__init__()
    self.image_embedder = nn.Sequential(
      NeuralNetwork(image_embed_size, hidden_size, output_size, n_hidden), nn.Sigmoid()
      )
    # self.image_gate = ContentGate(output_size)
    self.text_embedder = nn.Sequential(
      NeuralNetwork(text_embed_size, hidden_size, output_size, n_hidden), nn.Sigmoid()
      )
    # self.text_gate = ContentGate(output_size)

  def forward(self, image_content=None, text_content=None):
    have_text = tensor_is_set(text_content)
    have_image = tensor_is_set(image_content)

    if have_image:
      image_embedding = self.image_embedder(image_content)
      # image_gate = self.image_gate(image_embedding)
      # image_embedding = image_embedding*image_gate

    if have_text:
      text_embedding = self.text_embedder(text_content)
      # text_gate = self.text_gate(text_embedding)
      # text_embedding = text_embedding*text_gate

    if have_text and have_image:
      return torch.mul(image_embedding, text_embedding)
    elif have_text: return text_embedding
    elif have_image: return image_embedding
    else:
      raise NotImplementedError("Must give image or text embedding")

class FeatureModel(nn.Module):
  """docstring for FeatureModel"""
  def __init__(self, user_size, image_embed_size, text_embed_size, hidden_size, joint_embedding_size):
    super(FeatureModel, self).__init__()
    self.ContentEmbedder4Following = JointContentEmbedder(image_embed_size, text_embed_size, hidden_size, joint_embedding_size, n_hidden=4)

    self.ContentEmbedder4Prediction = JointContentEmbedder(image_embed_size, text_embed_size, hidden_size, joint_embedding_size, n_hidden=4)
    
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

  def follower_predictions(self, r_embed, p_embed, c_embed, c_vector, r_vector_other, p_vector_other, c_vector_other, image, text):


    # MICROSCOPIC INFO: probability of following. this is done via negative sampling
    content_embed = self.ContentEmbedder4Following(image, text)
    p_followed_true = self.FollowingPredictor(
      # element wise product
      torch.mul(p_embed, 
        content_embed.mul(self.FollowerUserEmbedder(c_vector))
      )
    )

    p_followed_false = []
    p_followed_false.append(self.FollowingPredictor(
      torch.mul(p_embed,
        content_embed.mul(self.FollowerUserEmbedder(p_vector_other)),
      )))
    p_followed_false.append(self.FollowingPredictor(
      torch.mul(c_embed,
        content_embed.mul(self.FollowerUserEmbedder(c_vector_other)),
      )))
    p_followed_false.append(self.FollowingPredictor(
      torch.mul(r_embed,
        content_embed.mul(self.FollowerUserEmbedder(r_vector_other)),
      )))
    
    returning = {k:v for k,v in zip(("p_followed_true", "p_followed_false"),(p_followed_true, torch.cat(p_followed_false)))}



    for k, v in returning.items():
      if (v != v).any():
        print(k)
        import ipdb; ipdb.set_trace()

    return p_followed_true, p_followed_false

  def forward(self, r_vector, p_vector, c_vector, r_vector_other, p_vector_other, c_vector_other, image, text):
    # in case there are nans, set values to 0!!!
    text[(text != text)] = 0  
    image[(image != image)] = 0  

    r_embed = self.StartUserEmbedder(r_vector)
    p_embed = self.StartUserEmbedder(p_vector)
    c_embed = self.StartUserEmbedder(c_vector)

    # MICROSCOPIC INFO: probability of following. this is done via negative sampling
    p_followed_true, p_followed_false = self.follower_predictions(r_embed, p_embed, c_embed, c_vector, r_vector_other, p_vector_other, c_vector_other, image, text)


    # MACROSCOPIC INFO: for depth recursion
    content_embed = self.ContentEmbedder4Prediction(image, text)
    # add exponent to keep value >=1 for numerical stability
    p_value = self.DepthPrediction(torch.mul(
      p_embed, content_embed))
    c_value = self.DepthPrediction(torch.mul(
      c_embed, content_embed))
    r_value = self.DepthPrediction(torch.mul(
      r_embed, content_embed))

    # TARGETs
    target_input = torch.mul(r_embed, content_embed)
    tree_size = self.TreeSizePrediction(target_input)
    max_depth = self.MaxDepthPrediction(target_input)
    avg_depth = self.AvgDepthPrediction(target_input)

    returning = {k:v for k,v in zip(("p_value", "c_value", "r_value", "tree_size", "max_depth", "avg_depth"),(p_value, c_value, r_value, tree_size, max_depth, avg_depth))}
    import ipdb; ipdb.set_trace()
    for k, v in returning.items():
      try:
        if (v != v).any():
          print(k)
          import ipdb; ipdb.set_trace()
      except Exception as e:
        import ipdb; ipdb.set_trace()
        raise e

    return p_followed_true, p_followed_false, p_value, c_value, r_value, tree_size, max_depth, avg_depth

def main():
  from pprint import pprint
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)

  model = FeatureModel(user_size=args['user_size'],
    image_embed_size=args['image_size'],
    text_embed_size=args['text_size'],
    hidden_size=args['hidden_size'],
    joint_embedding_size=args['joint_embedding_size'])

if __name__ == '__main__':
  main()
