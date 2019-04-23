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
