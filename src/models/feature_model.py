from src.dataloader import TwitterDataloader
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from src.utils import tensor_is_set

class NeuralNetwork(nn.Module):
  """docstring for NeuralNetwork"""
  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralNetwork, self).__init__()
    self.nn = nn.Sequential(
      nn.Linear(input_size, hidden_size), nn.ReLU(),
      nn.Linear(hidden_size, hidden_size), nn.ReLU(),
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

class JointContentEmbedder(object):
  """docstring for JointContentEmbedder"""
  def __init__(self, image_embed_size, text_embed_size, hidden_size, output_size):
    super(JointContentEmbedder, self).__init__()
    self.image_embedder = NeuralNetwork(image_embed_size, hidden_size, output_size)
    self.image_gate = ContentGate(output_size)
    self.text_embedder = NeuralNetwork(text_embed_size, hidden_size, output_size)
    self.text_gate = ContentGate(output_size)

  def forward(self, image_content=None, text_content=None):
    have_text = tensor_is_set(text_content)
    have_image = tensor_is_set(image_content)
    if have_image:
      image_embedding = self.image_embedder(image_content)
      image_gate = self.image_gate(image_embedding)
      image_embedding = image_embedding*image_gate

    if have_text:
      text_embedding = self.text_embedder(text_content)
      text_gate = self.text_gate(text_embedding)
      text_embedding = text_embedding*text_gate

    if have_text and have_image:
      return torch.prod(image_embedding, text_embedding)
    elif have_text: return text_embedding
    elif have_image: return image_embedding
    else:
      raise NotImplementedError("Must give image or text embedding")


class Model(object):
  """docstring for Model"""
  def __init__(self, vocab_size, image_embed_size, text_embed_size, hidden_size, joint_embedding_size):
    super(Model, self).__init__()
    self.ContentEmbedder4Following = JointContentEmbedder(image_embed_size, text_embed_size, hidden_size, joint_embedding_size)

    self.ContentEmbedder4Prediction = JointContentEmbedder(image_embed_size, text_embed_size, hidden_size, joint_embedding_size)
    
    self.StartUserEmbedder = nn.Embedding(vocab_size, joint_embedding_size)
    self.FollowerUserEmbedder = nn.Embedding(vocab_size, joint_embedding_size)

    self.FollowingPredictor = nn.Sequential(
      nn.Linear(joint_embedding_size, 1), nn.Sigmoid()
    )

    self.ViralityPrediction = NeuralNetwork(joint_embedding_size, hidden_size, 1)

    self.DepthPrediction = NeuralNetwork(joint_embedding_size, hidden_size, 1)

  def follower_predictions(self, r_embed, p_embed, c_embed, cid, rid_other, pid_other, cid_other, image, text):
    # MICROSCOPIC INFO: probability of following. this is done via negative sampling
    content_embed = self.ContentEmbedder4Following(image, text)
    p_followed_true = self.FollowingPredictor(torch.prod([
      p_embed,
      self.FollowerUserEmbedder(cid),
      content_embed
      ]))

    p_followed_false = []
    p_followed_false.append(
      `self.FollowingPredictor(torch.prod([
        p_embed,
        self.FollowerUserEmbedder(pid_other),
        content_embed
      ])))
    p_followed_false.append(
      `self.FollowingPredictor(torch.prod([
        c_embed,
        self.FollowerUserEmbedder(cid_other),
        content_embed
      ])))
    p_followed_false.append(
      `self.FollowingPredictor(torch.prod([
        r_embed,
        self.FollowerUserEmbedder(rid_other),
        content_embed
      ])))
    p_followed_false = torch.cat(p_followed_false)
    return p_followed_true, p_followed_false

  def forward(self, rid, pid, cid, rid_other, pid_other, cid_other, image, text):


    p_embed = self.StartUserEmbedder(pid)
    c_embed = self.StartUserEmbedder(cid)
    r_embed = self.StartUserEmbedder(rid)

    # MICROSCOPIC INFO: probability of following. this is done via negative sampling
    p_followed_true, p_followed_false = self.follower_predictions(r_embed, p_embed, c_embed, cid, rid_other, pid_other, cid_other, image, text)


    # MACROSCOPIC INFO: for depth recursion
    content_embed = self.ContentEmbedder4Prediction(image, text)
    p_value = self.DepthPrediction(torch.prod([
      p_embed, content_embed]))
    c_value = self.DepthPrediction(torch.prod([
      c_embed, content_embed]))
    r_value = self.DepthPrediction(torch.prod([
      r_embed, content_embed]))

    # TARGET: viralirty
    target = self.ViralityPrediction(torch.prod([
      r_embed, content_embed]))



def negative_sampling_loss(positive, negative):

  positive_y = torch.ones(positive.shape[0])
  negative_y = torch.zeros(negative.shape[0])

  positive_loss = F.binary_cross_entropy_with_logits(positive, positive_y)
  negative_loss = F.binary_cross_entropy_with_logits(negative, negative_y)

  return positive_loss + negative_loss

if __name__ == '__main__':

  from pprint import pprint
  from src.config import load_parser
  parser = load_parser()
  args, unknown = parser.parse_known_args()
  args = vars(args)
  pprint(args)

  dataloader = TwitterDataloader(args['master_filename'], args['label_filename'], args['image_filename'], args['text_filename'], args['colnames'], args['file_length'], args['batch_size'])

  unique_ids = dataloader.unique_ids()
  model = FeatureModel(vocab_size=len(unique_ids))
