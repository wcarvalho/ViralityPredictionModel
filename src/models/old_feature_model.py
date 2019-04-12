
class FeatureModel(nn.Module):
  """docstring for FeatureModel"""
  def __init__(self, vocab_size, embedding_size=256, content_embed_size=256):
    super(FeatureModel, self).__init__()
    self.vocab_size = vocab_size
    self.embedder = UserEmbedder(vocab_size, embedding_size, content_embed_size)
    self.p_followed_net = nn.Sequential(
      nn.Linear(512, 1), nn.Sigmoid()
    )

    self.target_net = nn.Sequential(
      nn.Linear(256, 1)
    )

    self.value_net = nn.Sequential(
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

  def forward(self, rid, pid, cid, rid_other, pid_other, cid_other, content_embed):

    # embedding of users
    p_embed = self.embedder(pid, content_embed)
    c_embed = self.embedder(cid, content_embed)
    r_embed = self.embedder(cid, content_embed)

    r_other_embed = self.embedder(rid_other, content_embed)
    p_other_embed = self.embedder(pid_other, content_embed)
    c_other_embed = self.embedder(cid_other, content_embed)


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