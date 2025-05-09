import torch
import torch.nn as nn
import numpy as np

# token信息的embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(TokenEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings



# token位置的embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.embedding = torch.zeros(max_len, embedding_dim)
        self.embedding.requires_grad = False
        self.pos = torch.arange(0, max_len).unsqueeze(1).float()
        self.embedding[:,0::2] = torch.sin(self.pos * 2 * np.pi / self.embedding_dim)
        self.embedding[:,1::2] = torch.cos(self.pos * 2 * np.pi / self.embedding_dim)

    def forward(self, x):
        return self.embedding[:, :x.size(1)]