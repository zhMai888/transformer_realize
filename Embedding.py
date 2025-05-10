import torch
import torch.nn as nn
import math

# token信息的embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(TokenEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings



# token位置的embedding
class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.embedding = torch.zeros(max_len, embedding_dim)
        self.embedding.requires_grad = False
        self.pos = torch.arange(0, max_len).unsqueeze(1).float()
        self.div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        self.embedding[:,0::2] = torch.sin(self.pos * self.div_term)
        self.embedding[:,1::2] = torch.cos(self.pos * self.div_term)

    def forward(self, x):
        return self.embedding[:, :x.size(1)]


class EmbeddingLayer(nn.Module):
    def __init__(self, max_len, embedding_dim,num_embeddings,drop_prob=0.1):
        super(EmbeddingLayer,self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.drop_prob = drop_prob
        self.token_embedding = TokenEmbedding(num_embeddings, embedding_dim)
        self.pos_embedding = PositionEmbedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(p=self.drop_prob)

    def forward(self, x):
        tokenEmbedding = self.token_embedding(x)
        positionEmbedding = self.pos_embedding(x)
        return self.dropout(tokenEmbedding + positionEmbedding)