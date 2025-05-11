import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from LayerNorm import LayerNorm
from FeedForward import FeedForward

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, ff_dim, dropout=0.1):
        super(Encoder,self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.multiHeadAttention = MultiHeadAttention(n_head, d_model, dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.FeedForward = FeedForward(d_model, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mul = self.multiHeadAttention(x)
        x = self.layer_norm1(x + self.dropout(mul))
        ff = self.FeedForward(x)
        x = self.layer_norm2(x + self.dropout(ff))
        return x

if __name__ == '__main__':
    d_model = 512
    n_head = 8
    ff_dim = 2048
    dropout = 0.1

    encoder = Encoder(d_model, n_head, ff_dim, dropout)
    x = torch.randn(2, 10, d_model)
    output = encoder(x)
    print(output.shape)  # [2, 10, 512]