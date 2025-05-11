import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from LayerNorm import LayerNorm
from FeedForward import FeedForward
from Encoder import Encoder


class Decoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_prob=0.1):
        super(Decoder, self).__init__()
        self.multiHeadAttention1 = MultiHeadAttention(n_head, d_model, drop_prob)
        self.norm1 = LayerNorm(d_model)

        self.multiHeadAttention2 = MultiHeadAttention(n_head, d_model, drop_prob)
        self.norm2 = LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff, drop_prob)
        self.norm3 = LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, encoder_output, mask1, mask2=None):
        attn_output = self.multiHeadAttention1(x, mask1)
        x = self.norm1(x + self.dropout(attn_output))

        x = x + encoder_output
        cross_attn_output = self.multiHeadAttention2(x, mask2)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

if __name__ == '__main__':
    d_model = 512
    n_head = 8
    d_ff = 2048
    drop_prob = 0.1
    batch_size = 2
    seq_length = 10

    decoder = Decoder(d_model, n_head, d_ff, drop_prob)
    encoder_output = Encoder(d_model, n_head, d_ff, drop_prob)

    x = torch.randn(2, 10, d_model)
    output1 = encoder_output(x)
    mask1 = torch.tril(torch.ones(seq_length, seq_length)).bool()
    mask2 = torch.ones(seq_length, seq_length).bool()

    output2 = decoder(x, output1, mask1, mask2)
    print(output2.shape)        # [2, 10, 512]
