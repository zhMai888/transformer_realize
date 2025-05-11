from dataclasses import dataclass
import torch.nn as nn
from Embedding import EmbeddingLayer
from Encoder import Encoder
from Decoder import Decoder

@dataclass
class AttentionConfig:
    max_len: int                # 句子最大长度
    num_embeddings: int         # 词汇表大小
    n_encoder_layers: int = 8   # encoder层数
    n_decoder_layers: int = 8   # decoder层数
    d_model: int = 512          # 模型维度
    n_heads: int = 8            # 注意力头数
    ff_dim: int = 1024          # 前馈网络中间变换维度

    drop_prob: float = 0.1      # dropout概率

class Transformer(nn.Module):
    def __init__(self, config: AttentionConfig):
        super(Transformer,self).__init__()

        self.n_encoder_layers = config.n_encoder_layers
        self.n_decoder_layers = config.n_decoder_layers

        # embedding层
        self.embedding_input = EmbeddingLayer(config.max_len, config.d_model,
                                              config.num_embeddings,config.drop_prob)
        self.embedding_output = EmbeddingLayer(config.max_len, config.d_model,
                                               config.num_embeddings,config.drop_prob)

        # encoder层
        self.encoders = nn.ModuleList([
            Encoder(config.d_model, config.n_heads, config.ff_dim, config.drop_prob)
            for _ in range(config.n_encoder_layers)
        ])

        # decoder层
        self.decoders = nn.ModuleList([
            Decoder(config.d_model, config.n_heads, config.ff_dim, config.drop_prob)
            for _ in range(config.n_decoder_layers)
        ])

        # linear层
        self.linear = nn.Linear(config.d_model, config.num_embeddings)

        # softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, y, out_mask1, out_mask2=None):
        # encoder
        x = self.embedding_input(x)
        for i in range(self.n_encoder_layers):
            x = self.encoders[i](x)

        # decoder
        y = self.embedding_output(y)
        for i in range(self.n_decoder_layers):
            y = self.decoders[i](y, x, out_mask1, out_mask2)

        # linear
        y = self.linear(y)

        # softmax
        y = self.softmax(y)

        return y



