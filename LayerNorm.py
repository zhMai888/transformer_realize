import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(d_model))  # 偏移参数
        self.eps = eps  # 防止除以0

    def forward(self, x):
        # 均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)

        # 归一化
        x_normalized = (x - mean) / torch.sqrt(std + self.eps)

        # 进行缩放和偏移
        return self.gamma * x_normalized + self.beta

if __name__ == '__main__':
    layer_norm = LayerNorm(512)
    x = torch.randn(2, 10, 512)
    output = layer_norm(x)
    print(output.shape)  # 应该是 [2, 10, 512]

