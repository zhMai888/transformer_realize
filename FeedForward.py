import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, m_layer, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, m_layer)
        self.linear2 = nn.Linear(m_layer, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

if __name__ == '__main__':
    d_model = 64
    m_layer = 128
    batch_size = 4
    seq_length = 10

    ff = FeedForward(d_model, m_layer)
    x = torch.randn(batch_size, seq_length, d_model)
    output = ff(x)

    print(x.shape)  # [4, 10, 64]
    print(output.shape)  # [4, 10, 64]