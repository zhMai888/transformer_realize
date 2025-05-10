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