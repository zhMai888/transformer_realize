import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,n_head, d_model, dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.n_head = n_head  # 头数
        self.d_model = d_model  # 模型维度
        self.d_h = d_model // n_head  # 每个头的维度

        self.w_qkv = nn.Linear(d_model, d_model* 3, bias=False)  # w矩阵
        self.w_out = nn.Linear(d_model, d_model,bias=False)  # 合并输出矩阵

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # 算出Q、K、V
        qkv = self.w_qkv(x).reshape(batch_size, seq_len, 3, self.n_head, self.d_h)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        # 转成 (batch_size, n_head, seq_len, d_h)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 计算注意力
        atten = (q @ k.transpose(-2, -1)) / (self.d_h ** 0.5)
        if mask is not None:
            atten = atten.masked_fill(mask == 0, float('-inf'))

        atten = self.softmax(atten)
        atten = self.dropout(atten)

        # 计算注意力得分
        atten_score = (atten @ v).transpose(1, 2).reshape(batch_size, seq_len, d_model)

        # 合并输出
        output = self.w_out(atten_score)
        return output