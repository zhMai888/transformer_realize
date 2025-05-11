import torch
from Transformer import Transformer,AttentionConfig

config = AttentionConfig(
    max_len=10,          # 最大句子长度
    num_embeddings=100,  # 词汇表大小
    n_encoder_layers=2,  # encoder层数（测试时减少层数以加快速度）
    n_decoder_layers=2,  # decoder层数
    d_model=64,          # 模型维度（测试时使用较小的维度）
    n_heads=4,           # 注意力头数
    ff_dim=128,          # 前馈网络维度
    drop_prob=0.1        # dropout概率
)


model = Transformer(config)

# 生成随机输入数据
batch_size = 4
seq_length = 8

# 源序列
src_tokens = torch.randint(0, config.num_embeddings, (batch_size, seq_length))
# 目标序列
tgt_tokens = torch.randint(0, config.num_embeddings, (batch_size, seq_length))

# 创建mask, 第二个mask为None
tgt_mask = torch.tril(torch.ones(seq_length, seq_length)).bool()

# 训练10轮
for i in range(10):
    model.zero_grad()
    logits = model(src_tokens, tgt_tokens, tgt_mask)
    # 计算损失(我的Transformer最后带了一层softmax,所以这里用nll_loss)
    loss = torch.nn.functional.nll_loss(logits.view(-1, config.num_embeddings), tgt_tokens.view(-1))
    loss.backward()
    # 更新参数
    for param in model.parameters():
        param.data -= 0.01 * param.grad.data
        param.grad.data.zero_()
    print(f"Epoch {i+1}, Loss: {loss.item()}")