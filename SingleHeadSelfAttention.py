import torch
import torch.nn as nn

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embedding_size=64, head_dim=64):
        self.embedding_size = embedding_size
        self.head_dim = head_dim

        self.W_q = nn.Linear(embedding_size, head_dim, bias=False)
        self.W_k = nn.Linear(embedding_size, head_dim, bias=False)
        self.W_v = nn.Linear(embedding_size, head_dim, bias=False)

    def forward(self, x):
        print(f"x.shape:{x.shape}")
        Q = self.W_q(x)  # [batch_size, seq_len, head_dim]
        K = self.W_k(x)
        V = self.W_v(x)

        return Q, K, V  # just for now
    

if __name__ == "main":
    x = torch.randn(2, 4, 8)  # batch=2, seq_len=4, embedding_size=8
    attn = SingleHeadSelfAttention(embedding_size=8, head_dim=4)
    Q, K, V = attn(x)

    