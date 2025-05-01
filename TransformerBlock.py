from dataset import *
import torch.nn as nn
import torch.nn.functional as F
from ManualLayerNorm import ManualLayerNorm
from SingleHeadSelfAttention import SingleHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, input_embedding_size=64) -> None:
        super().__init__()
        
        self.ln1 = ManualLayerNorm(input_embedding_size)
        self.attn = SingleHeadSelfAttention(embedding_size=input_embedding_size, head_dim=input_embedding_size)
        self.ln2 = ManualLayerNorm(input_embedding_size)

        self.mlp = nn.Sequential(
            nn.Linear(input_embedding_size, input_embedding_size * 4),
            nn.GELU(),
            nn.Linear(input_embedding_size * 4, input_embedding_size)
        )

    def forward(self, x):
        
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm)
        x = x + attn_out  # Residual connection

        # Feedforward with residual
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out  # Residual connection

        return x